"""SPEC §24 — Arweave-plane backup event tarball schema.

One tarball per (event, component) — components are personality, timechain,
soul. Each tarball carries:
  - `__event_metadata.json` — per-file index keyed by arc_name (file path
    inside the tarball), with the diff_dict subset needed to reconstruct
    that file via diff_encoders.apply_diff() at restore time.
  - `files/<arc_name>` — the patch_bytes for each per-file diff payload.
    For diff_mode="full" / "incremental" / "tail" the bytes are whatever
    the encoder produced; the encoder name in the metadata routes the
    apply step correctly.

This schema is the bridge between:
  - the §24.5 diff_encoders (Phase 3, SHIPPED), which produce per-file
    diff_dicts during the upload-side
  - the §24 UnifiedManifest (Phase 2, SHIPPED), which records the
    tarball's tx_id + merkle_root on the manifest event
  - the §3.1 restore protocol (Phase 6, this file's caller), which walks
    the manifest, fetches tarballs from Arweave, and unpacks them to
    feed apply_diff() per file.

Schema is intentionally flat (no nested manifests) so a Maker inspecting
a tarball via `tar tzf` sees the file list immediately.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tarfile
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import zstandard

logger = logging.getLogger(__name__)


# Schema version for the embedded __event_metadata.json. Bump only when
# the per-file entry shape changes incompatibly (older readers can't
# decode). Additive fields are fine without bumping.
EVENT_TARBALL_SCHEMA_VERSION = 1

# Reserved member name for the inner metadata index. Must NOT collide
# with any arc_name in the files/ tree.
EVENT_METADATA_MEMBER = "__event_metadata.json"

# All per-file payloads live under this prefix inside the tarball, so the
# metadata member sits alongside cleanly and the file tree is easy to
# inspect with `tar tzf`.
FILES_PREFIX = "files/"

# Phase 5 chunk 5F (2026-05-19) — gzip-9 → zstd-3 for unified_v2 tarballs.
# zstd-3 produces ~same ratio as gzip-9 on diff payloads (xdelta3-compressed
# bytes are already near-entropy) at ~10× CPU and ~5× less peak RAM during
# compress. Read side auto-detects via magic bytes for backwards-compat with
# existing 17 historical Arweave entries + Day 1 genesis anchor (all gzip).
EVENT_TARBALL_ZSTD_LEVEL = 3
EVENT_TARBALL_EXT = ".tar.zst"
GZIP_MAGIC = b"\x1f\x8b"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _detect_compression(head: bytes) -> str:
    """Return 'zstd' | 'gzip' from the first few bytes of a tarball.

    Raises ValueError on unknown magic — caller's tarball is not a
    pack_event_tarball product (corruption / wrong-format upload).
    """
    if len(head) >= 4 and head[:4] == ZSTD_MAGIC:
        return "zstd"
    if len(head) >= 2 and head[:2] == GZIP_MAGIC:
        return "gzip"
    raise ValueError(
        f"unknown tarball magic bytes {head[:4].hex()!r} — expected zstd "
        f"({ZSTD_MAGIC.hex()}) or gzip ({GZIP_MAGIC.hex()})"
    )


@dataclass(frozen=True)
class FileDiffSpec:
    """One file's pack-time specification.

    arc_name: stable identifier for the file inside the manifest + tarball
        (NOT the on-host source path — that varies between hosts). Matches
        the second element of PERSONALITY_PATHS/TIMECHAIN_PATHS tuples.
    diff_dict: output of diff_encoders.encode_diff() for this file.
    """
    arc_name: str
    diff_dict: dict


def _file_meta_from_diff(diff_dict: dict) -> dict:
    """Extract the metadata subset needed at restore time.

    Excludes patch_bytes / patch_path / patch_owned / patch_size_bytes
    (the bytes live in their own tarball member; the local file-system
    pointers are upload-only). Keeps everything else verbatim so the
    diff_dict can be reconstructed at apply time by adding patch_bytes back.
    """
    _SKIP = {"patch_bytes", "patch_path", "patch_owned", "patch_size_bytes"}
    meta = {k: v for k, v in diff_dict.items() if k not in _SKIP}
    # Required fields per all encoders
    for k in ("diff_mode", "merkle_root", "size_bytes", "encoder"):
        if k not in meta:
            raise ValueError(
                f"diff_dict missing required field {k!r} (encoder must populate)"
            )
    return meta


def pack_event_tarball(
    event_id: str,
    event_type: str,
    component: str,
    file_specs: Iterable[FileDiffSpec],
    output_path: str,
    *,
    ts_unix: Optional[float] = None,
    extra_metadata: Optional[dict] = None,
) -> dict:
    """Pack one event's per-file diffs into a gzipped tarball at output_path.

    Args:
        event_id: UnifiedManifest event_id this tarball belongs to.
        event_type: "baseline" | "incremental".
        component: "personality" | "timechain" | "soul".
        file_specs: Per-file FileDiffSpec entries (one per in-scope file).
            Order is preserved in the metadata for deterministic Merkle.
        output_path: Where to write the tarball.
        ts_unix: Event timestamp (defaults to time.time()).
        extra_metadata: Optional dict merged into the top-level event
            metadata (e.g. {"block_range": [first, last]} for timechain).

    Returns: dict with {"path", "size_bytes", "tarball_sha256", "file_count"}.
        tarball_sha256 is the on-disk gzipped file hash — that's what the
        UnifiedManifest event's merkle_root should be set to so restore can
        verify the byte sequence we shipped to Arweave matches what came back.
    """
    if event_type not in ("baseline", "incremental"):
        raise ValueError(
            f"event_type must be 'baseline' or 'incremental', got {event_type!r}"
        )
    if component not in ("personality", "timechain", "soul"):
        raise ValueError(
            f"component must be 'personality', 'timechain', or 'soul', "
            f"got {component!r}"
        )

    parent = os.path.dirname(output_path) or "."
    os.makedirs(parent, exist_ok=True)

    file_specs = list(file_specs)
    # Validate arc_name uniqueness — duplicates would silently overwrite
    # tarball members and corrupt the index.
    seen: set[str] = set()
    for spec in file_specs:
        if not isinstance(spec.arc_name, str) or not spec.arc_name:
            raise ValueError("FileDiffSpec.arc_name must be a non-empty string")
        if spec.arc_name in seen:
            raise ValueError(f"duplicate arc_name {spec.arc_name!r}")
        seen.add(spec.arc_name)

    files_meta = []
    ts = ts_unix if ts_unix is not None else time.time()
    # Phase 5 (2026-05-19) — STREAMING refactor. Encoders return patch_path
    # (on-disk pointer) instead of patch_bytes (in-memory bytes). pack_event_tarball
    # streams from patch_path into the gzipped tar via tar.addfile() reading
    # the file-handle directly (tarfile reads info.size bytes from the fd —
    # no Python bytes object materialized for the file content). For backward-
    # compatibility with any legacy diff_dict that still carries patch_bytes
    # (e.g. tests that pre-date this refactor), we fall through to the
    # in-memory path. Owned temp paths (encoder-created vcdiff / tail files)
    # are unlinked AFTER tarball is finalized.
    owned_paths_to_cleanup: list[str] = []

    # Helper: stream sha256 from a file-handle WHILE reading exactly N bytes,
    # so we don't read the file twice (once for sha256, once for tar.addfile).
    # We achieve this by reading the file ONCE into the tar, but for the
    # per-file patch_bytes_sha256 record (needed at restore-time verify) we
    # re-hash via streaming. Per-file extra disk read is unavoidable here
    # unless we tee — RSS impact is zero (1 MiB chunks).
    def _sha256_path(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # Write the zstd-compressed tarball atomically via a .tmp file. Crash
    # mid-write leaves the prior (if any) output_path intact.
    # Phase 5 chunk 5F: zstd-3 streaming writer wraps the tar in mode "w|"
    # so neither the patch_bytes nor the tar buffer is materialized in RAM.
    tmp_path = output_path + ".tmp"
    try:
        cctx = zstandard.ZstdCompressor(level=EVENT_TARBALL_ZSTD_LEVEL)
        with open(tmp_path, "wb") as f_out, \
                cctx.stream_writer(f_out, closefd=False) as zst_writer, \
                tarfile.open(fileobj=zst_writer, mode="w|") as tar:  # noqa: async-block — backup cascade runs via loop.run_until_complete on a dedicated sequential worker loop — no concurrent coroutines to starve, not the FastAPI loop
            for spec in file_specs:
                patch_path = spec.diff_dict.get("patch_path")
                if patch_path is not None:
                    # STREAMING PATH (Phase 5 default): tar reads from fd in
                    # blocksize chunks; no bytes object holds the file content.
                    # 2026-05-23 D-SPEC-123 follow-up: if the patch_path
                    # disappeared between encode and pack (rolling-retention
                    # source that the encoder did NOT snapshot — e.g. a
                    # future encoder regression), SKIP the file with a clear
                    # WARN rather than raising. Raising aborts the whole
                    # unified_v2 event and falls back to the bug-laden legacy
                    # cascade — strictly worse than missing one file from one
                    # event (the next event picks up the new content; legacy
                    # cascade costs money + has known bugs). The primary
                    # mitigation is the encoder-side hardlink snapshot in
                    # full_ship.encode_diff; this is defense-in-depth.
                    if not os.path.exists(patch_path):
                        logger.warning(
                            "[pack_event_tarball] SKIP %s — patch_path "
                            "vanished between encode and pack: %s (rolling-"
                            "retention source not snapshotted by encoder?)",
                            spec.arc_name, patch_path)
                        continue
                    patch_size = spec.diff_dict.get(
                        "patch_size_bytes", os.path.getsize(patch_path)
                    )
                    member_name = FILES_PREFIX + spec.arc_name
                    info = tarfile.TarInfo(name=member_name)
                    info.size = patch_size
                    info.mtime = int(ts)
                    info.mode = 0o644
                    with open(patch_path, "rb") as fh:
                        tar.addfile(info, fh)
                    # Compute restore-time verify hash via streaming
                    patch_sha = _sha256_path(patch_path)
                    if spec.diff_dict.get("patch_owned"):
                        owned_paths_to_cleanup.append(patch_path)
                else:
                    # LEGACY PATH (pre-Phase-5 callers / tests): in-memory bytes.
                    patch_bytes = spec.diff_dict.get("patch_bytes", b"")
                    if not isinstance(patch_bytes, (bytes, bytearray)):
                        raise ValueError(
                            f"diff_dict[{spec.arc_name!r}] must have either "
                            f"'patch_path' (Phase 5 streaming) or 'patch_bytes' "
                            f"(legacy); got neither valid"
                        )
                    member_name = FILES_PREFIX + spec.arc_name
                    info = tarfile.TarInfo(name=member_name)
                    info.size = len(patch_bytes)
                    info.mtime = int(ts)
                    info.mode = 0o644
                    tar.addfile(info, io.BytesIO(bytes(patch_bytes)))
                    patch_sha = hashlib.sha256(bytes(patch_bytes)).hexdigest()

                file_meta = _file_meta_from_diff(spec.diff_dict)
                file_meta["arc_name"] = spec.arc_name
                file_meta["member"] = member_name
                file_meta["patch_bytes_sha256"] = patch_sha
                files_meta.append(file_meta)

            # Embedded metadata member — placed LAST so streaming readers
            # see all file members first (helps progress reporting at restore).
            metadata = {
                "schema_version": EVENT_TARBALL_SCHEMA_VERSION,
                "event_id": event_id,
                "event_type": event_type,
                "component": component,
                "ts_unix": ts,
                "files": files_meta,
            }
            if extra_metadata:
                # Merge under a separate key so it can't shadow the
                # canonical fields above.
                metadata["extra"] = dict(extra_metadata)

            metadata_bytes = json.dumps(
                metadata, indent=2, sort_keys=False
            ).encode("utf-8")
            info = tarfile.TarInfo(name=EVENT_METADATA_MEMBER)
            info.size = len(metadata_bytes)
            info.mtime = int(ts)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(metadata_bytes))

        # Compute final hash (post-zstd-compress; this is what Arweave will
        # receive and what the §24.7 event Merkle commits to).
        h = hashlib.sha256()
        size = 0
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
                size += len(chunk)
        tarball_sha256 = h.hexdigest()

        os.replace(tmp_path, output_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        # On failure, still clean up encoder-owned temp paths so we don't
        # leak vcdiff / tail tempfiles. The encoders rely on us doing this
        # because they pass ownership to the caller (patch_owned=True).
        for _p in owned_paths_to_cleanup:
            try:
                os.unlink(_p)
            except OSError:
                pass
        raise

    # Phase 5 (2026-05-19) — clean up encoder-owned temp paths now that the
    # tarball has been written + atomically renamed into place. Best-effort:
    # OS error on unlink does NOT fail the cascade.
    for _p in owned_paths_to_cleanup:
        try:
            os.unlink(_p)
        except OSError:
            pass

    return {
        "path": output_path,
        "size_bytes": size,
        "tarball_sha256": tarball_sha256,
        "file_count": len(files_meta),                # actually packed
        "files_requested": len(file_specs),           # what caller asked for
        "files_skipped_vanished": (
            len(file_specs) - len(files_meta)
        ),    # SKIP'd by the patch_path-vanished guard (D-SPEC-123 follow-up)
        "packed_arc_names": [m["arc_name"] for m in files_meta],
    }


@dataclass
class UnpackedEvent:
    """Parsed event metadata + a streaming accessor for per-file patch bytes.

    Holds the open tarfile so per-file patches can be lazily extracted
    without re-reading the whole archive. Call .close() when done (or use
    as a context manager).

    For zstd-compressed path inputs we materialize a temp-decompressed file
    alongside (random-access requires seek, which zstd stream readers don't
    provide pre-Python-3.14). The temp file is unlinked on close().
    """
    event_id: str
    event_type: str
    component: str
    ts_unix: float
    files: list[dict]
    extra: dict
    schema_version: int
    _tar: tarfile.TarFile
    _temp_decompressed_path: Optional[str] = field(default=None)

    def __enter__(self) -> "UnpackedEvent":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._tar.close()
        except Exception:
            pass
        if self._temp_decompressed_path:
            try:
                os.unlink(self._temp_decompressed_path)
            except OSError:
                pass
            self._temp_decompressed_path = None

    def get_patch_bytes(self, arc_name: str, verify_hash: bool = True) -> bytes:
        """Return the raw patch_bytes for one file in this event tarball.

        Verifies the bytes' sha256 matches `patch_bytes_sha256` recorded
        at pack time. Mismatch raises ValueError — tarball corruption
        between pack and unpack (rare on Arweave but possible with
        gateway-rewriting proxies).

        `verify_hash=False` downgrades a per-file mismatch to a logged WARNING
        and returns the bytes anyway. This is ONLY safe when the WHOLE tarball
        has already been authenticated against its on-chain arc
        (`sha256(tarball)[:32]==arc`, the v=3 memo, INV-MBR-4/12): once the arc
        matches, every member byte is provably what was committed on-chain, so
        the per-file `patch_bytes_sha256` is strictly redundant — and for
        tarballs packed before the 2026-05-31 copy-snapshot fix (commit
        ed5f4d0c) it can be STALE (a live append-only log grew between the
        addfile write and the separate verify-hash re-read). The sovereign
        restore passes verify_hash=False because it verifies the on-chain arc
        per component before apply; every other (arc-unverified) caller keeps
        the default strict behaviour.
        """
        meta = next(
            (f for f in self.files if f.get("arc_name") == arc_name), None
        )
        if meta is None:
            raise KeyError(
                f"arc_name {arc_name!r} not in event {self.event_id} files index"
            )
        member_name = meta["member"]
        try:
            member = self._tar.getmember(member_name)
        except KeyError:
            raise KeyError(
                f"tarball member {member_name!r} missing for arc_name "
                f"{arc_name!r} in event {self.event_id}"
            )
        f = self._tar.extractfile(member)
        if f is None:
            raise ValueError(
                f"tarball member {member_name!r} is not a regular file "
                f"(corrupt tarball?)"
            )
        data = f.read()
        expected_hash = meta.get("patch_bytes_sha256")
        if expected_hash:
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected_hash:
                msg = (
                    f"patch_bytes sha256 mismatch for {arc_name!r}: "
                    f"expected {expected_hash}, got {actual} (tarball corruption)"
                )
                if verify_hash:
                    raise ValueError(msg)
                # arc-verified caller (sovereign restore): the on-chain arc has
                # already authenticated every byte of this tarball, so the bytes
                # are genuine; the per-file hash is stale (pre-ed5f4d0c pack race).
                logger.warning(
                    "[event_tarball] %s — proceeding: tarball already "
                    "on-chain-arc-verified; per-file hash is advisory.", msg)
        return data

    def diff_dict_for(self, arc_name: str, verify_hash: bool = True) -> dict:
        """Reconstruct the diff_dict for `arc_name` (metadata + patch_bytes).

        Returned dict is suitable for `diff_encoders.apply_diff(baseline, dict,
        output)`. patch_bytes is loaded lazily via get_patch_bytes().

        `verify_hash` is forwarded to get_patch_bytes — see its docstring; pass
        False ONLY from a caller that has already authenticated the tarball
        against its on-chain arc.
        """
        meta = next(
            (f for f in self.files if f.get("arc_name") == arc_name), None
        )
        if meta is None:
            raise KeyError(
                f"arc_name {arc_name!r} not in event {self.event_id} files index"
            )
        d = {k: v for k, v in meta.items()
             if k not in ("arc_name", "member", "patch_bytes_sha256")}
        d["patch_bytes"] = self.get_patch_bytes(arc_name, verify_hash=verify_hash)
        return d


def unpack_event_tarball(source: str | bytes) -> UnpackedEvent:
    """Open a packed event tarball for streaming access.

    `source` is either a filesystem path (str) or the raw compressed bytes
    (bytes — used when fetched from Arweave in-memory).

    Auto-detects gzip vs zstd compression via magic bytes so the 17
    pre-chunk-5F historical Arweave entries (gzip-9) remain readable
    alongside new zstd-3 events.

    Caller MUST close (via context manager or .close()).

    Raises ValueError on:
      - unknown/corrupt magic bytes
      - missing/corrupt __event_metadata.json
      - schema_version mismatch
      - missing required metadata fields
    """
    temp_decompressed_path: Optional[str] = None
    if isinstance(source, (bytes, bytearray)):
        head = bytes(source[:4])
        comp = _detect_compression(head)
        if comp == "zstd":
            # Streaming writer doesn't embed content-size in the frame
            # header, so dctx.decompress(blob) raises "could not determine
            # content size". Use stream_reader to decode without that hint.
            dctx = zstandard.ZstdDecompressor()
            decomp_buf = io.BytesIO()
            with dctx.stream_reader(io.BytesIO(bytes(source))) as reader:
                while True:
                    chunk = reader.read(1 << 20)
                    if not chunk:
                        break
                    decomp_buf.write(chunk)
            decomp_buf.seek(0)
            tar = tarfile.open(fileobj=decomp_buf, mode="r:")
        else:  # gzip
            tar = tarfile.open(fileobj=io.BytesIO(bytes(source)), mode="r:gz")
    elif isinstance(source, str):
        with open(source, "rb") as f_head:
            head = f_head.read(4)
        comp = _detect_compression(head)
        if comp == "zstd":
            # Random-access requires seekable input; zstd stream readers in
            # zstandard ≤0.25 are forward-only. Materialize a decompressed
            # sibling, opened in tarfile "r:" mode for getmember()/extractfile().
            # The sibling is unlinked when UnpackedEvent.close() runs.
            temp_decompressed_path = source + ".decomp.tmp"
            dctx = zstandard.ZstdDecompressor()
            with open(source, "rb") as f_in, open(
                    temp_decompressed_path, "wb") as f_out:
                dctx.copy_stream(f_in, f_out)
            tar = tarfile.open(temp_decompressed_path, mode="r:")
        else:  # gzip — native tarfile seek support
            tar = tarfile.open(source, mode="r:gz")
    else:
        raise TypeError(
            f"source must be str (path) or bytes, got {type(source).__name__}"
        )

    def _cleanup_on_error() -> None:
        try:
            tar.close()
        except Exception:
            pass
        if temp_decompressed_path:
            try:
                os.unlink(temp_decompressed_path)
            except OSError:
                pass

    try:
        try:
            member = tar.getmember(EVENT_METADATA_MEMBER)
        except KeyError:
            raise ValueError(
                f"event tarball missing {EVENT_METADATA_MEMBER!r} — not a "
                f"valid backup_event_tarball"
            )
        f = tar.extractfile(member)
        if f is None:
            raise ValueError(
                f"event tarball {EVENT_METADATA_MEMBER!r} is not a regular file"
            )
        try:
            metadata = json.loads(f.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(
                f"event tarball {EVENT_METADATA_MEMBER!r} is not valid JSON: {e}"
            ) from None
    except Exception:
        _cleanup_on_error()
        raise

    schema = metadata.get("schema_version")
    if schema != EVENT_TARBALL_SCHEMA_VERSION:
        _cleanup_on_error()
        raise ValueError(
            f"event tarball schema_version {schema!r} unknown "
            f"(supported: {EVENT_TARBALL_SCHEMA_VERSION})"
        )
    for k in ("event_id", "event_type", "component", "files"):
        if k not in metadata:
            _cleanup_on_error()
            raise ValueError(
                f"event tarball metadata missing required field {k!r}"
            )
    if not isinstance(metadata["files"], list):
        _cleanup_on_error()
        raise ValueError("event tarball metadata.files must be a list")

    return UnpackedEvent(
        event_id=metadata["event_id"],
        event_type=metadata["event_type"],
        component=metadata["component"],
        ts_unix=float(metadata.get("ts_unix", 0.0)),
        files=list(metadata["files"]),
        extra=dict(metadata.get("extra", {})),
        schema_version=schema,
        _tar=tar,
        _temp_decompressed_path=temp_decompressed_path,
    )
