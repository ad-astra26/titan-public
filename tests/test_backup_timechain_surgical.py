"""SPEC §24 + rFP §3.3 — Arweave-sourced timechain surgical repair tests."""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from typing import Optional

import pytest

from titan_hcl.logic.backup_event_tarball import (
    FileDiffSpec,
    pack_event_tarball,
)
from titan_hcl.logic.backup_restore import (
    HALT_APPLY_FAILED,
    HALT_MANIFEST_EMPTY,
    HALT_TARBALL_FETCH_FAILED,
    HALT_TARBALL_HASH_MISMATCH,
)
from titan_hcl.logic.backup_timechain_surgical import (
    extract_fork_file_to_scratch,
    find_covering_event,
    fork_arc_name,
    surgical_repair_from_arweave,
)
from titan_hcl.logic.backup_unified_manifest import (
    UnifiedManifest,
    make_event,
)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ── Stub CorruptionReport + ChainIntegrity ────────────────────────────────


@dataclass
class StubCorruptionReport:
    """Minimal stand-in for timechain_integrity.CorruptionReport."""
    fork_id: int
    fork_name: str
    corruption_height: int
    total_blocks: int = 1000


@dataclass
class StubRepairResult:
    """Minimal stand-in for timechain_integrity.RepairResult."""
    tier: int
    fork_id: int
    success: bool
    blocks_before: int
    blocks_after: int
    blocks_recovered: int
    blocks_lost: int
    orphans_found: int = 0
    orphans_recommitted: int = 0
    detail: str = ""


class StubChainIntegrity:
    """Records surgical_repair calls + returns scripted result."""
    def __init__(self, *, result_success: bool = True, detail: str = "ok"):
        self.calls: list[dict] = []
        self.result_success = result_success
        self.detail = detail
        self.raise_on_call: Optional[Exception] = None

    def surgical_repair(self, fork_id, corruption, backup_data_dir):
        self.calls.append({
            "fork_id": fork_id,
            "corruption_height": corruption.corruption_height,
            "backup_data_dir": backup_data_dir,
            # Verify the expected file was extracted before the call:
            "scratch_contents": sorted(os.listdir(backup_data_dir)),
        })
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return StubRepairResult(
            tier=1, fork_id=fork_id, success=self.result_success,
            blocks_before=corruption.total_blocks,
            blocks_after=corruption.total_blocks if self.result_success else 0,
            blocks_recovered=corruption.corruption_height,
            blocks_lost=0 if self.result_success else corruption.total_blocks,
            detail=self.detail,
        )


# ── fork_arc_name ────────────────────────────────────────────────────────


def test_fork_arc_name_canonical():
    assert fork_arc_name("main") == "timechain/chain_main.bin"
    assert fork_arc_name("episodic") == "timechain/chain_episodic.bin"
    assert fork_arc_name("meta") == "timechain/chain_meta.bin"


# ── find_covering_event ──────────────────────────────────────────────────


def _bare_event(event_id: str, ev_type: str, prev_id: Optional[str],
                tc_block_ranges: Optional[dict] = None,
                tc_block_range: Optional[list] = None) -> dict:
    """Build a minimal manifest event for find_covering_event testing."""
    tc = {
        "tx_id": "tx_" + event_id, "merkle_root": "a" * 64,
        "size_bytes": 100, "diff_mode": ev_type,
    }
    if tc_block_ranges is not None:
        tc["block_ranges"] = tc_block_ranges
    if tc_block_range is not None:
        tc["block_range"] = tc_block_range
    return make_event(
        event_id=event_id, event_type=ev_type, prev_event_id=prev_id,
        baseline_trigger="first_event" if ev_type == "baseline" else None,
        personality={"tx_id": "px", "merkle_root": "b" * 64,
                     "size_bytes": 1, "diff_mode": ev_type},
        timechain=tc,
    )


def test_find_covering_event_via_block_ranges(tmp_path):
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    m.append_event(_bare_event(
        "e1", "baseline", None,
        tc_block_ranges={"main": [0, 100], "episodic": [0, 50]},
    ))
    m.append_event(_bare_event(
        "e2", "incremental", "e1",
        tc_block_ranges={"main": [101, 200], "episodic": [51, 80]},
    ))
    m.append_event(_bare_event(
        "e3", "incremental", "e2",
        tc_block_ranges={"main": [201, 300], "episodic": [81, 120]},
    ))
    # main 150 → covered by e2
    ev = find_covering_event(m, "main", 150)
    assert ev is not None and ev["event_id"] == "e2"
    # episodic 60 → covered by e2
    ev = find_covering_event(m, "episodic", 60)
    assert ev is not None and ev["event_id"] == "e2"
    # main 50 → covered by e1 (walk_chain yields newest first; first
    # match wins → e1 in this layout since it's the only one with 50)
    ev = find_covering_event(m, "main", 50)
    assert ev is not None and ev["event_id"] == "e1"


def test_find_covering_event_legacy_block_range_main_only(tmp_path):
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    m.append_event(_bare_event(
        "e1", "baseline", None,
        tc_block_range=[0, 100],
    ))
    # Legacy block_range only matches fork "main"
    assert find_covering_event(m, "main", 50) is not None
    # Other forks don't match the legacy field
    assert find_covering_event(m, "episodic", 50) is None


def test_find_covering_event_no_match(tmp_path):
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    m.append_event(_bare_event(
        "e1", "baseline", None,
        tc_block_ranges={"main": [0, 100]},
    ))
    # Height 999 way out of range
    assert find_covering_event(m, "main", 999) is None
    # Unknown fork
    assert find_covering_event(m, "episodic", 50) is None


def test_find_covering_event_empty_manifest_raises(tmp_path):
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    with pytest.raises(ValueError, match=HALT_MANIFEST_EMPTY):
        find_covering_event(m, "main", 50)


def test_find_covering_event_returns_most_recent(tmp_path):
    """When two events both cover the height (overlap), the newest one wins."""
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    m.append_event(_bare_event(
        "e1", "baseline", None,
        tc_block_ranges={"main": [0, 100]},
    ))
    # A re-baselined event that re-covers the same range (e.g. monthly
    # rebase). This is a baseline so it includes the full file again.
    m.append_event(_bare_event(
        "e2", "baseline", "e1",
        tc_block_ranges={"main": [0, 100]},
    ))
    ev = find_covering_event(m, "main", 50)
    assert ev is not None and ev["event_id"] == "e2"  # most recent wins


# ── extract_fork_file_to_scratch ─────────────────────────────────────────


def _full_dict(content: bytes) -> dict:
    return {
        "diff_mode": "full", "patch_bytes": content,
        "merkle_root": _sha256(content), "size_bytes": len(content),
        "encoder": "full_ship",
    }


def _tail_dict(tail: bytes, prev_offset: int, post: bytes, br=None) -> dict:
    return {
        "diff_mode": "tail", "patch_bytes": tail,
        "merkle_root": _sha256(post), "size_bytes": len(post),
        "prev_offset_bytes": prev_offset,
        "block_range": list(br) if br else None,
        "encoder": "timechain_tail",
    }


def test_extract_fork_file_to_scratch_full_mode(tmp_path):
    chain_bytes = b"BLOCK_DATA" * 100
    out = tmp_path / "evt.tar.gz"
    pack_event_tarball(
        event_id="e1", event_type="baseline", component="timechain",
        file_specs=[
            FileDiffSpec("timechain/chain_main.bin", _full_dict(chain_bytes)),
        ],
        output_path=str(out),
    )
    scratch = tmp_path / "scratch"
    extracted = extract_fork_file_to_scratch(
        tarball_bytes=out.read_bytes(),
        fork_name="main",
        scratch_dir=str(scratch),
    )
    assert extracted == str(scratch / "chain_main.bin")
    assert (scratch / "chain_main.bin").read_bytes() == chain_bytes


def test_extract_fork_file_to_scratch_rejects_tail_mode(tmp_path):
    """Tail-only payloads can't be directly extracted as the pre-tamper
    file — they're a diff, not a full snapshot."""
    out = tmp_path / "evt.tar.gz"
    pack_event_tarball(
        event_id="e2", event_type="incremental", component="timechain",
        file_specs=[
            FileDiffSpec(
                "timechain/chain_main.bin",
                _tail_dict(b"new_blocks", prev_offset=1000,
                           post=b"x" * 1010, br=(100, 105)),
            ),
        ],
        output_path=str(out),
    )
    with pytest.raises(ValueError, match="tail"):
        extract_fork_file_to_scratch(
            tarball_bytes=out.read_bytes(),
            fork_name="main",
            scratch_dir=str(tmp_path / "scratch"),
        )


def test_extract_fork_file_to_scratch_missing_arc(tmp_path):
    """Tarball with chain_main.bin but request for chain_episodic.bin."""
    out = tmp_path / "evt.tar.gz"
    pack_event_tarball(
        event_id="e1", event_type="baseline", component="timechain",
        file_specs=[
            FileDiffSpec("timechain/chain_main.bin",
                         _full_dict(b"main data")),
        ],
        output_path=str(out),
    )
    with pytest.raises(ValueError, match="no member"):
        extract_fork_file_to_scratch(
            tarball_bytes=out.read_bytes(),
            fork_name="episodic",
            scratch_dir=str(tmp_path / "scratch"),
        )


# ── surgical_repair_from_arweave orchestration ────────────────────────────


def _build_full_timechain_event(
    *, event_id: str, ev_type: str, prev_id: Optional[str],
    fork_files: dict[str, bytes],  # {fork_name: full_bytes}
    block_ranges: dict[str, list[int]],
    arweave_store: dict[str, bytes],
    tmp_path,
) -> dict:
    """Build a manifest event whose timechain tarball physically contains
    each fork_files entry as a full-mode payload, with block_ranges
    populated on the manifest event."""
    out = tmp_path / f"{event_id}_tc.tar.gz"
    specs = [
        FileDiffSpec(f"timechain/chain_{fname}.bin", _full_dict(data))
        for fname, data in fork_files.items()
    ]
    info = pack_event_tarball(
        event_id=event_id, event_type=ev_type, component="timechain",
        file_specs=specs, output_path=str(out),
    )
    tarball_bytes = out.read_bytes()
    tx_id = "ar_" + uuid.uuid4().hex[:16]
    arweave_store[tx_id] = tarball_bytes
    tc = {
        "tx_id": tx_id, "merkle_root": info["tarball_sha256"],
        "size_bytes": info["size_bytes"], "diff_mode": ev_type,
        "block_ranges": block_ranges,
    }
    return make_event(
        event_id=event_id, event_type=ev_type, prev_event_id=prev_id,
        baseline_trigger="first_event" if ev_type == "baseline" else None,
        personality={"tx_id": "px_" + event_id, "merkle_root": "b" * 64,
                     "size_bytes": 1, "diff_mode": ev_type},
        timechain=tc,
    )


@pytest.mark.asyncio
async def test_surgical_repair_from_arweave_happy_path(tmp_path,
                                                       monkeypatch):
    """End-to-end: walk manifest → find event → fetch → extract → delegate
    to stub ChainIntegrity → success."""
    # Patch FORK_NAMES to avoid pulling the full timechain module
    monkeypatch.setattr(
        "titan_hcl.logic.timechain.FORK_NAMES",
        {1: "main", 3: "episodic"},
        raising=False,
    )

    arweave_store: dict[str, bytes] = {}
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    main_bytes = b"MAIN_FORK_DATA" * 50
    epi_bytes = b"EPISODIC_FORK_DATA" * 30
    event = _build_full_timechain_event(
        event_id="e1", ev_type="baseline", prev_id=None,
        fork_files={"main": main_bytes, "episodic": epi_bytes},
        block_ranges={"main": [0, 100], "episodic": [0, 50]},
        arweave_store=arweave_store, tmp_path=tmp_path,
    )
    manifest.append_event(event)

    async def fetch(tx_id):
        return arweave_store[tx_id]

    integrity = StubChainIntegrity(result_success=True, detail="splice ok")
    corruption = StubCorruptionReport(
        fork_id=1, fork_name="main", corruption_height=50, total_blocks=101,
    )
    result = await surgical_repair_from_arweave(
        chain_integrity=integrity,
        fork_id=1,
        corruption=corruption,
        manifest=manifest,
        arweave_fetch=fetch,
    )
    assert result.status == "success", (result.halt_reason, result.errors)
    assert result.sourced_from_event_id == "e1"
    assert result.sourced_from_tx_id == event["timechain"]["tx_id"]
    assert result.sourced_block_range == [0, 100]
    assert result.tarball_size_bytes > 0
    # Verify the stub was called with the extracted file present
    assert len(integrity.calls) == 1
    call = integrity.calls[0]
    assert call["fork_id"] == 1
    assert call["corruption_height"] == 50
    assert "chain_main.bin" in call["scratch_contents"]
    # Detail augmented with audit
    assert "sourced_from_arweave_tx" in result.repair_result.detail
    assert event["timechain"]["tx_id"] in result.repair_result.detail


@pytest.mark.asyncio
async def test_surgical_repair_no_covering_event(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "titan_hcl.logic.timechain.FORK_NAMES",
        {1: "main"}, raising=False,
    )
    arweave_store: dict[str, bytes] = {}
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    event = _build_full_timechain_event(
        event_id="e1", ev_type="baseline", prev_id=None,
        fork_files={"main": b"data"},
        block_ranges={"main": [0, 100]},
        arweave_store=arweave_store, tmp_path=tmp_path,
    )
    manifest.append_event(event)

    async def fetch(tx_id): return arweave_store[tx_id]

    integrity = StubChainIntegrity()
    corruption = StubCorruptionReport(
        fork_id=1, fork_name="main", corruption_height=9999,
        total_blocks=10000,
    )
    result = await surgical_repair_from_arweave(
        chain_integrity=integrity, fork_id=1, corruption=corruption,
        manifest=manifest, arweave_fetch=fetch,
    )
    assert result.status == "no_covering_event"
    assert integrity.calls == []


@pytest.mark.asyncio
async def test_surgical_repair_arweave_fetch_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "titan_hcl.logic.timechain.FORK_NAMES",
        {1: "main"}, raising=False,
    )
    arweave_store: dict[str, bytes] = {}
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    event = _build_full_timechain_event(
        event_id="e1", ev_type="baseline", prev_id=None,
        fork_files={"main": b"data"},
        block_ranges={"main": [0, 100]},
        arweave_store=arweave_store, tmp_path=tmp_path,
    )
    manifest.append_event(event)

    async def fetch(tx_id):
        raise RuntimeError("simulated 503")

    integrity = StubChainIntegrity()
    corruption = StubCorruptionReport(
        fork_id=1, fork_name="main", corruption_height=50, total_blocks=101,
    )
    result = await surgical_repair_from_arweave(
        chain_integrity=integrity, fork_id=1, corruption=corruption,
        manifest=manifest, arweave_fetch=fetch,
    )
    assert result.status == "halted"
    assert result.halt_reason == HALT_TARBALL_FETCH_FAILED
    assert integrity.calls == []


@pytest.mark.asyncio
async def test_surgical_repair_tarball_tamper_halts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "titan_hcl.logic.timechain.FORK_NAMES",
        {1: "main"}, raising=False,
    )
    arweave_store: dict[str, bytes] = {}
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    event = _build_full_timechain_event(
        event_id="e1", ev_type="baseline", prev_id=None,
        fork_files={"main": b"original"},
        block_ranges={"main": [0, 100]},
        arweave_store=arweave_store, tmp_path=tmp_path,
    )
    manifest.append_event(event)
    # Tamper the stored tarball after upload
    arweave_store[event["timechain"]["tx_id"]] = b"TAMPERED"

    async def fetch(tx_id): return arweave_store[tx_id]

    integrity = StubChainIntegrity()
    corruption = StubCorruptionReport(
        fork_id=1, fork_name="main", corruption_height=50, total_blocks=101,
    )
    result = await surgical_repair_from_arweave(
        chain_integrity=integrity, fork_id=1, corruption=corruption,
        manifest=manifest, arweave_fetch=fetch,
    )
    assert result.status == "halted"
    assert result.halt_reason == HALT_TARBALL_HASH_MISMATCH
    assert integrity.calls == []


@pytest.mark.asyncio
async def test_surgical_repair_delegate_failure_halts(tmp_path, monkeypatch):
    """When chain_integrity.surgical_repair returns success=False, we
    surface HALT_APPLY_FAILED with the detail."""
    monkeypatch.setattr(
        "titan_hcl.logic.timechain.FORK_NAMES",
        {1: "main"}, raising=False,
    )
    arweave_store: dict[str, bytes] = {}
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    event = _build_full_timechain_event(
        event_id="e1", ev_type="baseline", prev_id=None,
        fork_files={"main": b"data"},
        block_ranges={"main": [0, 100]},
        arweave_store=arweave_store, tmp_path=tmp_path,
    )
    manifest.append_event(event)

    async def fetch(tx_id): return arweave_store[tx_id]

    integrity = StubChainIntegrity(
        result_success=False, detail="splice point mismatch",
    )
    corruption = StubCorruptionReport(
        fork_id=1, fork_name="main", corruption_height=50, total_blocks=101,
    )
    result = await surgical_repair_from_arweave(
        chain_integrity=integrity, fork_id=1, corruption=corruption,
        manifest=manifest, arweave_fetch=fetch,
    )
    assert result.status == "halted"
    assert result.halt_reason == HALT_APPLY_FAILED
    assert any("splice point mismatch" in e for e in result.errors)
    # The delegated call was made — it returned success=False which we
    # surface as a halt. The call IS recorded in integrity.calls.
    assert len(integrity.calls) == 1
    assert integrity.calls[0]["fork_id"] == 1


@pytest.mark.asyncio
async def test_surgical_repair_delegate_raises_halts(tmp_path, monkeypatch):
    """When chain_integrity.surgical_repair raises, we surface HALT_APPLY_FAILED."""
    monkeypatch.setattr(
        "titan_hcl.logic.timechain.FORK_NAMES",
        {1: "main"}, raising=False,
    )
    arweave_store: dict[str, bytes] = {}
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    event = _build_full_timechain_event(
        event_id="e1", ev_type="baseline", prev_id=None,
        fork_files={"main": b"data"},
        block_ranges={"main": [0, 100]},
        arweave_store=arweave_store, tmp_path=tmp_path,
    )
    manifest.append_event(event)

    async def fetch(tx_id): return arweave_store[tx_id]

    integrity = StubChainIntegrity()
    integrity.raise_on_call = RuntimeError("disk full")
    corruption = StubCorruptionReport(
        fork_id=1, fork_name="main", corruption_height=50, total_blocks=101,
    )
    result = await surgical_repair_from_arweave(
        chain_integrity=integrity, fork_id=1, corruption=corruption,
        manifest=manifest, arweave_fetch=fetch,
    )
    assert result.status == "halted"
    assert result.halt_reason == HALT_APPLY_FAILED
    assert any("disk full" in e for e in result.errors)


@pytest.mark.asyncio
async def test_surgical_repair_uses_provided_scratch_dir(tmp_path, monkeypatch):
    """When scratch_dir is provided, the extracted file lands there and
    the dir is NOT cleaned up after (caller manages lifecycle)."""
    monkeypatch.setattr(
        "titan_hcl.logic.timechain.FORK_NAMES",
        {1: "main"}, raising=False,
    )
    arweave_store: dict[str, bytes] = {}
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    event = _build_full_timechain_event(
        event_id="e1", ev_type="baseline", prev_id=None,
        fork_files={"main": b"persistent_data"},
        block_ranges={"main": [0, 100]},
        arweave_store=arweave_store, tmp_path=tmp_path,
    )
    manifest.append_event(event)

    async def fetch(tx_id): return arweave_store[tx_id]

    scratch = tmp_path / "my_scratch"
    integrity = StubChainIntegrity()
    corruption = StubCorruptionReport(
        fork_id=1, fork_name="main", corruption_height=50, total_blocks=101,
    )
    result = await surgical_repair_from_arweave(
        chain_integrity=integrity, fork_id=1, corruption=corruption,
        manifest=manifest, arweave_fetch=fetch,
        scratch_dir=str(scratch),
    )
    assert result.status == "success"
    # Caller-provided scratch_dir is NOT auto-cleaned
    assert scratch.exists()
    assert (scratch / "chain_main.bin").read_bytes() == b"persistent_data"
