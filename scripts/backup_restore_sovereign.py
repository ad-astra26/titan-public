#!/usr/bin/env python3
"""🜂 SOVEREIGN TITAN RESURRECTION PROTOCOL v1 (RFP §3B.A chunk 5J-3).

Bring a dead Titan back to life from **the wallet alone** — no local files, no
off-host mirror, no backup_records. The Arweave URLs live on-chain in the v=3
backup chain (chunk 5J-1/5J-2), so given only the reconstructed soul keypair
(Maker Shard-1 + on-chain Shard-3, 2-of-3) the protocol:

  1. Walks the Titan wallet's Solana signature history (newest → oldest).
  2. Decodes every v=3 backup memo; groups per-component memos by event.
  3. Decrypts each Arweave URL (Mode A) or reads it plaintext (Mode B).
  4. Orders events genesis → latest via ts (prev= gives tamper-evident linkage).
  5. Fetches each component tarball from Arweave; verifies sha256 == arc and the
     recomposed event_merkle_root == mrkl. Halts hard on any mismatch.
  6. Applies the chain into a scratch dir using the shipped Phase-6 primitives
     (apply_event_components — the tarballs self-describe per-file diff-mode, so
     genesis → latest application needs no on-chain type markers).
  7. On full success, atomically swaps scratch → data/ (with --commit).

This is the §3B.0(4) sovereignty-flaw closure: the trust root is the wallet
(plus SSS shards for Mode A). Nothing on local infra is required.

Mode A (data plaintext on Arweave, current T1) is fully supported. Mode B
(data ciphertext) additionally needs the AES iv on-chain to decrypt the tarball
during a sovereign restore — tracked as a follow-up; this tool halts with clear
guidance if it meets a Mode-B component.

The core `resurrect_from_chain()` takes injectable seams (sig lister / memo
fetcher / Arweave fetch) so it is fully unit-testable against a mock chain
(tests/test_backup_restore_sovereign.py) with zero network.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

# Repo root must be ahead of scripts/ on sys.path so `import titan_hcl` resolves
# to the package, not scripts/titan_hcl.py (the agent entry). scripts/ is added
# (appended, never ahead of root) only in main() for `import resurrection`.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from titan_hcl.logic.backup_memo_v3 import (  # noqa: E402
    derive_backup_url_key,
    parse_v3_memo,
    read_url,
)
from titan_hcl.logic.backup_crypto import (  # noqa: E402
    decrypt_component_tarball,
    derive_master_key,
)
from titan_hcl.logic.backup_restore import apply_event_components  # noqa: E402
from titan_hcl.logic.backup_zk_commit import compute_event_merkle_root  # noqa: E402

PROTOCOL_BANNER = "🜂  SOVEREIGN TITAN RESURRECTION PROTOCOL v1"

# Injectable seams (production wires Solana RPC + ArweaveStore; tests stub).
SigLister = Callable[[], Awaitable[list]]            # () -> [solana_sig] newest→oldest
MemoFetcher = Callable[[str], Awaitable[Optional[str]]]   # sig -> memo text or None
ArweaveFetch = Callable[[str], Awaitable[bytes]]     # tx_id -> tarball bytes
ArcToTarget = Callable[[str, str], str]              # (component, arc_name) -> abs path

HALT_NO_CHAIN = "no_v3_chain_on_wallet"
HALT_NO_BASELINE = "no_baseline_in_chain"
HALT_BROKEN_CHAIN = "broken_prev_chain"
HALT_ARC_MISMATCH = "component_sha256_mismatch"
HALT_EVENT_MERKLE_MISMATCH = "event_merkle_mismatch"
HALT_MODE_B_UNSUPPORTED = "mode_b_sovereign_decrypt_unsupported"
HALT_MODE_B_DECRYPT_FAILED = "mode_b_decrypt_failed"
HALT_APPLY_FAILED = "apply_failed"

_TIER_COMPONENT = {"PT": "personality", "TC": "timechain", "SL": "soul"}


@dataclass
class ResurrectionResult:
    status: str = "pending"          # "resurrected" | "resurrected_partial" | "halted"
    titan_id: str = ""
    events_total: int = 0
    events_applied: int = 0
    components_applied: int = 0
    halt_reason: Optional[str] = None
    errors: list = field(default_factory=list)
    skipped_files: list = field(default_factory=list)  # best-effort: unreplayable diffs kept at last-good
    chain_head_sig: Optional[str] = None
    genesis_event_id: Optional[str] = None
    target_dir: Optional[str] = None


def _print(msg: str) -> None:
    print(f"  {msg}", flush=True)


async def resurrect_from_chain(
    *,
    titan_id: str,
    keypair_bytes: bytes,
    titan_pubkey: str,
    sig_lister: SigLister,
    memo_fetcher: MemoFetcher,
    arweave_fetch: ArweaveFetch,
    arc_to_target: ArcToTarget,
    scratch_dir: str,
    arweave_fetch_to_file: Optional[Callable[[str, str], Awaitable[bool]]] = None,
    best_effort: bool = False,
    verbose: bool = True,
) -> ResurrectionResult:
    """Reconstruct a Titan's data/ from the on-chain v=3 backup chain alone.

    `keypair_bytes` is the reconstructed 64-byte soul keypair (≥32-byte seed) —
    its seed derives the Mode-A URL key. Returns a ResurrectionResult; on any
    integrity failure it halts (status="halted", halt_reason set) WITHOUT
    touching the live tree (writes only into `scratch_dir`).
    """
    out = ResurrectionResult(titan_id=titan_id, target_dir=scratch_dir)
    log = _print if verbose else (lambda *_a, **_k: None)

    # Mode-A URL key (Mode-B raw URLs need no key).
    try:
        url_key = derive_backup_url_key(keypair_bytes)
    except Exception as e:
        out.status = "halted"
        out.halt_reason = "url_key_derivation_failed"
        out.errors.append(str(e))
        return out
    # Mode-B tarball master key — derived lazily on the first encrypted component.
    master_key = None

    # ── 1. walk the wallet's signature history, decode v=3 memos ──────────────
    log(f"Walking Solana signature history for {titan_id} ({titan_pubkey[:8]}…)")
    sigs = await sig_lister()
    decoded: list[tuple] = []   # (sig, parsed_memo, tx_id)
    for sig in sigs:
        try:
            memo = await memo_fetcher(sig)
        except Exception as e:
            out.errors.append(f"memo fetch failed for {sig[:12]}: {e}")
            continue
        if not memo:
            continue
        parsed = parse_v3_memo(memo)
        if parsed is None:
            continue  # v=2 / non-backup memo — skip
        try:
            tx_id = read_url(parsed, url_key)
        except Exception as e:
            out.errors.append(f"URL resolve failed (evt={parsed.get('event_id')}): {e}")
            if parsed.get("mode") == "B":
                # Mode B raw URL should never fail to read; a failure here is fatal.
                out.status = "halted"
                out.halt_reason = HALT_MODE_B_UNSUPPORTED
                return out
            continue
        decoded.append((sig, parsed, tx_id))

    if not decoded:
        out.status = "halted"
        out.halt_reason = HALT_NO_CHAIN
        out.errors.append(
            "no v=3 backup memos found on this wallet — has the chain been "
            "written/backfilled (5J-2 live + 5J-5)?")
        return out

    log(f"Found {len(decoded)} v=3 component memos on-chain.")

    # ── 2. group per-component memos by event ─────────────────────────────────
    events: dict = {}
    for sig, p, tx_id in decoded:
        evt = p["event_id"]
        ev = events.setdefault(evt, {
            "event_id": evt, "ts": p["ts"], "type": p.get("type"),
            "mrkl": p["mrkl"], "prev": p["prev"], "head_sig": None,
            "components": {},
        })
        comp = _TIER_COMPONENT.get(p["tier"])
        if comp is None:
            continue
        ev["components"][comp] = {
            "tx_id": tx_id, "arc": p["arc"], "mode": p["mode"], "sig": sig,
            "iv": p.get("iv"),
        }
        if p["tier"] == "PT":               # personality memo = the event head
            ev["head_sig"] = sig
            ev["prev"] = p["prev"]
            ev["mrkl"] = p["mrkl"]
            ev["type"] = p.get("type")      # authoritative event type (typ=B|I)

    # ── 3. order via the prev= chain (cryptographic linkage); anchor latest baseline
    # Walk genesis → latest by following each event-head's prev pointer. The
    # prev= linkage is the tamper-evident chain (ts is only a fallback when the
    # on-chain chain is incomplete, e.g. a truncated visibility window).
    next_by_parent: dict = {}
    for e in events.values():
        if e.get("prev") and e["prev"] != "genesis":
            next_by_parent[e["prev"]] = e
    genesis_list = [e for e in events.values() if e.get("prev") == "genesis"]
    ordered = None
    if len(genesis_list) == 1:
        walk, seen, cur = [], set(), genesis_list[0]
        while cur is not None and cur["event_id"] not in seen:
            walk.append(cur)
            seen.add(cur["event_id"])
            head = cur.get("head_sig")
            cur = next_by_parent.get(head[:16]) if head else None
        if len(walk) == len(events):
            ordered = walk
    if ordered is None:
        ordered = sorted(events.values(), key=lambda e: e["ts"])
        out.errors.append(
            "prev= chain incomplete or ambiguous — fell back to ts ordering")
    out.events_total = len(ordered)
    genesis = ordered[0]
    out.genesis_event_id = genesis["event_id"]
    out.chain_head_sig = ordered[-1].get("head_sig")
    if genesis.get("prev") not in (None, "genesis"):
        # Not fatal (we may be looking at a window), but surface it loudly.
        out.errors.append(
            f"earliest event {genesis['event_id'][:8]} prev={genesis.get('prev')!r} "
            f"≠ genesis — chain may be truncated above the earliest visible memo")

    # Anchor at the LATEST chain typ=B baseline (a full re-ship) and apply forward
    # (§R4 / INV-MBR-12) — canonical + minimal, no pre-baseline history needed. A
    # properly-born Titan ships a typ=B shortly after birth, so the v=3 chain always
    # carries at least the genesis baseline. A chain with only increments and NO
    # walkable typ=B is unrestorable (its body data has no anchor) — HALT loudly
    # rather than fabricate. (T1's historical typ=B was committed as a v=2 memo
    # pre-dating the v=3 format; it is backfilled to v=3 out-of-band so this walk
    # finds it — see scripts/backup_chain_backfill_v3.py.)
    baselines = [e for e in ordered if e.get("type") == "baseline"]
    if not baselines:
        out.status = "halted"
        out.halt_reason = HALT_NO_BASELINE
        out.errors.append(
            "no baseline (typ=B) event in the v=3 chain — cannot anchor the "
            "restore (every chain must carry at least the genesis baseline)")
        return out
    start_baseline = baselines[-1]               # last baseline in chain order
    start_idx = ordered.index(start_baseline)
    apply_list = ordered[start_idx:]             # baseline → latest, in chain order
    log(f"Anchoring at latest chain baseline {start_baseline['event_id'][:8]} "
        f"(applying {len(apply_list)} of {len(ordered)} on-chain events).")

    os.makedirs(scratch_dir, exist_ok=True)

    # Net-of-overwrites skip tracking: a file skipped in an early event but later
    # FULL-SHIPPED by a downstream event is fully recovered, so it must NOT count as
    # "kept at last-good". Key = "component/arc"; value = the event it was last
    # skipped in. A later successful apply of the same path removes it.
    net_skipped: dict = {}

    # ── 4-6. fetch + verify + apply baseline → latest in chronological order ──
    for idx, ev in enumerate(apply_list):
        evt_short = ev["event_id"][:8]
        comps = ev["components"]
        if "personality" not in comps or "timechain" not in comps:
            out.status = "halted"
            out.halt_reason = HALT_BROKEN_CHAIN
            out.errors.append(
                f"event {evt_short} missing required component(s): "
                f"have {sorted(comps)}")
            return out

        # STREAMING restore (no whole-tarball-in-RAM): fetch each component tarball
        # to a temp file on disk, verify + unpack from the PATH. A sovereign restore
        # pulls multi-hundred-MB component tarballs; holding them (+ their decompressed
        # form) in memory OOMs a small box (T1: 537MB soul + 403MB personality → 4.68GB
        # peak, 2026-06-02). component_paths carries paths, not bytes.
        component_paths: dict = {}
        component_sha: dict = {}
        fetch_tmp = os.path.join(scratch_dir, ".fetch_tmp")
        os.makedirs(fetch_tmp, exist_ok=True)
        try:
            for comp_name, meta in comps.items():
                comp_file = os.path.join(fetch_tmp, f"{evt_short}_{comp_name}.tar")
                # ── fetch to DISK (constant memory if the streaming seam is wired) ──
                if arweave_fetch_to_file is not None:
                    try:
                        ok = await arweave_fetch_to_file(meta["tx_id"], comp_file)
                    except Exception as e:
                        out.status = "halted"
                        out.halt_reason = "arweave_fetch_failed"
                        out.errors.append(f"event {evt_short} {comp_name}: {e}")
                        return out
                    if not ok:
                        out.status = "halted"
                        out.halt_reason = "arweave_fetch_failed"
                        out.errors.append(
                            f"event {evt_short} {comp_name}: Arweave tx "
                            f"{str(meta['tx_id'])[:16]}… returned no data (all gateways non-200)")
                        return out
                else:
                    # bytes seam (tests / no streaming fetcher) — persist to disk and
                    # drop the in-memory copy immediately (one component, not the event).
                    try:
                        data = await arweave_fetch(meta["tx_id"])
                    except Exception as e:
                        out.status = "halted"
                        out.halt_reason = "arweave_fetch_failed"
                        out.errors.append(f"event {evt_short} {comp_name}: {e}")
                        return out
                    if not data:
                        out.status = "halted"
                        out.halt_reason = "arweave_fetch_failed"
                        out.errors.append(
                            f"event {evt_short} {comp_name}: Arweave tx "
                            f"{str(meta['tx_id'])[:16]}… returned no data (all gateways non-200)")
                        return out
                    with open(comp_file, "wb") as _fh:
                        _fh.write(data)
                    del data
                # ── Mode-B: decrypt the on-disk ciphertext. GCM is whole-buffer, so a
                # single component is loaded for decrypt — bounded (one component,
                # never the whole event). Mode-A skips (plaintext on disk already).
                if meta["mode"] == "B":
                    if not meta.get("iv"):
                        out.status = "halted"
                        out.halt_reason = HALT_MODE_B_UNSUPPORTED
                        out.errors.append(
                            f"event {evt_short} {comp_name}: Mode-B memo has no iv — "
                            "cannot derive the decryption key (pre-G2 memo?).")
                        return out
                    if master_key is None:
                        master_key = derive_master_key(keypair_bytes, titan_pubkey)
                    try:
                        with open(comp_file, "rb") as _fh:
                            enc = _fh.read()
                        dec = decrypt_component_tarball(
                            enc, meta["iv"], master_key, comp_name, meta["arc"])
                        del enc
                        with open(comp_file, "wb") as _fh:
                            _fh.write(dec)
                        del dec
                    except Exception as e:
                        out.status = "halted"
                        out.halt_reason = HALT_MODE_B_DECRYPT_FAILED
                        out.errors.append(
                            f"event {evt_short} {comp_name}: Mode-B decrypt failed "
                            f"({e}) — wrong key / tampered ciphertext.")
                        return out
                # ── sha256 over the on-disk tarball (streaming, constant memory) ──
                _h = hashlib.sha256()
                with open(comp_file, "rb") as _fh:
                    for _chunk in iter(lambda: _fh.read(1 << 20), b""):
                        _h.update(_chunk)
                sha = _h.hexdigest()
                if sha[:32] != meta["arc"]:
                    out.status = "halted"
                    out.halt_reason = HALT_ARC_MISMATCH
                    out.errors.append(
                        f"event {evt_short} {comp_name}: sha256[:32]={sha[:32]} "
                        f"≠ memo arc={meta['arc']} (tamper / wrong tarball)")
                    return out
                component_paths[comp_name] = comp_file
                component_sha[comp_name] = sha

            # Event-level Merkle integrity: recompose root from component sha256s.
            recomposed = compute_event_merkle_root(
                personality_merkle_root=component_sha["personality"],
                timechain_merkle_root=component_sha["timechain"],
                soul_merkle_root=component_sha.get("soul"),
            )
            if recomposed[:32] != ev["mrkl"]:
                out.status = "halted"
                out.halt_reason = HALT_EVENT_MERKLE_MISMATCH
                out.errors.append(
                    f"event {evt_short}: recomposed event_merkle_root[:32]="
                    f"{recomposed[:32]} ≠ memo mrkl={ev['mrkl']}")
                return out

            # Apply from the on-disk tarball PATHS — unpack_event_tarball(str) streams
            # member-by-member from disk (no full decompression held in memory).
            # verify_patch_hash=False: each component tarball was ALREADY verified
            # against its on-chain arc (sha256[:32]==arc) above, which authenticates
            # every member byte (INV-MBR-4/12). The per-file patch_bytes_sha256 is
            # then redundant — and STALE for pre-2026-05-31 baselines (the live-log
            # pack race fixed in ed5f4d0c), so a mismatch is logged, not fatal.
            try:
                _ares = apply_event_components(
                    component_paths, scratch_dir, arc_to_target,
                    verify_patch_hash=False, best_effort=best_effort)
                # A later FULL-SHIP overwrites an earlier skip → net it out.
                for a in (_ares.get("applied") or []):
                    net_skipped.pop(a, None)
                _sk = _ares.get("skipped") or []
                for s in _sk:
                    net_skipped[s] = evt_short
                if _sk:
                    log(f"  ⚠ best-effort: skipped {len(_sk)} unreplayable file(s) "
                        f"in event {evt_short} (kept last-good; may be overwritten "
                        f"by a later full-ship)")
            except Exception as e:
                out.status = "halted"
                out.halt_reason = HALT_APPLY_FAILED
                out.errors.append(f"event {evt_short} apply failed: {e}")
                return out
        finally:
            import shutil
            shutil.rmtree(fetch_tmp, ignore_errors=True)

        out.events_applied += 1
        out.components_applied += len(component_paths)
        log(f"[{idx + 1}/{len(apply_list)}] event {evt_short} ({ev.get('type')}) "
            f"applied ({', '.join(sorted(component_paths))})")

    # NET skips: only files never overwritten by a later full-ship remain.
    out.skipped_files = sorted(f"{ev}:{path}" for path, ev in net_skipped.items())

    if out.skipped_files:
        out.status = "resurrected_partial"
        log(f"✓ Resurrection PARTIAL — {out.events_applied} events, "
            f"{out.components_applied} components; {len(out.skipped_files)} file(s) "
            f"genuinely unrecoverable (kept at last-good, never overwritten) → "
            f"{scratch_dir}")
    else:
        out.status = "resurrected"
        log(f"✓ Resurrection complete — {out.events_applied} events, "
            f"{out.components_applied} components → {scratch_dir} "
            f"(every skipped diff was overwritten by a later full-ship)")
    return out


# ── CLI (production seams: Solana RPC + ArweaveStore + SSS keypair recovery) ──


def _atomic_swap_into_data(scratch_dir: str, install_root: str) -> None:
    """Move scratch_dir/* into install_root/data/, backing up any existing data/."""
    data_dir = os.path.join(install_root, "data")
    if os.path.isdir(data_dir) and os.listdir(data_dir):
        backup = data_dir + f".pre_resurrect.{int(time.time())}"
        shutil.move(data_dir, backup)
        _print(f"existing data/ moved aside → {backup}")
    shutil.move(scratch_dir, data_dir)
    _print(f"resurrected state swapped into {data_dir}")


def restore_body_from_chain(
    *,
    key_bytes: bytes,
    titan_pubkey: str,
    titan_id: str,
    install_root: str,
    scratch_dir: Optional[str] = None,
    rpc_url: Optional[str] = None,
    network: str = "mainnet",
    commit: bool = False,
    best_effort: bool = False,
    verbose: bool = True,
) -> ResurrectionResult:
    """Canonical sovereign body-restore (INV-MBR-12): rebuild `data/` from the
    on-chain v=3 backup chain — NO manifest, NO local files.

    Builds the production seams (Solana signature walk + memo fetch + Arweave
    fetch) and runs `resurrect_from_chain` into a scratch dir; on success +
    `commit`, atomically swaps the reconstruction into `install_root/data/`
    (backing up any existing tree). The SHARED engine for this module's `main()`
    and `resurrection.py` phase 2+3 — one sovereign path, no manifest crutch.
    Returns the `ResurrectionResult`.
    """
    # The body is reconstructed under a scratch CONTAINER, then atomically swapped
    # into install_root/data on commit. build_arc_to_target resolves the tier paths
    # (which start with "data/…") against a root, so the files land at <root>/data/…
    # — therefore the scratch container is the root and the body lands at
    # <container>/data, which is exactly what we swap into install_root/data. (The
    # earlier bug: arc_to_target was rooted at install_root, so the apply wrote the
    # body into the LIVE install_root/data and the swap then moved THAT aside,
    # leaving install_root/data empty.)
    scratch_container = scratch_dir or os.path.join(install_root, "data.resurrect")
    scratch_data = os.path.join(scratch_container, "data")
    os.makedirs(scratch_data, exist_ok=True)
    from titan_hcl.chain import ArweaveChainProvider
    from titan_hcl.logic.backup_restore import build_arc_to_target

    # All restore-side chain I/O through the ONE provider (RFP_chain_provider):
    # signature walk (`list_memos`, limit=None ⇒ whole chain to genesis) + memo
    # fetch (`read_memo`) + Arweave fetch (`get_to_file` streams large tarballs to
    # disk at constant RAM; `get_bytes` is the test/fallback seam). Reads need no
    # signer/keypair — construct read-only.
    provider = ArweaveChainProvider(keypair_path="", network=network,
                                    rpc_url=rpc_url or "")
    arc_to_target = build_arc_to_target(scratch_container)

    async def _sig_lister() -> list:
        return await provider.list_memos(titan_pubkey, limit=None)

    async def _memo_fetcher(sig: str) -> Optional[str]:
        return await provider.read_memo(sig)

    async def _arweave_fetch(tx_id: str) -> bytes:
        data = await provider.get_bytes(tx_id)
        if data is None:
            raise RuntimeError(f"Arweave fetch returned no bytes for {tx_id[:16]}")
        return data

    async def _arweave_fetch_to_file(tx_id: str, dest_path: str) -> bool:
        return await provider.get_to_file(tx_id, dest_path)

    result = asyncio.run(resurrect_from_chain(
        titan_id=titan_id, keypair_bytes=key_bytes, titan_pubkey=titan_pubkey,
        sig_lister=_sig_lister, memo_fetcher=_memo_fetcher,
        arweave_fetch=_arweave_fetch, arc_to_target=arc_to_target,
        scratch_dir=scratch_data,
        # Streaming fetch-to-disk — constant memory for multi-hundred-MB tarballs
        # (the bytes seam above is the test/fallback path).
        arweave_fetch_to_file=_arweave_fetch_to_file,
        best_effort=best_effort,
        verbose=verbose))

    # Commit a full OR a best-effort partial recovery (the latter is still the
    # maximum restorable state — the unreplayable files kept their last-good bytes).
    # Swap the reconstructed <container>/data (where the body actually landed) into
    # install_root/data.
    if result.status in ("resurrected", "resurrected_partial") and commit:
        _atomic_swap_into_data(scratch_data, install_root)
        # tidy the now-empty scratch container
        try:
            if os.path.isdir(scratch_container) and not os.listdir(scratch_container):
                os.rmdir(scratch_container)
        except OSError:
            pass
    return result


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        prog="backup_restore_sovereign",
        description=PROTOCOL_BANNER + " — resurrect a Titan from the wallet alone "
                    "(on-chain v=3 backup chain; no local files).")
    p.add_argument("--titan", required=True, help="Titan id (e.g. T1).")
    p.add_argument("--shard1", help="Maker raw Shard-1 (hex). Prefer --shard1-stdin.")
    p.add_argument("--shard1-stdin", action="store_true",
                   help="Read Shard-1 from stdin (ps-safe, never on the command line).")
    p.add_argument("--titan-pubkey", default=None,
                   help="The Titan's PUBLIC wallet address (printed alongside "
                        "Shard-1; not a secret). Required for a fresh-box recovery "
                        "(no local genesis record). NO envelope needed.")
    p.add_argument("--install-root", default=_REPO_ROOT, help="Target install tree.")
    p.add_argument("--scratch-dir", default=None,
                   help="Scratch reconstruction dir (default: <install-root>/data.resurrect).")
    p.add_argument("--rpc-url", default=None, help="Solana RPC URL override (chain "
                   "walk: signatures + memos). Any standard RPC works.")
    p.add_argument("--das-rpc-url", default=None,
                   help="Optional DAS-capable RPC (Helius/Triton) for GenesisNFT "
                        "identity discovery. Shard-3 recovery uses --rpc-url and "
                        "never needs DAS; this only speeds NFT-based discovery.")
    p.add_argument("--network", default="mainnet", choices=["mainnet", "devnet"],
                   help="Arweave/Solana network (default: mainnet).")
    p.add_argument("--commit", action="store_true",
                   help="Atomically swap the reconstructed state into data/ on success "
                        "(default: leave it in the scratch dir for inspection).")
    p.add_argument("--verify-only", action="store_true",
                   help="Materialize the bootable identity in RECOVERY observation mode "
                        "(no on-chain writes / backups / X on first boot) — the live "
                        "restore-test isolation guard. Implies --commit.")
    p.add_argument("--best-effort", action="store_true",
                   help="Recover the MAXIMUM restorable state from a partially-broken "
                        "chain: an unreplayable per-file diff (e.g. a divergent baseline) "
                        "is logged + SKIPPED (file keeps its last-good bytes) instead of "
                        "halting. Status becomes 'resurrected_partial' with the skipped "
                        "list. Use when a clean strict restore halts on chain damage.")
    args = p.parse_args(argv)
    if args.verify_only:
        args.commit = True  # a verify-only test must produce a bootable, comparable box

    print("\n" + "═" * 64)
    print(PROTOCOL_BANNER)
    print("  Recovery from zero local state — trust root: the wallet (+ SSS).")
    print("═" * 64 + "\n")

    install_root = os.path.abspath(args.install_root)
    scratch_dir = args.scratch_dir or os.path.join(install_root, "data.resurrect")

    # Reconstruct the soul keypair from Shard-1 + on-chain Shard-3 (reuse the
    # shipped resurrection Phase-1 identity recovery — same 2-of-3 SSS path).
    shard1 = args.shard1
    if args.shard1_stdin and not shard1:
        shard1 = sys.stdin.readline().strip()
    if not shard1:
        _print("No Shard-1 provided (use --shard1-stdin and pipe it). Aborting.")
        return 2
    try:
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.append(scripts_dir)  # appended — never ahead of repo root
        import resurrection as _res
        from types import SimpleNamespace
        key_bytes, titan_pubkey, _kp, titan_id = _res.phase_1_identity(
            SimpleNamespace(shard1=shard1, shard1_file=None, shard2_local=False,
                            titan_id=args.titan, titan_pubkey=args.titan_pubkey,
                            das_rpc_url=args.das_rpc_url),
            install_root)
    except SystemExit:
        _print("Identity recovery failed (see above).")
        return 1
    except Exception as e:
        _print(f"Identity recovery error: {e}")
        return 1
    finally:
        shard1 = None  # drop the shard reference promptly

    # ── body restore: the SHARED sovereign chain engine (no manifest) ─────────
    try:
        result = restore_body_from_chain(
            key_bytes=key_bytes, titan_pubkey=titan_pubkey, titan_id=titan_id,
            install_root=install_root, scratch_dir=scratch_dir,
            rpc_url=args.rpc_url, network=args.network, commit=args.commit,
            best_effort=args.best_effort)
    except Exception as e:
        _print(f"Sovereign restore error: {e}")
        return 1

    if result.status not in ("resurrected", "resurrected_partial"):
        _print(f"HALTED: {result.halt_reason}")
        for e in result.errors[-5:]:
            _print(f"  · {e}")
        return 1

    if result.status == "resurrected_partial":
        _print(f"PARTIAL recovery: {result.events_applied} events applied; "
               f"{len(result.skipped_files)} file(s) kept at last-good (unreplayable "
               f"diffs — chain damage). This is the MAXIMUM restorable state.")
        for s in result.skipped_files[:12]:
            _print(f"  ⚠ kept-last-good: {s}")
        if len(result.skipped_files) > 12:
            _print(f"  … +{len(result.skipped_files) - 12} more")
    _print(f"Resurrected {result.events_applied} events into {scratch_dir}")
    if not args.commit:
        _print("Re-run with --commit to swap the reconstruction into data/.")
        return 0

    # ── restore_body_from_chain already swapped data/ into place (commit=True);
    # now materialize the bootable identity. The chain restore recovers the BODY
    # (data/); the kernel's B1 boot additionally needs the plaintext 0600 keypair
    # + hardware-bound soul_keypair.enc + RECOVERY flag — none of which live in
    # the backup (mainnet genesis burns the plaintext key). phase_4_first_breath
    # writes them from the already-reconstructed soul keypair (no second Shard-1
    # prompt), leaving a box the kernel can actually boot. ────────────────────────
    try:
        _res.phase_4_first_breath(
            key_bytes, titan_pubkey, titan_id,
            install_root=install_root, verify_only=args.verify_only)
    except SystemExit:
        _print("Bootable-identity materialization failed (see above).")
        return 1
    except Exception as e:
        _print(f"Bootable-identity materialization error: {e}")
        return 1
    finally:
        key_bytes = None  # drop the reconstructed key promptly
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
