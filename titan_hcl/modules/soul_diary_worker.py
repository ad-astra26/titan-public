"""titan_hcl/modules/soul_diary_worker.py — the Soul-Diary L2 worker.

`RFP_titan_authored_soul_diary` §1.0 (P1). A lean, self-contained L2 worker that
owns the daily soul-diary mechanic. On the day's last meditation it authors a
grounded, OVG-verified first-person reflection in Titan's own voice and persists
it (the narrative SELF). Pipeline (§1.0 ①②④⑤):

    MEDITATION_COMPLETE → ① latch (once/UTC-day) → ② gather grounded bundle (G18)
    → narrate (OWN LLM call: provider.complete + OutputVerifier.verify_safety —
    grounded prompt + OVG, NOT a bare distill; INV-SD-14) → ④ persist
    (core/soul.append_chronicle_entry, single-writer INV-SD-6) → ⑤ hash-chain
    ledger (core/soul_diary_chain).

Self-contained by design (Maker 2026-06-09 — simpler + debuggable than a
cross-worker compose round-trip): the worker makes its own LLM call rather than
delegating to social_worker's gateway.

P2 (§1.0 ⑥⑦) extends the pipeline past persist+hash:
    ⑥ ENRICH  → MEMORY_MEMPOOL_ADD (dst=memory, one-way) → promoted at the
                dream boundary into DuckDB+FAISS+Kuzu, classified `domain="self"`
                → Titan remembers + recalls his narrative path (INV-SD-15).
    ⑦ ANCHOR  → OuterMemoryWriter.emit(fork="main", cumulative_hash) →
                TIMECHAIN_COMMIT → main/genesis chain (FORK_MAIN=0, the SELF
                journey, NOT ACT-R forks) — a hash pointer (INV-SD-17).
P6 (§1.0 ⑨) opens Pillar B (the gated public expression): the SAME grounded
compose carries a "---SHARE---" public variant; the entry is split, and a
fail-closed `sanitize_for_public` backstop produces the privacy-clean public
projection (X-postable distillation + the archive's sanitized full entry +
a redaction count) stored alongside the hashes in the ledger row — the private
chronicle/ledger/self-memory keep the raw detail (INV-SD-3). P7-P10 (art /
DailyNFT / archive / X post) are downstream phases this same worker fires.
Every step soft-fails independently to a minimal grounded entry / skipped
enrich-anchor — never blocks the meditation cascade (INV-SD-13).
"""
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from queue import Empty

import titan_hcl.bus as bus
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL_S = 30.0
_COMPOSE_TIMEOUT_S = 90.0
_COMPOSE_MAX_TOKENS = 600
_COMPOSE_TEMPERATURE = 0.7

_WORKER_READY = False
_BOOT_DEADLINE = 0.0


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _completed_day() -> str:
    """The just-completed UTC day — the day this entry reflects on.

    RFP §6.2 / INV-SD-5: the latch fires on the *first meditation after UTC
    rollover*, authoring the **preceding** day — so the day-window holds a FULL
    day of real activity (engrams, sovereignty, felt), not a day still in
    progress. (Epoch/great-pulse-day is a DEFERRED phase: trinity §7 GREAT-PULSE
    time is [PARTIAL]/stalled and brain §247 forbids hardcoding the per-Titan
    epoch↔human-time rate — Maker 2026-06-10.)
    """
    return (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")


def _day_window_epochs(day: str) -> tuple[float, float]:
    """[start, end) wall-clock seconds for the UTC date string ``day``."""
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def _resolve_provider_name(inference_cfg: dict) -> str:
    """The configured inference provider for the worker's own LLM call.

    Canonical config key = ``inference_provider`` (``config.toml [inference]``);
    ``provider`` accepted as a legacy alias; ``"ollama_cloud"`` last-resort (the
    fleet default — Ollama Cloud ``deepseek-v3.1:671b``, per topic_infra_stack;
    venice is NOT a current provider).
    BUGFIX 2026-06-10: previously read only ``provider`` (the WRONG key) and fell
    back to ``"venice"`` — which is 402-dead on the fleet — so every diary entry
    soft-fell to the minimal grounded template instead of an authored reflection."""
    return (inference_cfg.get("inference_provider")
            or inference_cfg.get("provider") or "ollama_cloud")


async def _compose_diary(provider, verifier, prompts: dict) -> tuple[str, bool]:
    """The worker's OWN grounded-compose + OVG (§1.0 ③, self-contained).

    Grounded: the prompt already carries the real day's facts. OVG: the
    deterministic ``verify_safety`` truth-gate, run on the **strict
    ``channel="soul_diary"``** so a numeric claim diverging from the grounded
    gather is a HARD block (not the soft chat warning) — NO LLM-hallucinated
    figure enters Titan's permanent narrative-SELF record (Maker 2026-06-10;
    `_STRICT_CONSISTENCY_CHANNELS`). Returns ``(text, ovg_ok)``; an empty/failed
    compose or a blocked/erroring OVG → ``ovg_ok=False`` so the caller soft-fails
    to the numbers-only minimal grounded entry (INV-SD-2/13).
    """
    if provider is None:
        return "", False
    try:
        text = await provider.complete(
            prompt=prompts["user_prompt"], system=prompts["system_prompt"],
            temperature=_COMPOSE_TEMPERATURE, max_tokens=_COMPOSE_MAX_TOKENS,
            timeout=_COMPOSE_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] compose failed: %s", e)
        return "", False
    text = (text or "").strip()
    if not text:
        return "", False
    if verifier is None:
        return text, False  # no OVG available → soft-fail (fail-closed, INV-SD-2)
    try:
        result = verifier.verify_safety(
            text, channel="soul_diary", injected_context=prompts["user_prompt"])
        return text, bool(getattr(result, "passed", False))
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] OVG verify_safety failed: %s", e)
        return text, False


_MAX_ENGRAM_NAMES = 12


def _gather_engrams_today(target_day: str) -> list[str]:
    """Engram NAMES crystallized during ``target_day`` (UTC) — the §1.1 source.

    Read from the synthesis spine snapshot ``data/spine_snapshot.json`` — the
    atomic JSON synthesis_worker re-exports every ~60s. The worker can NOT open
    the Kuzu spine directly (Kuzu 0.11's ``read_only`` flag still acquires the
    exclusive write-lock vs the live synthesis writer; the JSON snapshot is the
    canonical cross-process read surface — the same one ``/v6/synthesis/engrams``
    reads). Latest version per ``concept_id`` whose ``created_at`` falls in the
    day-window, newest-first, bounded. Soft-fail → []."""
    try:
        from titan_hcl.core.shadow_data_dir import resolve_data_path
        path = resolve_data_path("data/spine_snapshot.json")
        with open(path, "r", encoding="utf-8") as f:
            snap = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as e:
        logger.info("[soul_diary] engram snapshot read failed: %s", e)
        return []
    start, end = _day_window_epochs(target_day)
    latest: dict = {}  # concept_id -> (version, name, created_at)
    for c in (snap.get("concepts") or []):
        try:
            cid = c.get("concept_id")
            ver = int(c.get("version", 0) or 0)
            created_at = float(c.get("created_at", 0) or 0)
            name = (c.get("name") or "").strip()
        except (TypeError, ValueError, AttributeError):
            continue
        if not cid or not name:
            continue
        cur = latest.get(cid)
        if cur is None or ver > cur[0]:
            latest[cid] = (ver, name, created_at)
    todays = [(name, ca) for (_v, name, ca) in latest.values()
              if start <= ca < end]
    todays.sort(key=lambda t: t[1], reverse=True)
    return [name for name, _ca in todays[:_MAX_ENGRAM_NAMES]]


def _gather_memory(shm_reader) -> dict:
    """24h memory/mempool snapshot (G18) ← memory_state.bin (memory_worker)."""
    if shm_reader is None:
        return {}
    try:
        ms = shm_reader.read_memory_state() or {}
    except Exception as e:  # noqa: BLE001
        logger.info("[soul_diary] memory_state read failed: %s", e)
        return {}
    if not ms:
        return {}
    return {
        "persistent": int(ms.get("persistent_count", 0) or 0),
        "mempool": int(ms.get("mempool_size", 0) or 0),
        "effective_24h": round(float(ms.get("effective_nodes_24h", 0) or 0), 2),
        "high_quality": int(ms.get("high_quality_count", 0) or 0),
        "learning_velocity": round(float(ms.get("learning_velocity", 0) or 0), 3),
        "kg_nodes": int(ms.get("kg_node_count", 0) or 0),
        "kg_edges": int(ms.get("kg_edge_count", 0) or 0),
    }


def _gather_social(shm_reader) -> dict:
    """Social snapshot (G18) ← social_graph_state.bin (social_graph_worker) +
    social_perception_state.bin (spirit_worker)."""
    if shm_reader is None:
        return {}
    out: dict = {}
    try:
        sg = shm_reader.read_social_graph_state() or {}
        if sg:
            out.update({
                "users": int(sg.get("users", 0) or 0),
                "edges": int(sg.get("edges", 0) or 0),
                "inspirations": int(sg.get("inspirations", 0) or 0),
                "donations": int(sg.get("donations", 0) or 0),
                "engagement_today": int(sg.get("engagement_ledger_today", 0) or 0),
            })
    except Exception as e:  # noqa: BLE001
        logger.info("[soul_diary] social_graph read failed: %s", e)
    try:
        sp = shm_reader.read_social_perception_state() or {}
        if sp:
            out["sentiment_ema"] = round(float(sp.get("sentiment_ema", 0) or 0), 3)
            out["interaction_rate"] = round(
                float(sp.get("interaction_rate", 0) or 0), 3)
    except Exception as e:  # noqa: BLE001
        logger.info("[soul_diary] social_perception read failed: %s", e)
    return out


def _gather_onchain(shm_reader) -> dict:
    """Metabolic / on-chain snapshot (G18) ← network_state.bin (balance_sol — the
    authoritative RPC balance) + body_state.bin (sol_norm/anchor_fresh; its
    sol_balance is a lagging cache, 0.0 at boot) + metabolism_state.bin
    (tier/balance_pct) — his real metabolic life, what governs him (INV-SD-7).

    SOL source priority = network_state.balance_sol (the direct RPC balance,
    authoritative) over body_state.sol_balance (a derived cache that lags / reads
    0.0 in the boot-grace window). balance_pct is guarded against its out-of-range
    boot sentinel (-1.0) so the entry never renders a nonsensical '-100% energy'."""
    if shm_reader is None:
        return {}
    out: dict = {}
    sol = None
    try:
        ns = shm_reader.read_network_state() or {}
        if ns.get("balance_sol") is not None:
            sol = float(ns.get("balance_sol") or 0)
    except Exception as e:  # noqa: BLE001
        logger.info("[soul_diary] network_state read failed: %s", e)
    try:
        bs = shm_reader.read_body_state() or {}
        if bs:
            # body_state.sol_balance is a fallback only when the RPC balance is
            # absent/zero (it lags and reads 0.0 during boot grace).
            if (sol is None or sol == 0.0) and bs.get("sol_balance"):
                sol = float(bs.get("sol_balance") or 0)
            out["sol_norm"] = round(float(bs.get("sol_norm", 0) or 0), 3)
            if bs.get("anchor_fresh") is not None:
                out["anchor_fresh"] = round(float(bs.get("anchor_fresh") or 0), 3)
    except Exception as e:  # noqa: BLE001
        logger.info("[soul_diary] body_state read failed: %s", e)
    if sol is not None:
        out["sol_balance"] = round(sol, 5)
    try:
        mb = shm_reader.read_metabolism_state() or {}
        if mb:
            if mb.get("tier"):
                out["metabolic_tier"] = str(mb.get("tier"))
            bp = mb.get("balance_pct")
            # Guard the out-of-range boot sentinel (e.g. -1.0); only a sane
            # 0..2 fraction is a real energy reading.
            if bp is not None and 0.0 <= float(bp) <= 2.0:
                out["balance_pct"] = round(float(bp), 3)
    except Exception as e:  # noqa: BLE001
        logger.info("[soul_diary] metabolism_state read failed: %s", e)
    return out


def _gather_felt(shm_reader) -> dict:
    """Felt/neuromod snapshot (G18) ← neuromod_state.bin (dominant modulator) +
    mind_state.bin (mood_valence/mood_label/mood_intensity).

    BUGFIX 2026-06-10: the old inline read treated ``read_neuromod()`` as a flat
    ``{name: level}`` dict with top-level valence/arousal — but the real shape is
    ``{"modulators": {name: {"level": …}}, "age_seconds", "seq"}``, so ``dominant``
    picked ``"age_seconds"``/``"seq"`` and valence/arousal were ALWAYS ``None`` →
    NO felt line ever surfaced live. Valence/mood live in mind_state."""
    if shm_reader is None:
        return {}
    felt: dict = {}
    try:
        nm = shm_reader.read_neuromod() or {}
        mods = (nm or {}).get("modulators") or {}
        if mods:
            dominant = max(
                mods, key=lambda k: float((mods[k] or {}).get("level", 0) or 0))
            felt["dominant"] = dominant
            felt["neuromod_levels"] = {
                k: round(float((v or {}).get("level", 0) or 0), 3)
                for k, v in mods.items()}
    except Exception as e:  # noqa: BLE001
        logger.info("[soul_diary] neuromod read failed: %s", e)
    try:
        mind = shm_reader.read_mind_state() or {}
        if mind:
            if mind.get("mood_valence") is not None:
                felt["valence"] = round(float(mind.get("mood_valence") or 0), 3)
            if mind.get("mood_intensity") is not None:
                felt["intensity"] = round(float(mind.get("mood_intensity") or 0), 3)
            label = (mind.get("mood_label") or "").strip()
            if label and label.lower() != "unknown":
                felt["mood_label"] = label
    except Exception as e:  # noqa: BLE001
        logger.info("[soul_diary] mind_state read failed: %s", e)
    return felt


def _gather_bundle(payload: dict, shm_reader, orchestrator, *,
                   target_day: str, titan_id: str = "", repo_root: str = "") -> dict:
    """② GATHER — assemble the grounded bundle from REAL G18 reads (best-effort).

    Every RFP §1.1 source is wired to its real read surface; each is
    independently guarded so a missing source degrades the entry, never crashes
    the cascade (INV-SD-7/13):
      · sovereignty   ← synthesis sovereignty_readout (read_rolling_sovereignty)
      · outcome       ← the MEDITATION_COMPLETE payload (this meditation)
      · felt          ← neuromod_state + mind_state SHM (_gather_felt)
      · engrams_today ← spine_snapshot.json day-window (_gather_engrams_today)
      · memory        ← memory_state SHM (_gather_memory)
      · social        ← social_graph + social_perception SHM (_gather_social)
      · onchain       ← body + network + metabolism SHM (_gather_onchain)
      · infra (§P5)   ← read-only self-inspection (journal errors + error→code)
    """
    sovereignty: dict = {}
    try:
        from titan_hcl.synthesis.sovereignty_readout import read_rolling_sovereignty
        sovereignty = read_rolling_sovereignty() or {}
    except Exception as e:  # noqa: BLE001
        logger.info("[soul_diary] sovereignty read failed: %s", e)

    outcome = {
        "promoted": int(payload.get("promoted", 0) or 0),
        "pruned": int(payload.get("pruned", 0) or 0),
        "epoch": payload.get("epoch"),
    }

    felt = _gather_felt(shm_reader)
    engrams_today = _gather_engrams_today(target_day)
    memory = _gather_memory(shm_reader)
    social = _gather_social(shm_reader)
    onchain = _gather_onchain(shm_reader)

    # P5 — scaffolding self-inspection (read-only, bounded, soft-fail; INV-SD-9).
    infra: dict = {}
    if titan_id and repo_root:
        try:
            from titan_hcl.core import self_inspect
            obs = self_inspect.gather_self_observations(titan_id, repo_root=repo_root)
            infra = {
                "summary": self_inspect.summarize_observations(obs),
                "structure": obs.get("structure") or {},
                "correlations": obs.get("correlations") or [],
            }
        except Exception as e:  # noqa: BLE001
            logger.info("[soul_diary] self-inspection failed: %s", e)

    return orchestrator.build_bundle(
        sovereignty=sovereignty, outcome=outcome, felt=felt,
        engrams_today=engrams_today, memory=memory, social=social,
        onchain=onchain, infra=infra)


def _enrich_synthesis(send_queue, src: str, today: str, entry: str) -> None:
    """⑥ ENRICH (INV-SD-15) — publish the diary as a self-domain thought to the
    mempool (one-way, no-RPC; G19). ``memory_worker`` promotes it at the next
    meditation/dream boundary into DuckDB+FAISS+Kuzu so Titan *recalls* his
    narrative path; consolidation classifies it into the (now clean) ``"self"``
    domain (consolidation_defaults split, P2). Soft-fail — never blocks the
    cascade (INV-SD-13); the private floor (persist+hash) is already committed."""
    try:
        send_queue.put({
            "type": bus.MEMORY_MEMPOOL_ADD,
            "src": src, "dst": "memory",
            "ts": time.time(),
            "payload": {
                "user_prompt": f"Soul-diary reflection — {today}",
                "agent_response": entry,
                "user_identifier": "Titan",
                "source": "soul_diary",
                "tags": ["soul_diary", "domain:self"],
            },
        })
        logger.info("[soul_diary] ⑥ enriched synthesis (mempool add → self domain) "
                    "for %s", today)
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] enrich (mempool add) failed: %s", e)


def _enrich_self_inspection(send_queue, src: str, today: str, infra: dict) -> None:
    """P5 ENRICH (INV-SD-9) — publish the self-inspection observation as its OWN
    `domain="self"` thought (`source="self_inspect"`), so his code/error/substrate
    observations become recallable self-Engrams (auto-linked to the `Self` hub via
    the P3a `SELF_HAS_ENGRAM` hook), distinct from the narrative diary entry. This
    is the BRAIN_DOMAIN_SELF seed. Soft-fail — never blocks the cascade."""
    summary = ((infra or {}).get("summary") or "").strip()
    if not summary:
        return
    try:
        send_queue.put({
            "type": bus.MEMORY_MEMPOOL_ADD,
            "src": src, "dst": "memory",
            "ts": time.time(),
            "payload": {
                "user_prompt": f"What I observed about my own substrate — {today}",
                "agent_response": summary,
                "user_identifier": "Titan",
                "source": "self_inspect",
                "tags": ["soul_diary", "self_inspect", "domain:self"],
            },
        })
        logger.info("[soul_diary] P5 self-inspection enriched "
                    "(mempool add, source=self_inspect) for %s", today)
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] self-inspection enrich failed: %s", e)


def _anchor_main_chain(send_queue, src: str, today: str, row: dict) -> None:
    """⑦ ANCHOR (INV-SD-17) — seal the diary's cumulative-hash head on the
    main/genesis chain (``fork="main"`` → ``FORK_MAIN=0``, the SELF journey — NOT
    the ACT-R declarative/procedural/episodic forks). A hash POINTER; the entry
    text stays in outer memory. ``significance=0.85`` (a once-daily SELF-journey
    milestone) clears the main-fork PoT threshold (0.20); empty ``neuromods``
    take the tonic 0.5 baseline (create_pot). Soft-fail (INV-SD-13)."""
    cumulative_hash = (row or {}).get("cumulative_hash", "")
    entry_hash = (row or {}).get("entry_hash", "")
    if not cumulative_hash:
        logger.warning("[soul_diary] anchor skipped — no cumulative_hash in ledger row")
        return
    try:
        from titan_hcl.synthesis.outer_memory_writer import (
            OuterMemoryEvent, OuterMemoryWriter,
        )
        writer = OuterMemoryWriter(send_queue, src=src)
        writer.emit(OuterMemoryEvent(
            fork="main",
            thought_type="dailyDiary",
            source="soul_diary",
            content={"date": today, "entry_hash": entry_hash,
                     "cumulative_hash": cumulative_hash},
            tags=["soul_diary", "dailyDiary", f"date:{today}"],
            significance=0.85,   # high — clears main PoT (0.20); SELF-journey milestone
            novelty=0.5,
            coherence=0.8,
        ))
        logger.info("[soul_diary] ⑦ anchored cumulative_hash=%s on main chain "
                    "(fork=main) for %s", cumulative_hash[:12], today)
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] anchor (main-chain emit) failed: %s", e)


def _render_art(orchestrator, row: dict, bundle: dict, target_day: str) -> None:
    """P7 ⑨ — render the day's felt-driven procedural art (INV-SD-4), seeded by
    the entry's cumulative_hash (deterministic + cryptographically tied to the
    entry), and record its path on the ledger row. His TRUE felt state
    (valence/arousal/neuromod profile/coherence) drives the render — NO image-LLM.
    The art feeds the P8 NFT + P10 X post. Soft-fail — never blocks the cascade
    (INV-SD-13); a missing renderer just leaves art_path unset."""
    cumulative_hash = (row or {}).get("cumulative_hash", "")
    if not cumulative_hash:
        return
    try:
        from titan_hcl.core.shadow_data_dir import resolve_data_path
        from titan_hcl.core.soul_diary import ART_REL_DIR, _ART_RESOLUTION
        from titan_hcl.expressive.art import ProceduralArtGen
        felt = orchestrator.build_art_felt(bundle)
        gen = ProceduralArtGen(output_dir=resolve_data_path(ART_REL_DIR))
        art_path = gen.generate_flow_field(
            cumulative_hash, orchestrator.art_complexity(bundle), 5,
            resolution=_ART_RESOLUTION, felt=felt)
        if art_path:
            orchestrator.record_art(target_day, art_path)
            if isinstance(row, dict):
                row["art_path"] = art_path   # keep the in-memory row fresh for P8 ⑩
            logger.info("[soul_diary] ⑨ rendered felt-art (felt_driven=%s) for %s: %s",
                        felt is not None, target_day, art_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] art render failed: %s", e)


def _build_chain_provider(config: dict):
    """Construct the data-plane ChainProvider for the Arweave upload from config
    (a seam — monkeypatched in tests). Devnet → local pseudo-tx; mainnet → Irys."""
    net = (config or {}).get("network", {}) or {}
    from titan_hcl.chain.provider import ArweaveChainProvider
    return ArweaveChainProvider(
        keypair_path=net.get("wallet_keypair_path", "") or "",
        network=net.get("solana_network", "devnet"),
        rpc_url=net.get("premium_rpc_url") or "")


def _build_network_client(config: dict):
    """Construct the trust-plane HybridNetworkClient (holds the wallet keypair +
    signs the mint) from config (a seam — monkeypatched in tests)."""
    from titan_hcl.core.network import HybridNetworkClient
    return HybridNetworkClient(config=(config or {}).get("network", {}) or {})


async def _upload_and_mint(*, config, date, entry_hash, cumulative_hash,
                           distillation, sovereignty, felt, art_path,
                           sovereignty_idx, total_nodes) -> dict:
    """Upload the art + the rich metadata JSON to Arweave (ChainProvider.put →
    real ``ar://`` uri) then mint via ``daily_nft.mint_epoch_nft``. Returns
    ``{nft_addr, arweave_uri}`` (nft_addr None if the mint no-ops)."""
    from titan_hcl.logic import daily_nft
    provider = _build_chain_provider(config)
    art_uri = None
    if art_path:
        try:
            art_tx = await provider.put(
                art_path, content_type="image/jpeg",
                tags={"app": "titan", "kind": "soul_diary_art", "date": date})
            art_uri = f"ar://{art_tx}"
        except Exception as e:  # noqa: BLE001
            logger.info("[soul_diary] art upload skipped: %s", e)
    meta = daily_nft.build_soul_diary_nft_metadata(
        date=date, entry_hash=entry_hash, cumulative_hash=cumulative_hash,
        distillation=distillation, sovereignty=sovereignty, felt=felt,
        art_uri=art_uri)
    meta_tx = await provider.put(
        json.dumps(meta, ensure_ascii=False).encode("utf-8"),
        content_type="application/json",
        tags={"app": "titan", "kind": "soul_diary_nft", "date": date})
    permanent_url = f"ar://{meta_tx}"
    network = _build_network_client(config)
    epoch = int(_day_window_epochs(date)[0])
    addr = await daily_nft.mint_epoch_nft(
        network, epoch=epoch, sovereignty_idx=sovereignty_idx,
        diary_entry=distillation, total_nodes=total_nodes,  # distillation = SANITIZED
        art_path=art_path, permanent_url=permanent_url)
    return {"nft_addr": addr, "arweave_uri": permanent_url}


def _mint_daily_nft(orchestrator, row: dict, bundle: dict, target_day: str, *,
                    config: dict, titan_id: str = "") -> None:
    """P8 ⑩ — mint the day's DailyNFT (per-Titan gated, INV-SD-11). Self-contained
    chain I/O (Maker 2026-06-10 — isolated from the backup-redesign): builds its
    own ChainProvider + HybridNetworkClient, uploads the art + a rich metadata
    JSON carrying BOTH hashes + the privacy-clean distillation (INV-SD-10/3) to
    Arweave, mints via the (re-homed) ``daily_nft.mint_epoch_nft`` with the real
    uri, and records nft_addr + arweave_uri on the ledger row. **Mainnet T1 mints
    NOTHING** (mint_enabled=false); devnet T2/T3 mint. Soft-fail — never blocks
    the cascade (INV-SD-13); the triple-anchor's other two roots (main-chain tx +
    durable ledger) are already committed."""
    if not orchestrator.mint_enabled(config):
        logger.info("[soul_diary] ⑩ DailyNFT mint gated OFF (mint_enabled=false) — "
                    "skipping mint for %s (INV-SD-11)", target_day)
        return
    cumulative_hash = (row or {}).get("cumulative_hash", "")
    if not cumulative_hash:
        return
    try:
        result = asyncio.run(_upload_and_mint(
            config=config, date=target_day,
            entry_hash=(row or {}).get("entry_hash", ""),
            cumulative_hash=cumulative_hash,
            distillation=(row or {}).get("distillation", "") or "",
            sovereignty=bundle.get("sovereignty") or {},
            felt=bundle.get("felt") or {},
            art_path=(row or {}).get("art_path"),
            sovereignty_idx=round(
                float((bundle.get("sovereignty") or {}).get("s", 0) or 0) * 100.0, 2),
            total_nodes=int((bundle.get("memory") or {}).get("kg_nodes", 0) or 0)))
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] DailyNFT mint failed: %s", e)
        return
    if result and result.get("nft_addr"):
        orchestrator.record_nft(target_day, nft_addr=result["nft_addr"],
                                arweave_uri=result.get("arweave_uri"))
        logger.info("[soul_diary] ⑩ minted DailyNFT %s (uri=%s) for %s",
                    result["nft_addr"], result.get("arweave_uri"), target_day)
    else:
        logger.info("[soul_diary] ⑩ mint enabled but no asset returned (no keypair / "
                    "soft-fail) for %s", target_day)


def _author_daily_entry(payload: dict, *, orchestrator, provider, verifier,
                        shm_reader, send_queue, src: str,
                        titan_id: str = "", repo_root: str = "",
                        config: dict = None) -> bool:
    """Run the full pipeline for one MEDITATION_COMPLETE. Returns True if a
    diary entry was authored (or correctly skipped), False on hard error."""
    target_day = _completed_day()  # the just-completed UTC day (RFP §6.2)
    if not orchestrator.should_author(target_day):
        return True  # already wrote that day — latch closed (INV-SD-5)

    bundle = _gather_bundle(payload, shm_reader, orchestrator,
                            target_day=target_day, titan_id=titan_id,
                            repo_root=repo_root)
    if not orchestrator.has_activity(bundle):
        logger.info("[soul_diary] no-op day (no activity) for %s — latching "
                    "without entry", target_day)
        orchestrator.mark_authored(target_day)
        return True

    prompts = orchestrator.build_compose_prompts(bundle)
    raw, ovg_ok = asyncio.run(_compose_diary(provider, verifier, prompts))
    authored = bool(ovg_ok and raw)
    if authored:
        # P6 ⑨ — the same grounded compose carries a "---SHARE---" public variant
        # (§6.3); split it off the private entry (only the entry is persisted+hashed).
        entry, share = orchestrator.split_entry_and_share(raw)
    else:
        entry, share = orchestrator.minimal_entry(bundle), ""
        logger.warning("[soul_diary] authoring soft-fell to minimal grounded entry "
                       "(ovg_ok=%s, text=%d chars)", ovg_ok, len(raw))

    # P6 ⑨ — privacy-clean public projection (INV-SD-3). The private `entry` is
    # persisted + hashed RAW below; only these derived public artifacts (the
    # X-postable distillation + the archive's sanitized full entry) are scrubbed
    # by the fail-closed sanitizer. `art_path` lands in P7; nft/arweave in P8.
    distillation, public_entry, redactions = \
        orchestrator.build_public_artifacts(entry, share)

    try:
        orchestrator.persist(entry)                      # ④ titan_chronicles.md → titan.md
        row = orchestrator.record_hash(                  # ⑤ hash-chain ledger + P6 public row
            target_day, entry, distillation=distillation,
            public_entry=public_entry, redactions=redactions)
        orchestrator.mark_authored(target_day)           # ① latch
    except Exception as e:  # noqa: BLE001
        logger.error("[soul_diary] persist/hash failed: %s", e, exc_info=True)
        return False

    # P2 — the committed entry now ENRICHES synthesis (he remembers + recalls his
    # narrative path) and ANCHORS the SELF journey on the main chain. Each
    # soft-fails independently (INV-SD-13); the private floor above is durable.
    _enrich_synthesis(send_queue, src, target_day, entry)         # ⑥ INV-SD-15
    _enrich_self_inspection(send_queue, src, target_day,
                            bundle.get("infra") or {})            # P5 INV-SD-9
    _anchor_main_chain(send_queue, src, target_day, row)          # ⑦ INV-SD-17
    _render_art(orchestrator, row, bundle, target_day)            # ⑨ P7 INV-SD-4
    _mint_daily_nft(orchestrator, row, bundle, target_day,        # ⑩ P8 INV-SD-11
                    config=config or {}, titan_id=titan_id)

    logger.info("[soul_diary] authored daily entry for %s (%d chars, authored=%s, "
                "public_redactions=%s)", target_day, len(entry), authored, redactions)
    return True


def soul_diary_worker_main(recv_queue, send_queue, name: str,
                           config: dict) -> None:
    """Main loop for the soul-diary worker subprocess (§1.0 P1)."""
    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    full_config = config or {}

    _state_writer = None
    try:
        from titan_hcl.core.module_state import BootPriority, ModuleStateWriter
        _state_writer = ModuleStateWriter(
            module_name=name, layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT)
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning("[soul_diary] ModuleStateWriter init failed: %s", _sw_err)

    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = ((full_config.get("info_banner", {}) or {}).get("titan_id")
                or resolve_titan_id())
    logger.info("[soul_diary] Booting — titan_id=%s", titan_id)

    from titan_hcl.core.soul_diary import SoulDiaryOrchestrator
    orchestrator = SoulDiaryOrchestrator()

    # Own LLM call (self-contained narration; INV-SD-14). Missing key/import →
    # provider None → authoring soft-fails to the minimal grounded entry.
    provider = None
    try:
        inference_cfg = full_config.get("inference", {}) or {}
        provider_name = _resolve_provider_name(inference_cfg)
        from titan_hcl.inference import get_provider
        provider = get_provider(provider_name, inference_cfg)
        logger.info("[soul_diary] inference provider=%s", provider_name)
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] inference provider unavailable "
                       "(diary will use minimal grounded entries): %s", e)

    # Own OVG truth-gate (lightweight, regex-based).
    verifier = None
    try:
        from titan_hcl.logic.output_verifier import OutputVerifier
        verifier = OutputVerifier(titan_id=titan_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] OutputVerifier init failed: %s", e)

    # Felt/neuromod SHM reads (G18).
    shm_reader = None
    try:
        from titan_hcl.api.shm_reader_bank import ShmReaderBank
        shm_reader = ShmReaderBank()
    except Exception as e:  # noqa: BLE001
        logger.info("[soul_diary] ShmReaderBank unavailable: %s", e)

    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
        except Exception:  # noqa: BLE001
            pass
    logger.info("[soul_diary] Ready — listening for MEDITATION_COMPLETE")

    last_heartbeat = 0.0
    processed = 0
    errors = 0

    while True:
        now = time.time()
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
            try:
                send_queue.put({
                    "type": bus.MODULE_HEARTBEAT, "src": name, "dst": "guardian",
                    "payload": {"alive": True, "ts": now, "processed": processed,
                                "errors": errors}, "ts": now})
            except Exception:  # noqa: BLE001
                pass
            if _state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
                try:
                    _state_writer.heartbeat()
                except Exception:  # noqa: BLE001
                    pass
            last_heartbeat = now

        try:
            msg = recv_queue.get(timeout=0.5)
        except Empty:
            continue
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(msg, dict):
            continue

        msg_type = msg.get("type")

        try:
            from titan_hcl.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:  # noqa: BLE001
            pass

        if msg_type == bus.MODULE_PROBE_REQUEST:
            try:
                from titan_hcl.core.probe_dispatcher import handle_module_probe_request
                handle_module_probe_request(
                    msg, send_queue=send_queue, module_name=name,
                    ready=_WORKER_READY)
            except Exception:  # noqa: BLE001
                pass
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[soul_diary] MODULE_SHUTDOWN — exiting")
            if _state_writer is not None:
                try:
                    _state_writer.write_state("stopping")
                except Exception:  # noqa: BLE001
                    pass
            break

        if msg_type == bus.MEDITATION_COMPLETE:
            payload = msg.get("payload") or {}
            try:
                if _author_daily_entry(payload, orchestrator=orchestrator,
                                       provider=provider, verifier=verifier,
                                       shm_reader=shm_reader,
                                       send_queue=send_queue, src=name,
                                       titan_id=titan_id,
                                       repo_root=project_root,
                                       config=full_config):
                    processed += 1
                else:
                    errors += 1
            except Exception as e:  # noqa: BLE001
                errors += 1
                logger.error("[soul_diary] MEDITATION_COMPLETE handling failed: %s",
                             e, exc_info=True)
            continue
