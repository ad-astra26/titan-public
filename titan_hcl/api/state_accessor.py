"""
titan_hcl/api/state_accessor.py — TitanStateAccessor + sub-accessors.

The single object endpoint code talks to. Composes:
  - ShmReaderBank — direct SHM reads (zero IPC) for L0+L1+L2 state slots
  - CommandSender — fire-and-forget bus.publish for write-side ops
  - ConfigAccessor — static config read once at boot

Endpoint code rewrite:
  state = request.app.state.titan_state
  bal = state.network.balance
  trinity = state.trinity.read()
  state.commands.reload_api()

Future transport changes (Phase C Rust kernel, gRPC, shared-memory queues)
touch only this file. Endpoint code stays put.

Phase A/B/C/D state-read unification (D-SPEC-71 → D-SPEC-82): every
sub-accessor is now SHM-direct per Preamble G18. The bus-cache → CachedState
pipeline is RETIRED; the legacy BusSubscriber + kernel snapshot push tick
no longer exist.
"""
from __future__ import annotations

import logging
from typing import Any

from titan_hcl.api.command_sender import CommandSender
from titan_hcl.api.shm_reader_bank import ShmReaderBank

logger = logging.getLogger(__name__)


# ── Base mixin for graceful fallback ──────────────────────────────────


class _SubAccessorBase:
    """Base class for typed sub-accessors. Provides __getattr__ fallback
    so endpoint code calling unrecognized methods doesn't 500 — instead
    we synthesize an empty-default getter per Preamble G18.
    """

    def __getattr__(self, name: str):
        # Skip Python internals + private attrs (real AttributeError).
        if name.startswith("_") or name.startswith("__"):
            raise AttributeError(name)
        cls = type(self).__name__.replace("Accessor", "").lower()
        return _CacheGetter(cls)


# ── Sub-accessors ─────────────────────────────────────────────────────


class NetworkAccessor(_SubAccessorBase):
    """Solana network state — SHM-direct via network_state.bin (publisher:
    titan_HCL kernel monitor_tick loop per SPEC §7.1 / D-SPEC-70). Phase A.4
    migration: bus-cache retired per Preamble G18."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    @property
    def balance(self) -> float:
        """Current SOL balance — last cached value (typically <30s old)."""
        state = self._shm.read_network_state() or {}
        return float(state.get("balance_sol", 0.0) or 0.0)

    @property
    def pubkey(self) -> str:
        state = self._shm.read_network_state() or {}
        return str(state.get("pubkey", "") or "")

    @property
    def rpc_urls(self) -> list[str]:
        state = self._shm.read_network_state() or {}
        urls = state.get("rpc_urls", []) or []
        return list(urls) if isinstance(urls, list) else []

    @property
    def premium_rpc(self) -> str | None:
        state = self._shm.read_network_state() or {}
        return state.get("premium_rpc")

    @property
    def rpc_endpoint(self) -> str:
        state = self._shm.read_network_state() or {}
        ep = state.get("rpc_endpoint", "")
        if ep:
            return str(ep)
        urls = self.rpc_urls
        return urls[0] if urls else "https://api.mainnet-beta.solana.com"

    @property
    def is_available(self) -> bool:
        state = self._shm.read_network_state() or {}
        return bool(state.get("network_available", False))

    def get_raw_account_data(self, pda: str) -> dict | None:
        """Return cached raw account data for a vault PDA, if recently fetched.
        Endpoints that need fresh data publish SOLANA_ACCOUNT_REFRESH_REQUEST
        via state.commands.publish(...) and read on next bus tick.
        Phase A.4: served from network_state.bin recent_account_data field."""
        state = self._shm.read_network_state() or {}
        accounts = state.get("recent_account_data", {}) or {}
        return accounts.get(pda)


class TrinityAccessor(_SubAccessorBase):
    """Trinity 162D state — direct shm read. Body/mind/spirit composite."""


    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def read(self) -> dict | None:
        return self._shm.read_trinity()

    @property
    def is_available(self) -> bool:
        return self._shm.read_trinity() is not None


class NeuromodAccessor(_SubAccessorBase):
    """Neuromodulator levels — direct shm read."""


    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def read(self) -> dict | None:
        return self._shm.read_neuromod()

    def level(self, name: str) -> float:
        data = self._shm.read_neuromod()
        if data is None:
            return 0.0
        mods = data.get("modulators", {})
        entry = mods.get(name, {})
        return float(entry.get("level", 0.0)) if isinstance(entry, dict) else 0.0


class EpochAccessor(_SubAccessorBase):
    """Consciousness epoch counter — direct shm read."""


    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def current(self) -> int:
        data = self._shm.read_epoch()
        return data.get("epoch", 0) if data else 0

    def read(self) -> dict | None:
        return self._shm.read_epoch()


class SpiritAccessor(_SubAccessorBase):
    """Inner Spirit state — SHM-direct (45D fast tensor + composite reads
    from canonical Rust L0+L1 slots per Phase B D-SPEC-78)."""


    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def read_45d(self) -> dict | None:
        """SAT-15 + CHIT-15 + ANANDA-15 fast tensor (S3b)."""
        return self._shm.read_inner_spirit_45d()

    def read_inner_trinity(self) -> dict:
        """Composite inner-trinity payload — SHM-direct composition from
        canonical Rust L1 slots per Preamble G15+G18 (state registry is the
        matrix). Phase A.4 gap-2 closure: bus-cache retired.

        Composes 7 SHM reads into the dict shape kernel snapshot used to
        provide via INNER_TRINITY_UPDATED bus event:
          inner_body_5d.bin   → body
          inner_mind_15d.bin  → mind
          inner_spirit_45d.bin → spirit
          outer_body_5d.bin   → outer_body
          outer_mind_15d.bin  → outer_mind
          outer_spirit_45d.bin → outer_spirit
          topology_30d.bin    → topology

        Any missing slot returns empty entry (cold-boot tolerant).
        """
        _shm = self._shm
        return {
            "body": _shm.read_inner_body_5d() or {},
            "mind": _shm.read_inner_mind_15d() or {},
            "spirit": _shm.read_inner_spirit_45d() or {},
            "outer_body": _shm.read_outer_body_5d() or {},
            "outer_mind": _shm.read_outer_mind_15d() or {},
            "outer_spirit": _shm.read_outer_spirit_45d() or {},
            "topology": _shm.read_topology_30d() or {},
        }

    def get_trinity(self) -> dict:
        """Legacy method name — callers from old endpoint code expect dict."""
        return self.read_inner_trinity()

    def get_coordinator(self) -> dict:
        """Composite coordinator view — SHM-direct overlay per Preamble G18.

        Phase A.3 migration (rFP_phase_c_state_read_unification_l0_l1_canonical
        2026-05-17): bus-cache overlays retired wherever the canonical SHM
        slot exists. Each overlay reads its source slot via ShmReaderBank
        per SPEC §7.1.

        Still cache-backed (pending Phase A.4 NEW SHM slot drafts):
          - kernel baseline coord dict (spirit.coordinator) — until cognitive_worker's
            coord summary is migrated to a dedicated SHM slot
          - `msl`, `meta_reasoning`, `reasoning` — 🆕 NEW slots per rFP §A.1.3

        Phase B (2026-05-18) SHIPPED: resonance/unified_spirit now from
        canonical Rust L0+L1 slots (resonance_metadata.bin + unified_spirit_metadata.bin).
        nervous_system overlay comes from the canonical NS L2 slots
        (titanvm_registers.bin / ns_program_urgencies_input.bin owned by
        ns_worker per SPEC §1 glossary). spirit_supplemental_state.bin
        retired — coordinator baseline now starts empty and is built up
        purely from per-overlay reads of canonical slots.
        """
        # Phase B.5 — coordinator baseline starts empty; canonical slot
        # overlays (chi/neuromod/topology/etc.) fill it. The previous
        # spirit_supplemental "coordinator" section (27 keys) retired with
        # SpiritSupplementalStatePublisher; each key now flows from its
        # owning worker's L2 slot or a Rust L1 slot per SPEC §9.A.
        coord: dict[str, Any] = {}

        # Phase A.3 — SHM-direct overlays (canonical Python L2 + Rust slots)
        _shm = self._shm
        _shm_overlays: list[tuple[str, Any]] = [
            ("chi", _shm.read_chi()),
            ("pi_heartbeat", _shm.read_pi_heartbeat()),
            ("dreaming", _shm.read_dream_state()),
            ("expression_composites", _shm.read_expression_state()),
            ("neuromodulators", _shm.read_neuromod()),
            ("language", _shm.read_language_state()),
            ("topology", _shm.read_topology_30d()),
        ]
        for coord_key, payload in _shm_overlays:
            if isinstance(payload, dict) and payload:
                coord[coord_key] = payload

        # Phase A.4 — SHM-direct overlays for the 3 previously bus-cache
        # paths. Producers per SPEC §7.1 / D-SPEC-70:
        #   msl_state.bin            ← cognitive_worker (MSL engine)
        #   meta_reasoning_state.bin ← cognitive_worker (MetaReasoningEngine)
        #   reasoning_state.bin      ← cognitive_worker (ReasoningEngine)
        _phase_a4_overlays: list[tuple[str, Any]] = [
            ("msl", _shm.read_msl_state()),
            ("meta_reasoning", _shm.read_meta_reasoning_state()),
            ("reasoning", _shm.read_reasoning_state()),
        ]
        for coord_key, payload in _phase_a4_overlays:
            if isinstance(payload, dict) and payload:
                coord[coord_key] = payload

        # Phase B.5 (2026-05-18): SHM overlay for neural_nervous_system +
        # resonance + unified_spirit. resonance/unified_spirit read from
        # the canonical Rust L0+L1 slots (resonance_metadata.bin +
        # unified_spirit_metadata.bin via titan-unified-spirit-rs
        # MetadataPublisher per B.0). neural_nervous_system overlay
        # sourced from ns_worker's titanvm_registers.bin (the canonical
        # NS L2 slot per SPEC §1 glossary) — replaces the retired
        # spirit_supplemental.nervous_system section. /status lifetime
        # fields (neural_maturity / neural_train_steps) continue to read
        # from coord["neural_nervous_system"] dict.
        try:
            ns_payload = self._shm.read_titanvm_registers()
            if isinstance(ns_payload, dict) and ns_payload:
                coord["neural_nervous_system"] = ns_payload
            res = self._shm.read_resonance_metadata()
            if isinstance(res, dict) and res:
                coord["resonance"] = res
            us = self._shm.read_unified_spirit_metadata()
            if isinstance(us, dict) and us:
                coord["unified_spirit"] = us
        except Exception as _shm_err:
            # SHM overlay is defensive; if any reader fails, keep the
            # cache-only path. Log once at INFO so it's visible without
            # spamming on persistent SHM unavailability.
            import logging
            logging.getLogger(__name__).debug(
                "[SpiritAccessor.get_coordinator] SHM overlay failed: %s",
                _shm_err)

        return coord

    def get_v4_state(self) -> dict:
        """Composite V4 Time Awareness state — SHM-direct composition per
        Preamble G18.

        Phase B.5 (2026-05-18): resonance + unified_spirit now from the
        canonical Rust L0+L1 slots (resonance_metadata.bin +
        unified_spirit_metadata.bin via titan-unified-spirit-rs
        MetadataPublisher per B.0); consciousness now from
        unified_spirit_metadata.latest_epoch (Rust). spirit_supplemental
        fallback path retired with its publisher.
        """
        _shm = self._shm
        epoch_payload = _shm.read_epoch() or {}
        unified_spirit = _shm.read_unified_spirit_metadata() or {}
        return {
            "sphere_clock": _shm.read_sphere_clocks() or {},
            "chi": _shm.read_chi() or {},
            "topology": _shm.read_topology_30d() or {},
            "epoch": int(epoch_payload.get("epoch", 0) or 0),
            "neuromodulators": _shm.read_neuromod() or {},
            "resonance": _shm.read_resonance_metadata() or {},
            "unified_spirit": unified_spirit,
            "consciousness": unified_spirit.get("latest_epoch") or {},
            "filter_down": _shm.read_filter_down_state() or {},
        }

    def get_sphere_clocks(self) -> dict:
        """SHM-direct via sphere_clocks.bin (Rust L1 producer per SPEC §9.A
        line 1243 / §7.1 row 763). Phase A.4 gap-2 closure: bus-cache
        fallback retired."""
        return self._shm.read_sphere_clocks() or {}

    def get_nervous_system(self) -> dict:
        """V5 Neural NervousSystem stats. Phase B.5 (2026-05-18) — sourced
        from canonical ns_worker L2 slot ``titanvm_registers.bin`` per
        SPEC §1 glossary (ns_worker owns the NS programs + V5 NN state
        tracking). spirit_supplemental fallback retired with its
        publisher. Empty dict if SHM unavailable."""
        return self._shm.read_titanvm_registers() or {}

    def get_expression_composites(self) -> dict:
        """Read 6 expression composites (SPEAK/ART/MUSIC/SOCIAL/KIN/LONGING)
        SHM-direct via expression_state.bin. Phase A.3 migration: bus-cache
        retired per Preamble G18 (publisher: expression_worker per SPEC §1
        glossary / D-SPEC-53 v1.7.4)."""
        return self._shm.read_expression_state() or {}

    def get_resonance(self) -> dict:
        """ResonanceDetector stats. Phase B.5 (2026-05-18) — canonical
        SHM source ``resonance_metadata.bin`` (Rust-owned per B.0;
        titan-unified-spirit-rs MetadataPublisher; G21 single-writer).
        Empty dict if SHM unavailable."""
        return self._shm.read_resonance_metadata() or {}

    def get_unified_spirit(self) -> dict:
        """UnifiedSpirit stats (epoch_count, velocity, stale,
        focus_multiplier, full_130dt, latest_epoch, etc.). Phase B.5
        (2026-05-18) — canonical SHM source ``unified_spirit_metadata.bin``
        flipped to Rust ownership at B.0 (titan-unified-spirit-rs
        MetadataPublisher; G21 single-writer; byte-identical schema with
        r4/r6 precision rounded). Empty dict if SHM unavailable."""
        return self._shm.read_unified_spirit_metadata() or {}


class BodyAccessor(_SubAccessorBase):
    """Body tensor + sense state — SHM-direct via body_state.bin (Session 4
    publisher: body_worker). Phase A.3 migration: bus-cache retired per
    Preamble G18 (state transport is SHM, never bus). Source-of-truth slot
    schema per SPEC §7.1 row 790."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def get_body_tensor(self) -> dict:
        return self._shm.read_body_state() or {}


class MindAccessor(_SubAccessorBase):
    """Mind tensor — SHM-direct via mind_state.bin (Session 4 publisher:
    mind_worker). Phase A.3 migration: bus-cache retired per Preamble G18.
    Source-of-truth slot schema per SPEC §7.1 row 789."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def get_mind_tensor(self) -> dict:
        return self._shm.read_mind_state() or {}


class IdentityAccessor(_SubAccessorBase):
    """Titan identity — SHM-direct. titan_id, maker_pubkey,
    kernel_instance_nonce."""


    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    @property
    def titan_id(self) -> str:
        data = self._shm.read_identity()
        if data:
            return data.get("titan_id", "")
        return self._shm.titan_id

    @property
    def maker_pubkey(self) -> str:
        data = self._shm.read_identity()
        if data and data.get("maker_pubkey"):
            return data["maker_pubkey"]
        # Phase A.4 fallback — SHM-direct via soul_state.bin (publisher:
        # sovereignty_worker per D-SPEC-70). Replaces soul.state bus-cache.
        soul = self._shm.read_soul_state() or {}
        return str(soul.get("maker_pubkey", ""))

    @property
    def kernel_instance_nonce(self) -> str:
        data = self._shm.read_identity()
        return data.get("kernel_instance_nonce", "") if data else ""


class SoulAccessor(_SubAccessorBase):
    """Soul governance state — SHM-direct via soul_state.bin (publisher:
    sovereignty_worker per SPEC §7.1 / D-SPEC-70). Phase A.4 migration:
    bus-cache retired per Preamble G18. Includes maker_pubkey,
    nft_address, current_gen, active_directives."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    @property
    def maker_pubkey(self) -> str:
        soul = self._shm.read_soul_state() or {}
        return str(soul.get("maker_pubkey", ""))

    @property
    def nft_address(self) -> str:
        soul = self._shm.read_soul_state() or {}
        return str(soul.get("nft_address", ""))

    @property
    def current_gen(self) -> int:
        soul = self._shm.read_soul_state() or {}
        return int(soul.get("current_gen", 0))

    def get_active_directives(self) -> list:
        soul = self._shm.read_soul_state() or {}
        return list(soul.get("active_directives", []))


class CGNAccessor(_SubAccessorBase):
    """CGN state — SHM-direct via cgn_engine_state.bin (publisher:
    cgn_worker per SPEC §7.1 / D-SPEC-70). Phase A.4 migration: bus-cache
    retired per Preamble G18. Sibling to cgn_live_weights.bin (tensor)
    + cgn_beta_state.bin (8-float per-consumer reward EMA)."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def get_stats(self) -> dict:
        return self._shm.read_cgn_engine_state() or {}


class ReasoningAccessor(_SubAccessorBase):
    """Reasoning engine state — SHM-direct via reasoning_state.bin (publisher:
    cognitive_worker per SPEC §7.1 / D-SPEC-70). Phase A.4 migration:
    bus-cache retired per Preamble G18."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def get_stats(self) -> dict:
        return self._shm.read_reasoning_state() or {}


class DreamingAccessor(_SubAccessorBase):
    """Dreaming state — SHM-direct via dream_state.bin (publisher:
    dream_state_worker per SPEC §7.1 row 801 / D-SPEC-56). Phase A.3
    migration: bus-cache retired per Preamble G18."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def get_state(self) -> dict:
        return self._shm.read_dream_state() or {}

    @property
    def is_dreaming(self) -> bool:
        return bool(self.get_state().get("is_dreaming", False))


class GuardianAccessor(_SubAccessorBase):
    """Guardian/module state — SHM-direct via guardian_state.bin (publisher:
    guardian — Python L1 supervisor per SPEC §7.1 / D-SPEC-70). Phase A.4
    migration: bus-cache retired per Preamble G18. get_status returns the
    per-module dict in the same shape callers expect (compatible with
    earlier bus-cache payload)."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def get_status(self) -> dict:
        payload = self._shm.read_guardian_state() or {}
        # Return the per-module dict (which is the shape the legacy
        # bus-cache key carried — guardian.status = {module_name: info_dict}).
        return payload.get("modules", {}) or {}

    def get_modules_by_layer(self, layer: str) -> list:
        payload = self._shm.read_guardian_state() or {}
        by_layer = payload.get("modules_by_layer", {}) or {}
        result = by_layer.get(layer, [])
        if isinstance(result, list):
            return list(result)
        # Fallback: derive from modules dict if modules_by_layer is empty
        modules = payload.get("modules", {}) or {}
        return [
            name for name, info in modules.items()
            if isinstance(info, dict) and info.get("layer") == layer
        ]


class AgencyAccessor(_SubAccessorBase):
    """Agency state — SHM-direct via agency_state.bin (publisher:
    agency_worker per SPEC §7.1 row 780 / Session 3). Phase A.3
    migration: bus-cache retired per Preamble G18."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def get_stats(self) -> dict:
        return self._shm.read_agency_state() or {}


class LanguageAccessor(_SubAccessorBase):
    """Language teacher state — SHM-direct via language_state.bin (publisher:
    language_worker per SPEC §7.1 row 791 / Session 4). Phase A.3
    migration: bus-cache retired per Preamble G18."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def get_stats(self) -> dict:
        return self._shm.read_language_state() or {}


class MetaTeacherAccessor(_SubAccessorBase):
    """Meta teacher state — SHM-direct via meta_teacher_state.bin (publisher:
    cognitive_worker per SPEC §7.1 / D-SPEC-70). Phase A.4 migration:
    bus-cache retired per Preamble G18."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def get_stats(self) -> dict:
        return self._shm.read_meta_teacher_state() or {}


class ExperienceAccessor(_SubAccessorBase):
    """Experience-orchestrator stats — SHM-direct via experience_stats.bin
    (publisher: cognitive_worker / ExperienceOrchestrator per SPEC §7.1 /
    D-SPEC-PHASE15). §3L Phase 15 chunk 15.1: replaces the retired frozen
    ExperienceMemory.get_stats recompute-on-read path per Preamble G18."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def get_stats(self) -> dict:
        return self._shm.read_experience_stats() or {}


class SocialAccessor(_SubAccessorBase):
    """Social/persona state — SHM-direct via social_graph_state.bin
    (publisher: social_graph_worker per SPEC §7.1 row 794 / D-SPEC-49).
    Phase A.3 migration: bus-cache retired per Preamble G18."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def get_stats(self) -> dict:
        return self._shm.read_social_graph_state() or {}


class MemoryAccessor(_SubAccessorBase):
    """Memory state — SHM-direct via memory_state.bin (publisher:
    memory_worker per SPEC §7.1 row 783 / Session 2 + Phase A.4 schema
    expansion D-SPEC-70). Phase A.4 migration: bus-cache retired per
    Preamble G18 + memory_state.bin schema expanded with top_memories /
    mempool_preview / knowledge_graph fields, eliminating the
    memory_proxy work-RPC fallback for these previously-bus-cache
    state-lookups. Mirrors plugin._proxies['memory'] (MemoryProxy)
    interface for legacy endpoint code.
    """

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def get_memory_status(self) -> dict:
        return self._shm.read_memory_state() or {}

    def get_topology(self) -> dict:
        state = self._shm.read_memory_state() or {}
        return state.get("topology_clusters_summary", {}) or {}

    def get_top_memories(self, n: int = 10) -> list:
        """SHM-direct via memory_state.bin `top_memories` field (Phase A.4
        gap-1 closure)."""
        state = self._shm.read_memory_state() or {}
        items = state.get("top_memories", []) or []
        if not isinstance(items, list):
            return []
        return list(items[:n])

    def get_neuromod_state(self) -> dict:
        # Phase A.3: route through canonical neuromod_state.bin (Rust
        # writer cognitive_worker per SPEC §7.1 row 760). This was the
        # `memory.neuromod_state` cache key — same data, canonical source.
        return self._shm.read_neuromod() or {}

    def get_ns_state(self) -> dict:
        # Phase A.3: route through canonical titanvm_registers.bin
        # (publisher ns_worker per §7.1 row 765). Same data as old
        # `memory.ns_state` cache key.
        return self._shm.read_titanvm_registers() or {}

    def get_reasoning_state(self) -> dict:
        # Phase A.4: SHM-direct via reasoning_state.bin (publisher:
        # cognitive_worker per D-SPEC-70). Replaces memory.reasoning_state
        # bus-cache key.
        return self._shm.read_reasoning_state() or {}

    def get_persistent_count(self) -> int:
        status = self._shm.read_memory_state() or {}
        if isinstance(status, dict):
            return int(status.get("persistent_count", 0) or 0)
        return 0

    def fetch_mempool(self) -> list:
        """SHM-direct via memory_state.bin `mempool_preview` field (Phase
        A.4 gap-1 closure)."""
        state = self._shm.read_memory_state() or {}
        items = state.get("mempool_preview", []) or []
        return list(items) if isinstance(items, list) else []

    def fetch_social_metrics(self) -> dict:
        # Phase A.3: canonical source is social_graph_state.bin (writer:
        # social_graph_worker per §7.1 row 794). Previous `memory.social_metrics`
        # cache key was a memory→social slice; canonical replacement is
        # the social_graph slot itself.
        return self._shm.read_social_graph_state() or {}

    def get_knowledge_graph(self, limit: int = 200) -> dict:
        """SHM-direct via memory_state.bin `knowledge_graph` field (Phase
        A.4 gap-1 closure). The `limit` arg is preserved for endpoint API
        compatibility but the SHM payload carries a fixed summary shape
        (node_count + edge_count + per-table counts); heavy full-graph
        queries that exceeded slot capacity remain on memory_proxy."""
        state = self._shm.read_memory_state() or {}
        kg = state.get("knowledge_graph", {}) or {}
        return kg if isinstance(kg, dict) else {}

    @property
    def _cognee_ready(self) -> bool:
        status = self.get_memory_status()
        return bool(status.get("cognee_ready", False))


# RLAccessor RETIRED — the recorder_state.bin slot + recorder_worker producer
# are gone (RFP_synthesis_decision_authority P1).


class LLMAccessor(_SubAccessorBase):
    """LLM state — SHM-direct via llm_state.bin (publisher: llm_worker per
    SPEC §7.1 / D-SPEC-70). Phase A.4 migration: bus-cache retired per
    Preamble G18."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def get_stats(self) -> dict:
        return self._shm.read_llm_state() or {}


class MediaAccessor(_SubAccessorBase):
    """Media state — SHM-direct via media_state.bin (publisher: studio_worker
    per SPEC §7.1 / D-SPEC-70). Phase A.4 migration: bus-cache retired per
    Preamble G18."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def get_stats(self) -> dict:
        return self._shm.read_media_state() or {}


class TimechainAccessor(_SubAccessorBase):
    """TimeChain state — SHM-direct via timechain_state.bin (publisher:
    timechain_worker per SPEC §7.1 row 784 / Session 3). Phase A.3
    migration: bus-cache retired per Preamble G18."""

    def __init__(self, shm: ShmReaderBank) -> None:
        self._shm = shm

    def get_stats(self) -> dict:
        return self._shm.read_timechain_state() or {}


class ConfigAccessor:
    """Static config — loaded once at boot from the config dict.

    Per PLAN v2 — config is immutable at runtime. Loaded from the
    `_full_config` dict passed at TitanStateAccessor construction time.
    No bus events; no cache invalidation needed.
    """


    def __init__(self, full_config: dict) -> None:
        self._config = full_config or {}

    def get(self, key: str, default: Any = None) -> Any:
        # Supports dotted-path lookup like "network.vault_program_id"
        parts = key.split(".")
        cur: Any = self._config
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur

    def section(self, name: str) -> dict:
        sect = self._config.get(name, {})
        return sect if isinstance(sect, dict) else {}

    @property
    def full(self) -> dict:
        return self._config


# ── Top-level accessor ────────────────────────────────────────────────


class TitanStateAccessor:
    """Single state-access object. Endpoint code reads via
    `request.app.state.titan_state.X.Y`.

    Composition (per PLAN v2 §2.1):
      shm-backed:  trinity, neuromods, epoch, spirit (45D), identity (post-S4),
                   sphere_clocks (post-S4), chi (post-S4), titanvm (post-S4)
      bus-cached:  network, soul, body, mind, cgn, reasoning, dreaming,
                   guardian, agency, language, meta_teacher, social
      static:      config (immutable at runtime)
      commands:    CommandSender (fire-and-forget bus.publish)
    """

    # No __slots__ so __getattr__ fallback can synthesize accessors for
    # less-common sub-accessors (metabolism, studio, gatekeeper, etc.) on
    # demand without enumerating every one upfront. Saves a class-per-name
    # boilerplate while keeping the typed accessors for the hot path.
    pass

    def __init__(
        self,
        shm: ShmReaderBank,
        commands: CommandSender,
        full_config: dict | None = None,
    ) -> None:
        self.shm = shm
        self.commands = commands

        # Sub-accessors — all SHM-direct per Preamble G18 (D-SPEC-71
        # Phase A + D-SPEC-78 Phase B + D-SPEC-79 Phase C + D-SPEC-82
        # Phase D — bus-cache → CachedState pipeline RETIRED fleet-wide).
        self.network = NetworkAccessor(shm)
        self.trinity = TrinityAccessor(shm)
        self.neuromods = NeuromodAccessor(shm)
        self.epoch = EpochAccessor(shm)
        self.spirit = SpiritAccessor(shm)
        self.body = BodyAccessor(shm)
        self.mind = MindAccessor(shm)
        self.identity = IdentityAccessor(shm)
        self.dreaming = DreamingAccessor(shm)
        self.agency = AgencyAccessor(shm)
        self.language = LanguageAccessor(shm)
        self.social = SocialAccessor(shm)
        self.memory = MemoryAccessor(shm)
        self.timechain = TimechainAccessor(shm)
        self.soul = SoulAccessor(shm)
        self.cgn = CGNAccessor(shm)
        self.reasoning = ReasoningAccessor(shm)
        self.guardian = GuardianAccessor(shm)
        self.meta_teacher = MetaTeacherAccessor(shm)
        self.experience = ExperienceAccessor(shm)
        self.llm = LLMAccessor(shm)
        self.media = MediaAccessor(shm)
        self.config = ConfigAccessor(full_config or {})

        # Legacy callsites use `plugin.event_bus.emit(...)` +
        # `plugin.bus.publish(...)`. Provide compatible shims that route
        # to CommandSender so the codemod doesn't have to rewrite call
        # shapes. Survives Phase B/C/D since the underlying transport is
        # bus.publish either way.
        self.event_bus = _EventBusShim(commands)
        self.bus = _BusShim(commands)

        logger.info(
            "[TitanStateAccessor] initialized for titan_id=%s "
            "(shm registries=%d, commands=%s)",
            shm.titan_id,
            sum(1 for v in shm.availability_report().values() if v),
            "yes" if commands._send_queue is not None else "stub",
        )

    # -- fallback accessor for less-common sub-accessors ---------------

    def __getattr__(self, name: str):
        """Synthesize an empty-default fallback for any unknown sub-accessor
        name. Phase D D-SPEC-82: bus-cache lookup retired; the fallback
        returns an empty-default _CacheGetter (no real cache reads).

        Underscore-prefixed names (e.g. plugin._dream_inbox,
        plugin._current_user_id) raise AttributeError so legacy paths
        like `hasattr(plugin, "_proxies")` still return False.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return _CacheGetter(name)

    # -- introspection ------------------------------------------------

    def availability(self) -> dict[str, Any]:
        """Per-accessor availability — used by /v4/api-status diagnostic."""
        return {
            "shm_registries": self.shm.availability_report(),
            "titan_id": self.shm.titan_id,
        }


_UNKNOWN_ACCESS_LOG_THROTTLE: dict[str, int] = {}


class _CacheGetter:
    """Generic empty-default fallback accessor for sub-accessors not
    explicitly typed.

    Phase A.4 (D-SPEC-71) + Phase D (D-SPEC-82) closure: state transport
    is SHM only per Preamble G18; the legacy bus-cache fallback was first
    neutralized in Phase A, then the CachedState pipeline retired in
    Phase D. Any unenumerated sub-accessor now reaches this class, which
    returns empty defaults. First-access per name logs a WARN so the
    drift surfaces in journalctl and can be migrated to its canonical SHM
    slot in a follow-up.

    Returned by TitanStateAccessor.__getattr__ on demand for any
    sub-accessor that wasn't pre-instantiated (the explicit set is the
    canonical surface per rFP §A.1; this fallback exists only as a
    backward-compat hedge while remaining call-sites are audited).
    """


    def __init__(self, name: str) -> None:
        self._name = name
        self._log_unknown_access(name)

    @staticmethod
    def _log_unknown_access(name: str) -> None:
        count = _UNKNOWN_ACCESS_LOG_THROTTLE.get(name, 0)
        _UNKNOWN_ACCESS_LOG_THROTTLE[name] = count + 1
        if count == 0 or count == 100 or count % 1000 == 0:
            logger.warning(
                "[StateAccessor] unenumerated sub-accessor '%s' accessed "
                "via _CacheGetter fallback (count=%d) — G18 returns empty "
                "default; add explicit SHM-direct sub-accessor for this path",
                name, count + 1)

    def __getattr__(self, attr: str):
        if attr.startswith("_"):
            raise AttributeError(attr)
        return _CacheGetter(f"{self._name}.{attr}")

    def __call__(self, *args, **kwargs):
        return {}

    def __await__(self):
        return _CallableValue({}).__await__()

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"<_CacheGetter name={self._name}>"


class _CallableValue:
    """Wrapped value usable as both a value and as a no-arg callable."""

    __slots__ = ("_v",)

    def __init__(self, v) -> None:
        self._v = v

    def __call__(self, *args, **kwargs):
        return self._v

    def __getattr__(self, name: str):
        return getattr(self._v, name)

    def __getitem__(self, key):
        return self._v[key]

    def __iter__(self):
        return iter(self._v)

    def __len__(self):
        return len(self._v)

    def __bool__(self):
        return bool(self._v)

    def __eq__(self, other):
        return self._v == other

    def __await__(self):
        # Make the wrapped value usable in `await ...` expressions.
        # Legacy endpoint code calls `await plugin.X.Y()` for what was an
        # async method on the real plugin. After codemod those become
        # `await titan_state.X.Y()` which now returns a sync value via
        # _CacheGetter; making this awaitable preserves call-site syntax.
        if False:
            yield  # makes this a generator → __await__ protocol
        return self._v

    def __repr__(self):
        return f"<_CallableValue {self._v!r}>"


_MISSING = object()


def _empty_callable(*args, **kwargs) -> dict:
    return {}


# ── Microkernel v2 D2 amendment shims (2026-04-26) ────────────────────


class _EventBusShim:
    """Compat shim for `plugin.event_bus.emit(type, payload)`.

    Routes via CommandSender → OBSERVATORY_EVENT bus → SSE/WebSocket
    subscribers. Maker.py + webhook.py callsites preserved verbatim.

    Async tolerance: legacy code does `await plugin.event_bus.emit(...)`.
    `emit` returns a string (request_id); making it awaitable via a noop
    coroutine wrapper keeps `await` legal without breaking sync callers.
    """

    __slots__ = ("_commands",)

    def __init__(self, commands: CommandSender) -> None:
        self._commands = commands

    def emit(self, event_type: str, payload: dict | None = None):
        rid = self._commands.emit(event_type, payload)
        return _AwaitableValue(rid)

    @property
    def subscriber_count(self) -> int:
        # Microkernel v2: WebSocket subscribers managed by api_subprocess
        # internals; cache key updated per-tick. Returns 0 when not yet
        # populated (frontend treats as "not connected").
        return 0


class _BusShim:
    """Compat shim for `plugin.bus`.

    Routes:
      - `.publish(msg)` → CommandSender.publish (msg shape preserved)
      - `.stats` → empty dict (canonical bus stats live on the kernel RPC)
      - bool truth (`if plugin.bus:`) → True (always available)

    Anything else falls through __getattr__ as a no-op (returns _empty_callable).
    """

    __slots__ = ("_commands",)

    def __init__(self, commands: CommandSender) -> None:
        self._commands = commands

    def publish(self, msg: dict) -> int:
        """Mimic DivineBus.publish(msg) — accepts a make_msg-shaped dict."""
        if not isinstance(msg, dict):
            return 0
        msg_type = str(msg.get("type", ""))
        dst = str(msg.get("dst", "all"))
        payload = msg.get("payload", {}) or {}
        rid = self._commands.publish(msg_type, dst, payload,
                                     src=str(msg.get("src", "api")))
        return 1 if rid else 0

    @property
    def stats(self) -> dict:
        """Bus broker stats — returns empty dict; canonical bus stats come
        from kernel_rpc proxy at
        `app.state.titan_hcl.kernel.bus_broker_stats()` (see
        `api/dashboard.py:/v4/state`'s bus_broker_stats path)."""
        return {}

    def __bool__(self) -> bool:
        return True

    def __getattr__(self, name: str):
        # Catch-all for less-common bus attributes — returns a no-op
        # callable so legacy code doesn't crash on miss.
        if name.startswith("_"):
            raise AttributeError(name)
        return _empty_callable


class _AwaitableValue:
    """A value that is also `await`-able (resolves to itself).

    Used for shim methods that legacy code calls with `await` even though
    the new sync path returns immediately. Avoids needing to rewrite
    callsites or introduce real coroutines for fire-and-forget operations.
    """

    __slots__ = ("_v",)

    def __init__(self, v) -> None:
        self._v = v

    def __await__(self):
        if False:
            yield  # pragma: no cover — generator marker
        return self._v

    def __repr__(self) -> str:
        return f"<_AwaitableValue {self._v!r}>"
