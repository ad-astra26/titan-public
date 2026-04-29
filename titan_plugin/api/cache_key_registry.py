"""
titan_plugin/api/cache_key_registry.py — single source of truth for the
observatory data contract.

Per **rFP_observatory_data_loading_v1 Phase 1**.

Every cache key the observatory reads is declared here as a CacheKeySpec
binding the four-link chain:

    producer (worker)  →  bus event / snapshot  →  cache key  →  endpoint  →  frontend hook

Audit: `python scripts/arch_map.py cache-keys --audit`

The audit verifies, for each entry:
  • producer_event (if set) is a constant in titan_plugin.bus
  • producer_module function exists and contains a `_send_msg(... event ...)`
    call (for `bus_event` / `hybrid` entries) OR is part of the kernel
    snapshot builder (for `snapshot` entries)
  • EVENT_TO_CACHE_KEY (derived below) covers every bus_event entry
  • consumer_endpoints exist in dashboard.py
  • frontend_hook exists in titan-observatory/hooks/useTitanAPI.ts
  • (live mode) cache age via /v4/cache-staleness within `publish_cadence_s × 3`

Anti-orphan: every `cache.get("X")` call in api/* code MUST resolve to a
registry entry (after stripping legacy / dynamic key allowlist).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Producer kinds ────────────────────────────────────────────────────
#
# bus_event       — producer publishes a *_UPDATED bus event; BusSubscriber
#                   sets the cache key from msg.payload (no snapshot path)
# snapshot        — populated by kernel._build_state_snapshot every ~2s via
#                   STATE_SNAPSHOT_RESPONSE bulk_update (no per-event publisher)
# hybrid          — both: kernel snapshot includes a placeholder/default
#                   shape AND a worker publishes a *_UPDATED event with the
#                   real values when they exist
# missing         — declared but no producer wired yet (catalog §3.1 of rFP);
#                   audit raises ERROR until a producer ships
# deprecated      — historical key, retained for legacy callsites; no
#                   producer required, audit raises WARN only

PRODUCER_KIND = ("bus_event", "snapshot", "hybrid", "missing", "deprecated")


@dataclass(frozen=True)
class CacheKeySpec:
    """One row of the registry — declares the contract for one cache key."""

    key: str
    """Cache key as referenced by `cache.get(key)` and `cache.set(key, ...)`."""

    kind: str
    """One of PRODUCER_KIND."""

    producer_event: Optional[str] = None
    """Bus event name (a constant from titan_plugin.bus). None for
    snapshot-only / missing / deprecated keys."""

    producer_module: str = ""
    """Dotted module + function reference, e.g. `titan_plugin.modules.spirit_worker._publish_chi`.
    For snapshot keys: the snapshot builder + line range, e.g.
    `titan_plugin.core.kernel.MicroKernel._build_state_snapshot`.
    Empty string for `missing` (audit complains)."""

    publish_cadence_s: float = 0.0
    """Expected publish interval in seconds (used by --status live check)."""

    schema: str = ""
    """One-line shape description. Replaced by Pydantic class ref in Phase 2."""

    consumer_endpoints: tuple[str, ...] = ()
    """FastAPI endpoint paths that read this key (or expose data derived from it)."""

    frontend_hook: Optional[str] = None
    """Exported hook name in titan-observatory/hooks/useTitanAPI.ts."""

    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind not in PRODUCER_KIND:
            raise ValueError(
                f"CacheKeySpec({self.key!r}): kind must be one of {PRODUCER_KIND}, "
                f"got {self.kind!r}")
        if self.kind == "bus_event" and not self.producer_event:
            raise ValueError(
                f"CacheKeySpec({self.key!r}): kind=bus_event requires producer_event")
        if self.kind == "missing" and self.producer_module:
            raise ValueError(
                f"CacheKeySpec({self.key!r}): kind=missing must NOT set "
                f"producer_module (it is by definition unwired)")


# ── REGISTRY ─────────────────────────────────────────────────────────
#
# Order: domain-grouped (chi/π/dreaming → memory → spirit composites →
# trinity tensors → metabolic/network/identity → guardian/agency/social
# → reasoning/meta/msl → legacy)
#
# When adding a new key:
#   1. Pick kind based on producer pattern
#   2. Set producer_event (uppercase, ends in _UPDATED) for bus_event/hybrid
#   3. Set producer_module to dotted path including function name
#   4. Set publish_cadence_s based on producer's emit cadence
#   5. Run `python scripts/arch_map.py cache-keys --audit` — it should pass
#   6. If audit fails, fix the producer wiring (or mark `missing` if intentional)

REGISTRY: list[CacheKeySpec] = [

    # ── Chi life-force ────────────────────────────────────────────────
    CacheKeySpec(
        key="chi.state",
        kind="hybrid",
        producer_event="CHI_UPDATED",
        producer_module="titan_plugin.modules.spirit_worker",
        publish_cadence_s=1.0,
        schema=(
            "{total:float, spirit:ChiLayer, mind:ChiLayer, body:ChiLayer, "
            "circulation:float, state:str, developmental_phase:str, "
            "weights:dict, contemplation:dict} where ChiLayer = "
            "{raw, effective, weight, thinking, feeling, willing}"
        ),
        consumer_endpoints=("/v4/chi",),
        frontend_hook="useChi",
        notes="Real values published by spirit_worker after life_force_engine.evaluate; "
              "kernel snapshot ships placeholder shape so frontend never crashes pre-CHI_UPDATED.",
    ),

    # ── π Heartbeat ───────────────────────────────────────────────────
    CacheKeySpec(
        key="pi_heartbeat.state",
        kind="bus_event",
        producer_event="PI_HEARTBEAT_UPDATED",
        producer_module="titan_plugin.modules.spirit_loop._publish_coord_snapshot",
        publish_cadence_s=2.0,
        schema="{total_epochs_observed:int, developmental_age:int, heartbeat_ratio:float, ...}",
        consumer_endpoints=("/v4/pi-heartbeat",),
        frontend_hook="usePiHeartbeat",
    ),

    # ── Dreaming ──────────────────────────────────────────────────────
    CacheKeySpec(
        key="dreaming.state",
        kind="bus_event",
        producer_event="DREAMING_STATE_UPDATED",
        producer_module="titan_plugin.modules.spirit_loop._publish_coord_snapshot",
        publish_cadence_s=2.0,
        schema="{is_dreaming:bool, cycle_count:int, fatigue:float, recovery_progress:float, "
               "developmental_age:int, distilled_count:int}",
        consumer_endpoints=("/v4/dreaming",),
        frontend_hook="useDreaming",
    ),

    # ── Meta-reasoning ────────────────────────────────────────────────
    CacheKeySpec(
        key="meta_reasoning.state",
        kind="bus_event",
        producer_event="META_REASONING_STATS_UPDATED",
        producer_module="titan_plugin.modules.spirit_loop._publish_coord_snapshot",
        publish_cadence_s=2.0,
        schema="{total_chains:int, total_eurekas:int, primitives:list, rewards:dict}",
        consumer_endpoints=("/v4/meta-reasoning/audit",),
        frontend_hook="useMetaReasoning",
    ),

    # ── Reasoning ─────────────────────────────────────────────────────
    CacheKeySpec(
        key="reasoning.state",
        kind="bus_event",
        producer_event="REASONING_STATS_UPDATED",
        producer_module="titan_plugin.modules.spirit_loop._publish_coord_snapshot",
        publish_cadence_s=2.0,
        schema="{total_chains:int, total_eurekas:int, last_chain_at:int, ...}",
        consumer_endpoints=("/v4/reasoning",),
        frontend_hook="useReasoning",
    ),
    CacheKeySpec(
        key="reasoning.stats",
        kind="deprecated",
        notes="Pre-microkernel-v2 alias; `reasoning.state` is canonical. Retained "
              "for one release while old callers migrate.",
    ),

    # ── Expression composites ─────────────────────────────────────────
    CacheKeySpec(
        key="expression.composites",
        kind="bus_event",
        producer_event="EXPRESSION_COMPOSITES_UPDATED",
        producer_module="titan_plugin.modules.spirit_loop._publish_coord_snapshot",
        publish_cadence_s=2.0,
        schema="{SPEAK:dict, ART:dict, MUSIC:dict, SOCIAL:dict, KIN:dict, LONGING:dict}",
        consumer_endpoints=("/v4/state-snapshot",),
        frontend_hook="useExpressionComposites",
    ),
    CacheKeySpec(
        key="spirit.expression_composites",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{SPEAK, ART, MUSIC, SOCIAL, KIN, LONGING composites — each {fired:bool, score:float, ...}}",
        consumer_endpoints=("/v4/inner-trinity", "/v4/state-snapshot"),
        notes="Snapshot mirror of expression.composites for frontend hooks that "
              "read the inner-trinity composite payload.",
    ),

    # ── Neuromodulators ───────────────────────────────────────────────
    CacheKeySpec(
        key="neuromods.full",
        kind="bus_event",
        producer_event="NEUROMOD_STATS_UPDATED",
        producer_module="titan_plugin.modules.spirit_loop._publish_coord_snapshot",
        publish_cadence_s=2.0,
        schema="{modulators:{DA:{level,baseline,...}, 5HT:..., NE:..., GABA:...}, ...}",
        consumer_endpoints=("/v4/inner-trinity",),
        frontend_hook="useNeuromodulators",
    ),

    # ── MSL — Multisensory Synthesis Layer ─────────────────────────────
    CacheKeySpec(
        key="msl.state",
        kind="bus_event",
        producer_event="MSL_STATE_UPDATED",
        producer_module="titan_plugin.modules.spirit_loop._publish_coord_subdomains",
        publish_cadence_s=2.0,
        schema="{i_confidence:float, i_depth:float, i_depth_components:dict, "
               "convergence_count:int, concept_confidences:dict, attention_weights:dict, "
               "attention_entropy:float, homeostatic:dict}",
        consumer_endpoints=("/v4/inner-trinity",),
        notes="Phase 4 wired 2026-04-26. Coord snapshot already builds msl state at "
              "build_coordinator_snapshot:1651; fan-out added to _publish_coord_subdomains.",
    ),

    # ── Memory subsystem ──────────────────────────────────────────────
    CacheKeySpec(
        key="memory.status",
        kind="bus_event",
        producer_event="MEMORY_STATUS_UPDATED",
        producer_module="titan_plugin.modules.memory_worker",
        publish_cadence_s=5.0,
        schema="{persistent_count:int, mempool_size:int, cognee_ready:bool, memory_backend_ready:bool}",
        consumer_endpoints=("/status/memory",),
        frontend_hook="useMemory",
    ),
    CacheKeySpec(
        key="memory.mempool",
        kind="bus_event",
        producer_event="MEMORY_MEMPOOL_UPDATED",
        producer_module="titan_plugin.modules.memory_worker",
        publish_cadence_s=5.0,
        schema="{count:int, items:list}",
        consumer_endpoints=("/status/memory",),
    ),
    CacheKeySpec(
        key="memory.top",
        kind="bus_event",
        producer_event="MEMORY_TOP_UPDATED",
        producer_module="titan_plugin.modules.memory_worker",
        publish_cadence_s=5.0,
        schema="{items:list, count:int}",
        consumer_endpoints=("/status/memory",),
    ),
    CacheKeySpec(
        key="memory.topology",
        kind="bus_event",
        producer_event="MEMORY_TOPOLOGY_UPDATED",
        producer_module="titan_plugin.modules.memory_worker",
        publish_cadence_s=30.0,
        schema="{by_topic_cluster:dict, by_entity_type:dict, total_classified:int}",
        consumer_endpoints=("/status/memory/topology",),
        frontend_hook="useMemoryTopology",
        notes="Phase 4 wired 2026-04-26. memory_worker._publish_memory_topology bins "
              "top_memories by keyword cluster + queries Kuzu for entity-type counts.",
    ),
    CacheKeySpec(
        key="memory.knowledge_graph",
        kind="bus_event",
        producer_event="MEMORY_KNOWLEDGE_GRAPH_UPDATED",
        producer_module="titan_plugin.modules.memory_worker",
        publish_cadence_s=30.0,
        schema="{available:bool, stats:dict, nodes:list[{id,label,type}], edges:list[{source,target,type}]}",
        consumer_endpoints=("/status/memory/knowledge-graph",),
        notes="Phase 4 wired 2026-04-26. memory_worker queries Kuzu directly for up to "
              "300 nodes (50/type) + 100 edges per cycle.",
    ),
    CacheKeySpec(
        key="memory.persistent_count",
        kind="deprecated",
        notes="Legacy direct key; use `memory.status.persistent_count` instead.",
    ),
    CacheKeySpec(
        key="memory.neuromod_state",
        kind="deprecated",
        notes="Legacy MemoryProxy mirror; use `neuromods.full` from spirit_loop.",
    ),
    CacheKeySpec(
        key="memory.ns_state",
        kind="deprecated",
        notes="Legacy MemoryProxy mirror of nervous-system stats.",
    ),
    CacheKeySpec(
        key="memory.reasoning_state",
        kind="deprecated",
        notes="Legacy MemoryProxy mirror; use `reasoning.state` from spirit_loop.",
    ),
    CacheKeySpec(
        key="memory.social_metrics",
        kind="deprecated",
        notes="Legacy MemoryProxy mirror; use `social.stats`.",
    ),

    # ── Spirit composites (snapshot-published) ────────────────────────
    CacheKeySpec(
        key="spirit.coordinator",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{consciousness, sphere_clocks, unified_spirit, resonance, observables, "
               "neuromodulators, cgn, metabolic, outer_trinity, trinity, "
               "expression_composites, neural_nervous_system, meta_reasoning, msl, "
               "pi_heartbeat, dreaming, chi}",
        consumer_endpoints=("/v4/inner-trinity", "/status",),
        notes="Aggregate coordinator object — the legacy 'one big payload' shape "
              "many /v4/* endpoints expect. Real values populate as per-event "
              "publishers fire (chi.state, pi_heartbeat.state, etc.).",
    ),
    CacheKeySpec(
        key="spirit.inner_trinity",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="alias of spirit.coordinator (same shape)",
        consumer_endpoints=("/v4/inner-trinity",),
        frontend_hook="useV4InnerTrinity",
    ),
    CacheKeySpec(
        key="spirit.sphere_clocks",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{clocks:{<name>:{phase:float, pulse_count:int, frequency_hz:float}}}",
        consumer_endpoints=("/v4/sphere-clocks",),
        frontend_hook="useSphereClocksV4",
    ),
    CacheKeySpec(
        key="spirit.resonance",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{recent_events:list, resonance_score:float, ...}",
        consumer_endpoints=("/v4/resonance",),
    ),
    CacheKeySpec(
        key="spirit.unified_spirit",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{tensor:list[130], ...}",
        consumer_endpoints=("/v4/unified-spirit",),
    ),
    CacheKeySpec(
        key="spirit.neural_nervous_system",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{is_firing:bool, transitions:int, steps:int, programs:dict}",
        consumer_endpoints=("/v4/nervous-system",),
        frontend_hook="useNervousSystem",
    ),
    CacheKeySpec(
        key="spirit.v4_state",
        kind="deprecated",
        notes="Endpoint /v4/state bypasses cache and goes directly through "
              "SpiritProxy.get_v4_state() (sync request/reply to spirit_worker). "
              "Cache accessor is dead — retained for legacy/future use only.",
    ),

    # ── Trinity tensors (kernel snapshot from state_register) ─────────
    CacheKeySpec(
        key="body.tensor",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{tensor:list[15], outer:list[5], focus:list, filter_down:list, details:dict, center_dist:float}",
        consumer_endpoints=("/v4/state-snapshot",),
    ),
    CacheKeySpec(
        key="mind.tensor",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{tensor:list[15], outer:list[5], focus:list, filter_down:list, center_dist:float}",
        consumer_endpoints=("/v4/state-snapshot",),
    ),
    CacheKeySpec(
        key="spirit.tensor",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="list[45] — SAT-15 + CHIT-15 + ANANDA-15",
        consumer_endpoints=("/v4/state-snapshot",),
    ),

    # ── Metabolic / network / identity ────────────────────────────────
    CacheKeySpec(
        key="metabolism.state",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{energy_state:str, sol_balance:float, balance_pct:float, mood_valence:float, ...}",
        consumer_endpoints=("/health", "/status",),
        notes="Populated from state_register snapshot snap['metabolic']. Real values "
              "come from spirit-side metabolic engine; need bus event "
              "METABOLISM_STATE_UPDATED on energy_state transition (rFP §3.2).",
    ),
    CacheKeySpec(
        key="network.balance",
        kind="bus_event",
        producer_event="SOLANA_BALANCE_UPDATED",
        producer_module="titan_plugin.core.kernel.MicroKernel._start_balance_publisher",
        publish_cadence_s=60.0,
        schema="{balance:float, last_updated:float}",
        consumer_endpoints=("/health", "/status",),
        notes="H4 BalancePublisher. Currently flag-OFF after first-deploy crash; "
              "see rFP §3.5 for delayed-first-publish fix (gate on Guardian-api-READY).",
    ),
    CacheKeySpec(
        key="network.info",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{pubkey:str, rpc_urls:list[str], premium_rpc:str|None}",
        consumer_endpoints=("/health",),
    ),
    CacheKeySpec(
        key="soul.state",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{maker_pubkey:str, nft_address:str, current_gen:int, active_directives:list}",
        consumer_endpoints=("/health", "/v4/sovereignty/status",),
    ),

    # ── Guardian / agency ─────────────────────────────────────────────
    CacheKeySpec(
        key="guardian.status",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{<module_name>:{layer:str, running:bool, restarts:int, last_heartbeat:float, ...}}",
        consumer_endpoints=("/status/guardian", "/v3/guardian",),
        frontend_hook="useGuardian",
    ),
    CacheKeySpec(
        key="gatekeeper.status",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{alive:bool, sovereignty_score:float, verified_count:int, rejected_count:int}",
        consumer_endpoints=("/health",),
        notes="Added 2026-04-26 to fix /health subsystems.gatekeeper=ABSENT.",
    ),
    CacheKeySpec(
        key="agency.stats",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{total_actions:int, recent_actions:list, ...}",
        consumer_endpoints=("/v3/agency",),
        frontend_hook="useAgency",
    ),
    CacheKeySpec(
        key="bus.stats",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{published:int, dropped:int, routed:int, timeouts:int}",
        consumer_endpoints=("/v4/bus-health",),
        notes="DivineBus internal counters. Backs `plugin.bus.stats` via _BusShim.",
    ),

    # ── Language ───────────────────────────────────────────────────────
    CacheKeySpec(
        key="language.stats",
        kind="bus_event",
        producer_event="LANGUAGE_STATS_UPDATED",
        producer_module="titan_plugin.modules.spirit_loop._publish_coord_subdomains",
        publish_cadence_s=2.0,
        schema="{vocab:int, prod:int, level:str, conf:float, last_teach_at:float, ...}",
        consumer_endpoints=("/v4/vocabulary",),
        notes="Phase 4 wired 2026-04-26. Coord snapshot already includes stats[\"language\"] "
              "from language_stats refs; fan-out added to _publish_coord_subdomains.",
    ),

    # ── Topology (Batch E — extends TopologyPanel with 30D space topology) ─
    CacheKeySpec(
        key="topology.state",
        kind="bus_event",
        producer_event="TOPOLOGY_STATE_UPDATED",
        producer_module="titan_plugin.modules.spirit_loop._publish_coord_subdomains",
        publish_cadence_s=2.0,
        schema=(
            "{volume:float, curvature:float, cluster_count:int, "
            "cluster_threshold:float, observables_30d:list[30], "
            "observables_dict:{layer:{coherence,magnitude,velocity,direction,polarity}}}"
        ),
        consumer_endpoints=("/v4/inner-trinity",),
        notes="Batch E (2026-04-26) — TopologyPanel data. SpiritAccessor.get_coordinator() "
              "overlay merges this into coord['topology'].",
    ),

    # ── Aspirational keys (no consumer reads them — dataflow goes via files / hardcoded) ─
    CacheKeySpec(
        key="social.stats",
        kind="deprecated",
        notes="No consumer reads this cache key — /status/social returns hardcoded "
              "placeholder; persona telemetry served directly from "
              "data/persona_telemetry.jsonl via /v4/persona-telemetry. "
              "Retained as deprecated; remove if no consumer ever wires up.",
    ),
    CacheKeySpec(
        key="meta_teacher.stats",
        kind="deprecated",
        notes="No consumer reads this cache key — /v4/meta-cgn/advisor-conflicts reads "
              "shadow_mode_log.jsonl directly. Retained as deprecated.",
    ),
    CacheKeySpec(
        key="cgn.stats",
        kind="deprecated",
        notes="No consumer reads this cache key — /v4/cgn-haov-stats reads "
              "data/cgn/haov_stats.json directly. cgn.state (snapshot) carries the "
              "live runtime view from state_register. Retained as deprecated.",
    ),

    # ── Timechain ─────────────────────────────────────────────────────
    CacheKeySpec(
        key="timechain.stats",
        kind="deprecated",
        notes="Endpoint /v4/timechain/status reads dashboard.py-local _tc_status_cache "
              "(refreshed every 8s by tc-status-warmer thread, in api_subprocess). "
              "Cache accessor titan_state.timechain.get_stats() is unused. Retained "
              "for legacy/future use only.",
    ),

    # ── Plugin private state (snapshot-cached) ────────────────────────
    CacheKeySpec(
        key="plugin._limbo_mode",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="bool",
        notes="Plugin private attr mirrored into cache for legacy proxy reads.",
    ),
    CacheKeySpec(
        key="plugin._dream_inbox",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="list[dict] — pending dream items",
        frontend_hook="useDreamInbox",
    ),
    CacheKeySpec(
        key="plugin._current_user_id",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="str",
    ),
    CacheKeySpec(
        key="plugin._is_meditating",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="bool",
    ),
    CacheKeySpec(
        key="plugin._start_time",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="float — Unix timestamp of plugin boot",
    ),
    CacheKeySpec(
        key="plugin._last_execution_mode",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="str",
    ),
    CacheKeySpec(
        key="plugin._last_commit_signature",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="str",
    ),
    CacheKeySpec(
        key="plugin._last_research_sources",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="list[dict]",
    ),
    CacheKeySpec(
        key="plugin._pending_self_composed",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="str|None",
    ),
    CacheKeySpec(
        key="plugin._pending_self_composed_confidence",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="float|None",
    ),

    # ── Static config ────────────────────────────────────────────────
    CacheKeySpec(
        key="config.full",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=0.0,  # one-shot at boot, refreshed on snapshot
        schema="dict — full titan_params + config.toml merge",
        consumer_endpoints=(),
    ),
    CacheKeySpec(
        key="v3.status",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="legacy v3 status dict",
        consumer_endpoints=("/v3/trinity",),
    ),

    # ── Mood/metabolism getter shims ──────────────────────────────────
    CacheKeySpec(
        key="mood_engine.get_mood_label",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="str",
    ),
    CacheKeySpec(
        key="mood_engine.get_mood_valence",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="float",
    ),
    CacheKeySpec(
        key="metabolism.get_current_state",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="dict — full metabolic state",
    ),
    CacheKeySpec(
        key="metabolism.get_tier_info",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{tier:str, sol_balance:float, balance_pct:float}",
    ),

    # ── State-register full payload (rich snapshot) ───────────────────
    CacheKeySpec(
        key="state_register.full",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="full state_register snapshot dict (body_tensor, mind_tensor, ...)",
        consumer_endpoints=("/v4/state-snapshot",),
    ),
    CacheKeySpec(
        key="state_register.age_seconds",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="float",
    ),
    CacheKeySpec(
        key="spirit.outer",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="list[5] — outer spirit projection",
    ),
    CacheKeySpec(
        key="spirit.consciousness",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="dict — consciousness slice from state_register",
    ),
    CacheKeySpec(
        key="spirit.observables_30d",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="list[30] — 30D observables vector",
    ),
    CacheKeySpec(
        key="spirit.observables_dict",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="dict — labeled observables",
    ),
    CacheKeySpec(
        key="spirit.nervous_system",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="{is_firing:bool, transitions:int, steps:int, programs:dict}",
    ),
    CacheKeySpec(
        key="neuromods.state",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="dict — neuromodulator state",
    ),
    CacheKeySpec(
        key="cgn.state",
        kind="snapshot",
        producer_module="titan_plugin.core.kernel.MicroKernel._build_state_snapshot",
        publish_cadence_s=2.0,
        schema="dict — CGN state from state_register",
    ),

    # ── Deprecated/legacy keys (no producer required) ─────────────────
    CacheKeySpec(
        key="rl.stats",
        kind="deprecated",
        notes="Legacy RL stats; modern reads via reasoning.state.",
    ),
    CacheKeySpec(
        key="llm.stats",
        kind="deprecated",
        notes="Legacy LLM stats; replaced by per-provider telemetry.",
    ),
    CacheKeySpec(
        key="media.stats",
        kind="deprecated",
        notes="Legacy media stats; modern reads via /status/art /status/audio.",
    ),
    CacheKeySpec(
        key="recorder.buffer",
        kind="deprecated",
        notes="Legacy recorder buffer; reads now via len(plugin.recorder.buffer).",
    ),
]


# ── Derived constants ────────────────────────────────────────────────

EVENT_TO_CACHE_KEY: dict[str, str] = {
    spec.producer_event: spec.key
    for spec in REGISTRY
    if spec.producer_event is not None and spec.kind in ("bus_event", "hybrid")
}
"""Derived from REGISTRY — every entry whose `kind` is bus_event or hybrid
contributes a (event → cache_key) mapping. BusSubscriber imports this."""


REGISTERED_KEYS: set[str] = {spec.key for spec in REGISTRY}
"""All cache keys with a registry entry (any kind)."""


# ── Allowlist for keys that intentionally bypass the registry ─────────
#
# `cache.get(...)` callsites with these keys will not raise audit ERROR
# even though they have no registry entry. Three reasons a key may be
# allowlisted:
#
#   1. Dynamic — composed at runtime (e.g. `network.account.{pda}`)
#   2. Local-variable shadowing — `cache` is also a common local name in
#      cache.py / coord_cache.py for module-local dicts that aren't the
#      observatory CachedState
#   3. Documentation placeholder — string in a docstring, not a real call

ALLOWLIST_PREFIXES: tuple[str, ...] = (
    "network.account.",        # dynamic (vault PDA accounts)
)

ALLOWLIST_KEYS: frozenset[str] = frozenset({
    # Documentation placeholder strings (in module docstrings)
    "<sub>.<attr>",
    "<sub>.<attr_path>",
    "<sub>.<method>",
    # Local _coordinator_cache dict in dashboard.py — same `.get` syntax
    # but reads a module-local dict, not the observatory CachedState.
    "data",
    "ts",
    # Local cache dicts in helpers
    "text",
    "tensor",
    # Legacy state-accessor fallbacks
    "center_dist",
    "interface_advisor",
    "self_prediction_accuracy",
})


def by_key(key: str) -> Optional[CacheKeySpec]:
    """Lookup spec by cache key. Returns None if not registered."""
    for s in REGISTRY:
        if s.key == key:
            return s
    return None


def by_event(event: str) -> Optional[CacheKeySpec]:
    """Lookup spec by bus event name. Returns None if not registered."""
    for s in REGISTRY:
        if s.producer_event == event:
            return s
    return None


def is_allowlisted(key: str) -> bool:
    """True if key is intentionally outside the registry (dynamic / local)."""
    if key in ALLOWLIST_KEYS:
        return True
    for prefix in ALLOWLIST_PREFIXES:
        if key.startswith(prefix):
            return True
    return False


def specs_by_kind(kind: str) -> list[CacheKeySpec]:
    """All entries of a given producer kind."""
    return [s for s in REGISTRY if s.kind == kind]
