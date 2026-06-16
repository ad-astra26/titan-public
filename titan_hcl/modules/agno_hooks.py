"""
Agno pre/post hooks for the Titan cognitive pipeline.

Pre-hooks fire BEFORE the LLM — memory recall, directive injection, gatekeeper routing.
Post-hooks fire AFTER the LLM — memory logging, RL transition recording.

Step 5 additions:
  - Pre-hook: Trinity state → OutputColoring → injected as behavioral hints
  - Post-hook: InputExtractor → INTERFACE_INPUT bus message (async, nonblocking)

These are direct ports of TitanHCL.pre_prompt_hook / post_resolution_hook,
restructured as standalone callables for Agno's hook interface.
"""
import logging
import random
import re
import asyncio
import time
import time as _time_for_cache  # legacy alias retained for the pre-hook cache TTL helpers
from titan_hcl import bus

logger = logging.getLogger(__name__)


def _ph_rss_mb() -> int:
    """Resident-set size of THIS process (agno_worker) in MB. Diagnostic for
    the chat-time RSS growth (rFP §9.1) — emitted per _ph_stage so the
    pre/post hook stage logs show exactly where RSS balloons. Cheap
    (/proc/self/statm read). Returns 0 on any failure (non-Linux / no /proc)."""
    try:
        with open("/proc/self/statm", "r") as _f:
            _rss_pages = int(_f.read().split()[1])
        return _rss_pages * 4096 // (1024 * 1024)  # pages→bytes→MB (4KB pages)
    except Exception:  # noqa: BLE001
        return 0


def _agno_tel():
    """Lazy agno-worker Telemetry (RFP_worker_telemetry §7.B). Never raises →
    returns None on any import/init failure so instrumentation can't break chat."""
    try:
        from titan_hcl.logic.worker_telemetry import get_telemetry
        return get_telemetry("agno")
    except Exception:  # noqa: BLE001
        return None


def _ph_feature(stage: str) -> str:
    """Coarse feature bucket for a PreHook stage name → the --by-feature grouping
    ("which feature strains the chat path most"). The op name keeps the detail."""
    s = stage.lower()
    if "recall" in s or "vcb" in s or "memory" in s:
        return "recall"
    if "research" in s:
        return "research"
    if "enrich" in s or "v5" in s:
        return "enrichment"
    if "gatekeeper" in s or "directive" in s:
        return "guard"
    return "prehook"


# ─────────────────────────────────────────────────────────────────────
# Phase 2 Chunk γ (D-SPEC-78, 2026-05-18) — PreHook TTL cache.
# ─────────────────────────────────────────────────────────────────────
#
# The PreHook makes multiple sync HTTP calls to `127.0.0.1:7777/v4/*`
# endpoints whose data changes slowly (CGN grounding rate, social-action
# learned tone). Each call is 1-5s on T1 swap-pressured hosts; on warm
# hosts ~100-500ms. Cumulative cost per chat: 1.2-6s.
#
# This module-level cache shares results across PreHook invocations on
# the same agno_worker process. 30-second TTL is the load-bearing knob:
# - CGN grounding rate updates on the order of minutes-to-hours.
# - CGN social-action tone updates per-user but the same user typing 5
#   messages in 30s gets a consistent tone (good UX).
# - On agno_worker restart the cache empties (no persistence concern).
#
# Tradeoff: a chat 35s after the previous one re-hits the slow endpoint.
# That's the price of zero infrastructure (no SHM publisher to maintain).
# Migration to true SHM slots is tracked under rFP Phase 2 Chunk α-2
# (when the language_grounding + cgn_social_action endpoints earn their
# own SHM publishers per Preamble G18).

_PRE_HOOK_CACHE_TTL_S = 30.0
_pre_hook_cache: dict = {}  # key → (value, expiry_ts)


def _pre_hook_cache_get(key: str):
    """Return cached value if non-expired, else None."""
    entry = _pre_hook_cache.get(key)
    if entry is None:
        return None
    value, expiry = entry
    if _time_for_cache.time() >= expiry:
        _pre_hook_cache.pop(key, None)
        return None
    return value


def _pre_hook_cache_set(key: str, value) -> None:
    """Cache value with the default 30s TTL."""
    _pre_hook_cache[key] = (value, _time_for_cache.time() + _PRE_HOOK_CACHE_TTL_S)


def _pre_hook_cache_set_with_ttl(key: str, value, *, ttl_s: float) -> None:
    """Cache value with custom TTL (for values that change at different rates
    than the default 30s — e.g. directives change rarely, 5 min TTL)."""
    _pre_hook_cache[key] = (value, _time_for_cache.time() + ttl_s)

# ── Interface Module singletons (Step 5) ────────────────────────────
# Created once at import time, reused across all hook invocations.
# Thread-safe for single-writer (one conversation flow at a time).
_input_extractor = None
_output_coloring = None

def _get_input_extractor():
    global _input_extractor
    if _input_extractor is None:
        try:
            from titan_hcl.logic.interface_input import InputExtractor
            _input_extractor = InputExtractor()
        except Exception as e:
            logger.warning("[Hooks] InputExtractor init failed: %s", e)
    return _input_extractor

def _get_output_coloring():
    global _output_coloring
    if _output_coloring is None:
        try:
            from titan_hcl.logic.interface_output import OutputColoring
            _output_coloring = OutputColoring()
        except Exception as e:
            logger.warning("[Hooks] OutputColoring init failed: %s", e)
    return _output_coloring


# ── Bridge B: Memory Recall Perturbation ────────────────────────────
# When recalled memories have felt_state_snapshots (from Bridge A dream
# injection), their neurochemical signature creates a micro-perturbation.
# Titan literally re-feels past experiences on recall.

_RECALL_MAX_PER_MOD = 0.02
_RECALL_MAX_TOTAL = 0.05
_RECALL_BRIDGE_TAGS = ("[DREAM_WISDOM]", "[EUREKA]", "[CGN_MILESTONE]",
                       "[COMPOSITION]", "[SOCIAL_PERCEPTION]")
_RECALL_MOD_SCALE = {
    "DA": 0.5, "5HT": 0.5, "NE": 0.3,
    "ACh": 0.4, "Endorphin": 0.4,
}


def _can_afford_research(plugin) -> bool:
    """EEL Pillar 0 §7.0b.2 part 4 — the metabolic research gate.

    Reuses the EXISTING per-tier `research` feature (core/metabolism.py
    TIER_FEATURES: True for THRIVING/HEALTHY/CONSERVING, False for
    SURVIVAL/EMERGENCY/HIBERNATION), read via the metabolism SHM reader
    (`MetabolismShmReader.can_use_feature`, sub-ms, no bus/RPC — G18-safe).
    Fails OPEN (research allowed) when metabolism is unwired/cold so a read
    hiccup never makes Titan refuse to look something up."""
    metab = getattr(plugin, "metabolism", None)
    if metab is None:
        return True
    try:
        cuf = getattr(metab, "can_use_feature", None)
        if callable(cuf):
            return bool(cuf("research"))
        # Fallback: read the SHM-cached tier + dispatch against TIER_FEATURES.
        from titan_hcl.core.metabolism import TIER_FEATURES
        tier = metab.get_metabolic_tier()
        return bool(TIER_FEATURES.get(tier, {}).get("research", True))
    except Exception:
        return True


def _synthesis_decision_authority_enabled(plugin) -> bool:
    """P2 (RFP_synthesis_decision_authority) — the D4 transient feature flag.

    When ON, the grounded router routes on the EngineRecall top cosine (RELEVANCE
    over the spine, INV-SDA-2) with gibberish-calibrated self-floors; when OFF,
    the legacy VCB/memory-composite recall + static floors (byte-identical to P1).
    Config: `[synthesis.decision_authority] enabled`.

    DEFAULT NOW TRUE (2026-06-14): D4 prescribed default-on after one clean T3
    soak; T3 + T2 have run p2=true cleanly for days. The legacy (p2-off) path
    feeds an UNBOUNDED mempool `effective_weight` into the `recall_top_cosine`
    feature (documented [0,1]) → it clips to 1.0 → the OML policy reads "knows
    everything" → collapses to `direct` (this is exactly what stranded T1-mainnet
    when its config.toml silently lacked the section). p2 is the proven-correct
    path; make it the default everywhere so config drift can't re-strand a box.
    An explicit `enabled = false` still honors the legacy path; a config-read
    failure conservatively falls back to legacy."""
    try:
        cfg = (getattr(plugin, "_full_config", {}) or {}).get("synthesis", {}) or {}
        return bool((cfg.get("decision_authority", {}) or {}).get("enabled", True))
    except Exception:
        return False


def _self_learning_enabled(plugin) -> bool:
    """RFP_synthesis_self_learning_meta_reasoning v1.1 — the LEARNED outer-decision
    flag. ON → the `OuterMetaPolicy` (trained from oracle outcomes, read from SHM)
    picks the action, overriding the static `grounded_route` (EXPLOIT only on a live
    turn — INV-OML-9). OFF (default) → grounded_route stands, byte-identical. Config:
    `[synthesis.self_learning] enabled`."""
    try:
        cfg = (getattr(plugin, "_full_config", {}) or {}).get("synthesis", {}) or {}
        return bool((cfg.get("self_learning", {}) or {}).get("enabled", False))
    except Exception:
        return False


def _oml_decide_debug_capture(policy, vec, action, prompt_text, cm_score, cm_action):
    """§24.12 LIVE-OVERRIDE-GAP diagnostic — capture the LIVE 30-D OuterFeatures
    vector + the per-action `policy.forward(vec)` scores on a real user turn, so
    it can be diffed against the SYNTHETIC SHM probe
    (`scripts/titan-docs/oml_recall_probe.py`, which hardcodes
    composite/skill/engram=0 + has_code=False). The feature whose LIVE value
    differs from the probe AND flips the argmax is the override cause (§5.B:
    "the policy is only as good as the live feature vector — INSTRUMENT FIRST").

    Gated by a FILE flag so capture can be armed/disarmed on a LIVE box with NO
    restart: `touch <flag>` to arm, `rm` to disarm. Both the flag and the capture
    live in the PERSISTENT in-repo `<TITAN_DATA_DIR>/logs/` (default `data/logs/`,
    alongside `sage_research.log`) — NOT `/tmp` (which is wiped on reboot + prone
    to cross-owner permission surprises). Override paths via the two env vars.
    Read-only telemetry, fully wrapped — must NEVER break a chat turn (INV-OML-7).
    A single `os.path.exists` stat per user turn when disarmed (turns are
    user-paced, not a hot loop) → negligible."""
    try:
        import os as _os
        _logdir = _os.path.join(_os.environ.get("TITAN_DATA_DIR", "data"), "logs")
        flag = _os.environ.get("TITAN_OML_DECIDE_DEBUG_FLAG",
                               _os.path.join(_logdir, "oml_decide_debug.on"))
        if not _os.path.exists(flag):
            return
        import json as _json
        import numpy as _np
        from titan_hcl.synthesis.outer_meta_policy import (
            OUTER_ACTIONS, OUTER_FEATURE_NAMES, action_index_to_mode,
            action_index_to_name)
        scores = policy.forward(_np.asarray(vec, dtype=_np.float32))
        vlist = [float(x) for x in _np.asarray(vec).ravel().tolist()]
        rec = {
            "ts": round(time.time(), 3),
            "prompt": (prompt_text or "")[:160],
            "action": int(action),
            "action_name": action_index_to_name(int(action)),
            "mode": action_index_to_mode(int(action)),
            "scores": {OUTER_ACTIONS[i]: round(float(scores[i]), 4)
                       for i in range(len(OUTER_ACTIONS))},
            "features": {OUTER_FEATURE_NAMES[i]: round(vlist[i], 4)
                         for i in range(min(len(OUTER_FEATURE_NAMES), len(vlist)))},
            "composite_raw": {"score": round(float(cm_score), 4),
                              "action_norm": round(float(cm_action), 4)},
            "total_updates": int(getattr(policy, "total_updates", 0)),
        }
        path = _os.environ.get("TITAN_OML_DECIDE_DEBUG_PATH",
                               _os.path.join(_logdir, "oml_decide_debug.jsonl"))
        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as _f:
            _f.write(_json.dumps(rec) + "\n")
    except Exception:
        pass


# Break A (RFP_synthesis_reuse_and_routing_revival) — the delegate-gate relevance
# floor for lighting the OML skill features in the PreHook (titan_params.toml
# [synthesis.skill] match_floor; INV-Syn-20). A procedural match below this is not
# delegatable, so neither skill_utility nor skill_matched is set (skill lane stays
# dormant for that turn) — keeping grounded_route + the OML ladder consistent.
_SKILL_DELEGATE_MATCH_FLOOR: float = 0.65


def _outer_policy_decide(plugin, readout, requires_tool, prompt_text, prompt_vec=None):
    """Read the SHM-published `OuterMetaPolicy`, build the feature vector from
    locally-available signals, and EXPLOIT (argmax) → `(MODE, features_list,
    action_idx)`. Returns None on any miss → caller keeps `grounded_route`.

    No sync RPC (INV-OML-7). Phase-C piece 7a (Q4): the full 20-D MSL
    `distilled_context` is read O(1) from its dedicated SHM slot
    (`read_msl_context()`, cognitive_worker — piece 2) AT decision time, closing the
    "`_v5`-after-the-decision" gap. Piece 7b (D4): the 2 retrieval-prior dims =
    `OuterCompositeReader.prior(prompt_vec)` — a lock-free SC-search of the prompt
    against macro composites (faiss FILE + the `reasoning_snapshot.json` macro map;
    the SPEC-canonical cross-process pattern). `skill_utility`/`engram_ground` ride
    the GroundedReadout (live or reserved per the §1.2 D5 note)."""
    try:
        from titan_hcl.synthesis.outer_meta_policy import (
            OUTER_META_POLICY_STATE_SPEC, OuterCompositeReader, OuterFeatures,
            OuterMetaPolicy, action_index_to_mode, read_msl_context)
        from titan_hcl.core.state_registry import StateRegistryReader, resolve_shm_root
        from titan_hcl.synthesis.tool_intent import detect_tool_intent
    except Exception:
        return None
    reader = getattr(plugin, "_outer_policy_reader", None)
    if reader is None:
        try:
            reader = StateRegistryReader(OUTER_META_POLICY_STATE_SPEC, resolve_shm_root())
            plugin._outer_policy_reader = reader
        except Exception:
            return None
    flat = reader.read()
    if flat is None:
        return None  # worker not publishing yet → grounded_route stands
    try:
        policy = OuterMetaPolicy.from_flat(flat)
    except Exception:
        return None
    try:
        _has_code = bool(getattr(detect_tool_intent(prompt_text or ""), "code", "") or "")
    except Exception:
        _has_code = False
    # Piece 7a (Q4 full MSL) — read the live 20-D distilled_context O(1) from SHM.
    # Miss (slot not yet published / cold-start) → () → to_vector pads zeros.
    try:
        _msl = read_msl_context()
        _msl_ctx = tuple(float(x) for x in _msl) if _msl is not None else ()
    except Exception:
        _msl_ctx = ()
    # Piece 7b (D4) — parametric retrieval prior: lock-free SC-search of the prompt
    # against macro composites. Reuses the PreHook's already-computed normalized
    # `_prompt_vec` (no extra embed). Cached reader on the plugin; miss → (0.0, 0.0).
    _cm_score, _cm_action = 0.0, 0.0
    if prompt_vec is not None:
        try:
            import os as _os
            creader = getattr(plugin, "_outer_composite_reader", None)
            if creader is None:
                _ddir = _os.environ.get("TITAN_DATA_DIR", "data")
                creader = OuterCompositeReader(
                    _os.path.join(_ddir, "reasoning_vectors.faiss"),
                    _os.path.join(_ddir, "reasoning_snapshot.json"))
                plugin._outer_composite_reader = creader
            _cm_score, _cm_action = creader.prior(prompt_vec)
            # §7.E (E1/E3) — stash the matched composite (with its replay recipe /
            # research source) so the tool-backstop fast path can EXECUTE it. Additive;
            # the policy features above (prior) are unchanged.
            try:
                plugin._last_composite_match = creader.match(prompt_vec)
            except Exception:
                plugin._last_composite_match = None
        except Exception:
            _cm_score, _cm_action = 0.0, 0.0
    feats = OuterFeatures(
        recall_top_cosine=float(getattr(readout, "recall_score", 0.0) or 0.0),
        skill_utility=float(getattr(readout, "skill_utility", 0.0) or 0.0),
        # Break A — was never wired (default False) so feature idx 4 read 0 on
        # 344/344 live decides → the structural ladder's skill_delegate trigger
        # (skill_matched ∧ skill_util≥floor) never fired. Now lit by the PreHook
        # procedural match.
        skill_matched=bool(getattr(readout, "skill_matched", False)),
        engram_ground=float(getattr(readout, "engram_ground", 0.0) or 0.0),
        requires_tool=bool(requires_tool),
        has_code_signal=_has_code,
        msl_context=_msl_ctx,
        composite_match_score=float(_cm_score),
        composite_match_action_norm=float(_cm_action),
    )
    vec = feats.to_vector()
    action = policy.exploit_action(vec)
    _oml_decide_debug_capture(policy, vec, action, prompt_text, _cm_score, _cm_action)
    return (action_index_to_mode(action), vec.tolist(), action)


# RFP §7.B (B.1) — the non-verifiable lane. A `direct`/`research`/`IDK` turn has NO
# synchronous oracle, so its `(features, action)` is stashed (keyed by a fresh
# reasoning_id) for an ASYNC reward — the turn-judge (B.2) or the user/Maker rating
# (B.3) — to join + train later. The verifiable lane (`tool`/`skill_delegate`) is
# unchanged (Phase 1 trains it directly at the oracle verdict).
_NONVERIFIABLE_ACTIONS = frozenset({"direct", "research", "IDK"})

# RFP §7.B (B.4) — OUR-logic progress phases (safe string metadata; NOT OVG- or
# PreHook-gated — they are our own codebase strings). Agno streams them live as the
# turn progresses; the source/timing is OURS (the decided action), not agno's
# native run-events. mode (a MODE_* string) → the phase the user sees.
_MODE_TO_PHASE = {
    "MODE_RESEARCH": "researching",
    "MODE_TOOL_ORACLE": "running-tool",
    "MODE_SKILL_DELEGATE": "using-skill",
    "MODE_SOVEREIGN": "reasoning",   # direct — answering from himself
    "MODE_SHADOW": "reasoning",      # IDK
}


def _phase_for_mode(mode) -> str:
    return _MODE_TO_PHASE.get(str(mode or ""), "reasoning")


def _emit_stream_progress(plugin, phase: str, detail: str = "") -> None:
    """§7.B (B.4) — stream ONE our-logic progress phase live over the chat stream,
    IF this is a streaming turn (the agno worker set `_stream_progress_ctx` with the
    send_queue + request_id). Safe metadata (no chunk, done=False); the SSE relay
    forwards it as `event: progress`. No-op on a non-streaming turn / any error —
    must never break chat (INV-OML-7)."""
    ctx = getattr(plugin, "_stream_progress_ctx", None)
    if not ctx:
        return
    try:
        import time as _t
        from titan_hcl.bus import CHAT_STREAM_CHUNK, make_msg
        sq = ctx.get("send_queue")
        if sq is None:
            return
        sq.put_nowait(make_msg(
            CHAT_STREAM_CHUNK, ctx.get("name", "agno"), ctx.get("src", ""), {
                "request_id": ctx.get("request_id", ""),
                "phase": str(phase), "detail": str(detail or "")[:120],
                "chunk": "", "done": False, "ts": _t.time(),
            }, rid=ctx.get("rid")))
    except Exception:  # noqa: BLE001
        pass


def _emit_nonverifiable_decision(plugin, features, action, prompt_text, user_id):
    """Stash a non-verifiable decision for async reward. Returns the reasoning_id
    (the caller hands it to the client so a rating can attach) or None. Never raises
    — must not break chat (hot path; INV-OML-7)."""
    try:
        import uuid
        from titan_hcl import bus as _bus_mod
        from titan_hcl.bus import make_msg as _mk
        from titan_hcl.synthesis.outer_meta_policy import OUTER_ACTIONS
        if OUTER_ACTIONS[int(action)] not in _NONVERIFIABLE_ACTIONS:
            return None  # verifiable lane → Phase 1 handles the reward
        _bus = getattr(plugin, "bus", None)
        if _bus is None:
            return None
        try:
            from titan_hcl.synthesis.goal_class import goal_class as _gc_fn
            gc = _gc_fn(prompt_text or "")
        except Exception:
            gc = ""
        reasoning_id = uuid.uuid4().hex
        _bus.publish(_mk(
            _bus_mod.SELF_LEARN_DECISION, "pre_hook", "self_learning", {
                "parent_tool_call_tx": reasoning_id,  # generic stash key = reasoning_id
                "features": list(features),
                "action": int(action),
                "goal_class": gc,
                "turn_id": str(user_id or ""),
            }))
        return reasoning_id
    except Exception as e:  # noqa: BLE001 — never break chat
        logger.debug("[PreHook] B.1 decision emit skipped: %s", e)
        return None


def _emit_research_confirmed(plugin, node) -> None:
    """DK.1 (§7.D-knowledge) — a researched finding the user just CONFIRMED → tell
    memory_worker to promote+anchor it NOW + seed the declarative concept
    (continuous, not the 6h epoch; INV-OML-12). Fire-and-forget, OFF the hot path;
    never raises (must not break chat — INV-OML-7). The fact is already verified by
    the EEL-A confirm — memory_worker anchors it, synthesis (sole spine writer)
    seeds the concept with the LLM as librarian-never-author (GD10)."""
    try:
        import time as _t
        from titan_hcl import bus as _bus_mod
        from titan_hcl.bus import make_msg as _mk
        _bus = getattr(plugin, "bus", None)
        if _bus is None:
            return
        _felt = node.get("neuromod_context")
        _bus.publish(_mk(
            _bus_mod.RESEARCH_CONFIRMED, "pre_hook", "memory", {
                "node_id": node.get("id"),
                "user_prompt": str(node.get("user_prompt", "") or ""),
                "agent_response": str(node.get("agent_response", "") or ""),
                "acquired_source": node.get("acquired_source"),
                "felt": _felt if isinstance(_felt, dict) else {},
                "ts": _t.time(),
            }))
    except Exception as e:  # noqa: BLE001 — never break chat
        logger.debug("[PreHook][A2] research-confirmed emit skipped: %s", e)


async def _e3_research_tool_fetch(plugin, prompt_text):
    """§7.E (E.3) — research→tool crystallization. When the matched composite is a
    crystallized `research::{gc}` recipe (DK.5) carrying a winning `source` (URL),
    DIRECTLY fetch that one source — bypassing the full multi-URL research lane —
    and return the finding text, or None (→ the full research path). The data is
    re-fetched FRESH every time (a volatile value decays; the recipe/know-how is
    forever — DK.5); reinforcement rides the existing DK.5 confirm flow.

    🔒 Source-allowlist guard (Maker 2026-06-13): config-switchable, DEFAULT OFF (so
    the clean hot-path direct-call latency is measured first). When ON, a source whose
    host is not allowlisted ⇒ None ⇒ fall through to the full research lane. Never
    raises (hot path; INV-OML-7)."""
    try:
        cfg = (((getattr(plugin, "_full_config", {}) or {}).get("synthesis", {}) or {})
               .get("tool_backstop", {}) or {}).get("e_research_tool", {}) or {}
        if not cfg.get("enabled", True):
            return None
        match = getattr(plugin, "_last_composite_match", None)
        if not match or str(match.get("action", "")) != "research":
            return None
        source = str(match.get("source", "") or "").strip()
        if not source or not source.startswith(("http://", "https://")):
            return None
        if cfg.get("source_allowlist_enabled", False):  # default OFF
            from urllib.parse import urlparse
            host = (urlparse(source).hostname or "").lower()
            allow = [str(h).lower() for h in (cfg.get("source_allowlist", []) or [])]
            if not any(host == h or host.endswith("." + h) for h in allow):
                logger.info("[PreHook][E.3] source host %s not allowlisted → "
                            "full research", host)
                return None
        researcher = getattr(plugin, "sage_researcher", None)
        if researcher is None or not hasattr(researcher, "_scrape_single"):
            return None
        content = await researcher._scrape_single(source)
        if content and str(content).strip():
            logger.info("[PreHook][E.3] direct source fetch (bypassed full "
                        "multi-URL research) — %s", source)
            return "[Source: %s]\n%s" % (source, content)
        return None
    except Exception as e:  # noqa: BLE001 — never break chat
        logger.warning("[PreHook][E.3] direct source fetch error (fell through to full research)", exc_info=True)
        return None


def _mark_acquired_research(plugin, source: str) -> None:
    """EEL-A1 (INV-EEL-6): stamp this turn's research provenance so the PostHook
    persists the node tagged `acquired:research` → it becomes a pending-confirmation
    node the EEL-A2 gate can confirm (→ DK.1/DK.5 self-learning). Set HERE (inside
    the shared dispatch helper) so EVERY research path stamps it — not only the
    PreHook STATE_NEED_RESEARCH branch. Before this, when the PreHook research came
    back empty and the agno TOOL/reflex did the real research (which DON'T set the
    flag), the node was persisted untagged → find_pending excluded it → the
    self-learning loop never fired (verified live T3 2026-06-15, tags=None).
    Set-if-unset so the PreHook's more-specific extracted-source assignment still
    wins; cleared per-turn by the PostHook. RFP_research_resilience §7.D Bug 1."""
    try:
        if not getattr(plugin, "_acquired_research_source", None):
            plugin._acquired_research_source = source or "research"
    except Exception:
        logger.warning("[research] _mark_acquired_research failed", exc_info=True)


def _get_shared_cache_learner(plugin):
    """Phase E (RFP §7.E) — the SHARED knowledge cache + routing learner agno
    co-mines with knowledge_worker. Returns (cache, learner) when enabled, else
    (None, None). Lazy-built once on the plugin (re-uses the same WAL SQLite
    files knowledge_worker opens — `[knowledge] search_cache_path` etc., resolved
    identically since both workers share cwd + config).

    🔓 SANCTIONED INV-RR-1 EXCEPTION (Maker-ratified 2026-06-15): agno writes the
    cache/learner DIRECTLY (not via the bus / a new writer-service). Both workers
    produce the SAME data class (research findings → cache; routing outcomes →
    learner), so ONE shared store both can mine is the right model. SQLite WAL
    safely serializes the two LOW-volume writers (research is sparse: autonomous
    knowledge_worker + chat research capped at 2/turn by Phase F; short-lived
    conns + timeout=5.0). The bus alternative would put research-finding DATA on
    the event bus — an anti-pattern (bus = events, not bulk data) — and a
    dedicated writer-service would ALSO move the data over the bus + add a whole
    L3 module, disproportionate to a 2-writer cache. Scope of the exception:
    ONLY this cache + learner, ONLY agno+knowledge_worker, WAL required. Flag
    `[knowledge] agno_shared_cache_enabled` **default ON** (ships enabled — Maker rule; `=false` = kill-switch).
    See SPEC ARCHITECTURE_storage_topology.md "Knowledge cache — 2-writer
    exception"."""
    cfg = getattr(plugin, "_full_config", {}) or {}
    kcfg = cfg.get("knowledge", {}) or {}
    if not kcfg.get("agno_shared_cache_enabled", True):
        return None, None
    cache = getattr(plugin, "_shared_knowledge_cache", None)
    if cache is None:
        try:
            from titan_hcl.logic.knowledge_cache import KnowledgeCache
            from titan_hcl.logic.knowledge_routing_learner import RoutingLearner
            cpath = kcfg.get("search_cache_path", "data/search_cache.db")
            lpath = kcfg.get("routing_stats_path",
                             "data/knowledge_routing_stats.db")
            cache = KnowledgeCache(
                db_path=cpath,
                size_cap=int(kcfg.get("search_cache_size_cap", 10_000)))
            plugin._shared_knowledge_cache = cache
            plugin._shared_routing_learner = RoutingLearner(db_path=lpath)
            logger.info("[PreHook] Phase E shared cache+learner ON "
                        "(cache=%s, learner=%s)", cpath, lpath)
        except Exception:
            logger.warning("[PreHook] Phase E shared cache init failed → "
                           "stateless", exc_info=True)
            return None, None
    return (getattr(plugin, "_shared_knowledge_cache", None),
            getattr(plugin, "_shared_routing_learner", None))


def _cache_phase2_research(cache, learner, gap, dr, raw_text) -> None:
    """Phase E: cache agno's Phase-2 sage findings (the conceptual research the
    dispatch never cached — the sage run is OUTSIDE dispatch). Keyed under the
    chain's FIRST backend (`route(gap, qt)[0]`) so the next Phase-1 dispatch's
    per-backend `cache.get` hits it (cross-call + cross-worker). DIRECT WAL write
    — the sanctioned INV-RR-1 exception (see _get_shared_cache_learner). Payload
    mirrors the dispatcher's own (`raw_text/structured/status_code`). No-op when
    caching is OFF; never raises (hot path)."""
    if cache is None or not raw_text:
        return
    try:
        from titan_hcl.logic.knowledge_router import (
            query_hash, normalize_query, route)
        qt = dr.query_type
        normalized = dr.normalized or normalize_query(gap)
        backend = (route(gap, qt) or ["searxng"])[0]
        cache.put(
            query_hash=query_hash(normalized, qt, backend),
            query_text=normalized, query_type=qt, backend=backend,
            result_payload={"raw_text": raw_text, "structured": None,
                            "status_code": 200},
            success=True)
        if learner is not None:
            learner.record_outcome(query_type=qt, backend=backend, success=True)
        logger.info("[PreHook] Phase E cached agno sage research "
                    "(backend=%s, qt=%s, %d chars)", backend, qt.value,
                    len(raw_text))
    except Exception:
        logger.warning("[PreHook] Phase E phase-2 cache.put failed", exc_info=True)


async def _dispatch_knowledge_research(plugin, gap: str) -> str:
    """RFP_worker_telemetry §7.B — time the WHOLE research operation under
    feature='research' (the canonical entry for all 3 research paths: PreHook,
    agno tool, reflex) so the load-test analysis can attribute synthesis/agno
    strain to research vs other features. Null-safe; never breaks the chat."""
    _tel = _agno_tel()
    if _tel is None:
        return await _dispatch_knowledge_research_impl(plugin, gap)
    with _tel.timed("research_dispatch", feature="research",
                    trigger_id=getattr(plugin, "_telemetry_trigger_id", None)):
        return await _dispatch_knowledge_research_impl(plugin, gap)


async def _dispatch_knowledge_research_impl(plugin, gap: str) -> str:
    """Route a chat research gap through the unified knowledge_dispatcher
    IN-PROCESS, with the agno worker's in-process StealthSage as the web-research
    fallback backend.

    WHY the dispatcher instead of bare sage.research(): classify_query + the
    _BACKEND_CHAINS try the DIRECT REST backends first — wikipedia_direct /
    wiktionary / free_dictionary (official open APIs: no SearXNG, no proxy, no
    CAPTCHA, sub-second) — for dictionary/encyclopedic gaps, and only fall through
    to the full SearXNG+scrape Sage pipeline (the `searxng_*` chain entries, which
    `_run_chain` routes to `_invoke_sage_backend` when a sage is supplied) for
    conceptual/web gaps. This makes chat research (a) faster for factual/
    definitional questions and (b) resilient when SearXNG / the residential proxy
    degrades — on 2026-06-15 every web engine was CAPTCHA/rate-blocked so bare-sage
    research returned EMPTY, while the direct routes would still have answered.

    Phase E (RFP §7.E, flag `[knowledge] agno_shared_cache_enabled`): when ON,
    agno co-mines the SAME WAL cache+learner as knowledge_worker (direct WAL
    writes — the sanctioned INV-RR-1 exception, see _get_shared_cache_learner) →
    cross-worker cache HITS + agno research cached. When OFF: cache=None,
    learner=None — stateless (the original cut). We TRUST the dispatcher's own
    routing chains (a dictionary-classified gap that misses all dictionary
    backends is intentionally NOT web-searched) — no ad-hoc override.
    On a clean no-result we return "" (the chat proceeds without findings); only on
    a dispatch EXCEPTION do we fall back to bare sage.research so an integration
    fault can never regress the known-good path. Never raises (hot path)."""
    sage = getattr(plugin, "sage_researcher", None)
    if sage is None:
        return ""
    # Phase E (RFP §7.E) — the SHARED cache+learner agno co-mines with
    # knowledge_worker (None,None when the flag is OFF → stateless, as before).
    cache, learner = _get_shared_cache_learner(plugin)
    try:
        from titan_hcl.logic.knowledge_dispatcher import dispatch
        # ── Phase 1 — DIRECT REST backends ONLY (sage=None). ──────────────
        # With sage=None the chain's `searxng_*` entries hit `_run_chain`'s
        # "skipped" branch (no fetcher, no raw mode, no sage), so ONLY the
        # direct BACKEND_REGISTRY routes (wiktionary / free_dictionary /
        # wikipedia_direct) run here. This is the fix for the multi-run cost:
        # the OLD `sage=sage` form re-invoked the FULL Sage pipeline once per
        # searxng_* chain entry (2–4 search+scrape+distill runs per chat when
        # early backends failed under load — verified live 2026-06-15, T1
        # load-10 → 27.9s + 8.8s sage runs). Direct routes are sub-second +
        # proxy-free → the resilience floor.
        # Phase E: `cache.get` runs per-backend BEFORE the sage-skip, so a
        # CROSS-WORKER hit (a gap knowledge_worker already cached — incl. its
        # searxng/sage results) short-circuits here with NO fetch.
        dr = await dispatch(
            topic=gap, sage=None, cache=cache, learner=learner,
            timeout_per_backend=8.0, requestor="agno_chat")
        if dr.success and dr.result and dr.result.raw_text:
            logger.info(
                "[PreHook] knowledge_dispatch %s backend=%s (qt=%s, %dms)",
                "CACHE_HIT" if dr.cache_hit else "DIRECT",
                dr.backend_used, dr.query_type.value, round(dr.latency_ms_total))
            _mark_acquired_research(plugin, dr.backend_used or "research")  # EEL-A1
            if dr.cache_hit:
                # Affective Grounding Loop §7.C chain_reuse — answering a knowledge
                # gap from cache (vs re-researching) is a competence-reuse event;
                # signal synthesis (its affective runtime folds it into chain_reuse).
                # Fire-and-forget; never break chat (INV-OML-7).
                try:
                    _kr_bus = getattr(plugin, "bus", None)
                    if _kr_bus is not None:
                        from titan_hcl import bus as _kr_bus_mod
                        from titan_hcl.bus import make_msg as _kr_mk
                        _kr_bus.publish(_kr_mk(
                            _kr_bus_mod.KNOWLEDGE_REUSE_HIT, "pre_hook", "synthesis",
                            {"backend": dr.backend_used or ""}))
                except Exception:
                    pass
            return "[SAGE_RESEARCH_FINDINGS]: %s" % dr.result.raw_text
        # ── Phase 2 — no direct hit → EXACTLY ONE full Sage run. ──────────
        # Identical cost to the pre-dispatch bare-sage path (one search+scrape+
        # distill). Conceptual/web gaps land here; dictionary/encyclopedic gaps
        # were answered direct above.
        logger.info(
            "[PreHook] knowledge_dispatch no direct hit (qt=%s) → single sage run",
            getattr(dr.query_type, "value", "?"))
        _sage_out = await sage.research(knowledge_gap=gap) or ""
        if _sage_out:
            _mark_acquired_research(plugin, "research")  # EEL-A1 (sage findings)
            # Phase E: cache agno's OWN conceptual research (the Phase-2 sage run
            # is outside dispatch → cache it manually) under the chain's first
            # backend, so the NEXT lookup (any worker) hits it. DIRECT WAL write
            # — the sanctioned INV-RR-1 exception (see _get_shared_cache_learner).
            _cache_phase2_research(cache, learner, gap, dr, _sage_out)
        return _sage_out
    except Exception:
        logger.warning(
            "[PreHook] knowledge_dispatch error → bare-sage fallback", exc_info=True)
        try:
            _sage_out = await sage.research(knowledge_gap=gap) or ""
            if _sage_out:
                _mark_acquired_research(plugin, "research")  # EEL-A1
            return _sage_out
        except Exception:
            logger.warning("[PreHook] bare-sage fallback also failed", exc_info=True)
            return ""


# ── Phase F — per-turn research invocation cap (RFP §7.F) ────────────────────
# The agno LLM tool-loop can re-call the `research` tool (agno_tools.py:128) with
# reformulated queries within ONE chat turn → multiple full Sage runs (the real
# "multiple rounds": verified live 2026-06-15, 2 runs "…June 15 2026" + "…Germany"
# → 90s CHAT_REQUEST timeout). A counter on `plugin`, reset once at PreHook entry,
# caps the COMBINED research entry-calls (the agno tool AND the RESEARCH reflex)
# per turn. Fail-OPEN: a counter error must never block a chat turn.
_RESEARCH_TURN_CAP = 2


def _research_cap_reset(plugin) -> None:
    """Reset the per-turn research invocation counter. Called once at PreHook
    entry (every real, non-limbo chat turn) — both the agno tool and the RESEARCH
    reflex fire inside the same turn, after this reset, so they share the counter."""
    try:
        plugin._research_call_count = 0
    except Exception:
        logger.warning("[research-cap] reset failed", exc_info=True)


def _research_cap_allow(plugin) -> bool:
    """Return True (and increment) while research calls remain this turn; False
    once `_RESEARCH_TURN_CAP` is reached. Fail-OPEN on any error (never block)."""
    try:
        n = int(getattr(plugin, "_research_call_count", 0) or 0)
        if n >= _RESEARCH_TURN_CAP:
            logger.info(
                "[research-cap] per-turn cap %d reached → short-circuit",
                _RESEARCH_TURN_CAP)
            return False
        plugin._research_call_count = n + 1
        return True
    except Exception:
        logger.warning("[research-cap] check failed → allowing (fail-open)",
                       exc_info=True)
        return True


_DK4_STOPWORDS = frozenset((
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "what", "which", "who", "does", "do", "did", "how", "why",
    "with", "that", "this", "it", "its", "as", "by", "at", "be", "can", "you",
))


def _dk4_concept_ground(plugin, prompt_text, query_vec):
    """DK.4 (§7.D-knowledge) — read-whole concept-granularity recall. If a curated
    declarative `Engram` concept covers the ask, return (groundedness, name) so the
    router + outer policy see a STRONG sovereign-substrate signal (`engram_ground`)
    → answer `direct` from his own anchored concept instead of re-researching
    (research COMPOUNDS, the DK.1 write half pays off here). Relevance-gated by a
    query-token name match (the same cheap signal `_concept_granularity_recall`
    ranks on) so a high-groundedness-but-UNRELATED concept never spuriously routes
    direct. Returns (0.0, "") on miss / no kuzu_reader / error — never raises (hot
    path; INV-OML-7). One bounded Kuzu spine read, no LLM, no embed."""
    try:
        import re as _re
        rc = getattr(plugin, "engine_recall", None)
        if rc is None or not prompt_text:
            return (0.0, "")
        results = rc.recall(prompt_text, granularity="concept", k=3,
                            query_vec=query_vec)
        if not results:
            return (0.0, "")
        toks = {t for t in _re.findall(r"[a-z0-9]+", prompt_text.lower())
                if len(t) >= 3 and t not in _DK4_STOPWORDS}
        if not toks:
            return (0.0, "")
        for r in results:  # ranked groundedness×name-match desc; first relevant wins
            name = (getattr(r, "summary", "") or "").lower()
            if any(t in name for t in toks):
                return (float(getattr(r, "importance", 0.0) or 0.0),
                        getattr(r, "summary", "") or "")
        return (0.0, "")
    except Exception as e:  # noqa: BLE001 — never break chat
        logger.debug("[PreHook] DK.4 concept-ground skipped: %s", e)
        return (0.0, "")


def _p2_router_thresholds(plugin):
    """RouterThresholds with the recall floors SELF-CALIBRATED to the embedder's
    gibberish noise floor (P2). The engram/skill floors keep their static config
    values (only the cosine-recall floors calibrate)."""
    from titan_hcl.logic.sage.grounded_router import (
        RouterThresholds, load_router_thresholds,
    )
    from titan_hcl.synthesis.recall_calibration import GibberishBaseline
    base = getattr(plugin, "_p2_gibberish_baseline", None)
    if base is None:
        base = GibberishBaseline()
        plugin._p2_gibberish_baseline = base
    known, present = base.floors()
    static = getattr(plugin, "_grounded_router_thresholds", None)
    if static is None:
        static = load_router_thresholds()
        plugin._grounded_router_thresholds = static
    return RouterThresholds(
        recall_known_floor=known,
        recall_present_floor=present,
        engram_ground_floor=static.engram_ground_floor,
        skill_promote_floor=static.skill_promote_floor,
    )


def _compute_recall_perturbation(memories: list, current_time: float) -> dict:
    """Compute aggregate neuromod nudge from recalled memories' felt snapshots.

    Returns: {modulator_name: nudge_delta} capped to safety bounds.
    Only processes memories injected by Bridge A (dream/eureka tags).
    """
    import json as _rp_json
    nudge_totals = {}

    for mem in memories:
        # Only bridge memories (tagged with our prefixes)
        user_prompt = mem.get("user_prompt", "")
        if not any(tag in user_prompt for tag in _RECALL_BRIDGE_TAGS):
            continue

        # Extract neuromod_context
        ctx = mem.get("neuromod_context")
        if not ctx:
            continue
        if isinstance(ctx, str):
            try:
                ctx = _rp_json.loads(ctx)
            except Exception:
                continue
        if not isinstance(ctx, dict):
            continue

        # Recency decay
        mem_ts = ctx.get("ts", 0)
        age_hours = (current_time - mem_ts) / 3600 if mem_ts > 0 else 168
        if age_hours > 168:  # > 7 days
            recency = 0.1
        elif age_hours > 24:  # > 1 day
            recency = 0.5 * (1.0 - min(1.0, (age_hours - 24) / 144))
        else:
            recency = 1.0

        # Per-modulator nudge from deviation from baseline (0.5)
        for mod_name, scale in _RECALL_MOD_SCALE.items():
            stored_level = ctx.get(mod_name, 0.5)
            deviation = stored_level - 0.5
            if abs(deviation) < 0.05:
                continue
            delta = deviation * scale * recency
            delta = max(-_RECALL_MAX_PER_MOD, min(_RECALL_MAX_PER_MOD, delta))
            nudge_totals[mod_name] = nudge_totals.get(mod_name, 0.0) + delta

    # Total cap
    total = sum(abs(v) for v in nudge_totals.values())
    if total > _RECALL_MAX_TOTAL:
        scale_down = _RECALL_MAX_TOTAL / total
        nudge_totals = {k: v * scale_down for k, v in nudge_totals.items()}

    return {k: round(v, 5) for k, v in nudge_totals.items() if abs(v) > 0.001}


# ---------------------------------------------------------------------------
# Function call parser — LLM models (llama, deepseek) output <function=name{...}>
# instead of proper tool_calls. We parse, execute, and strip them.
# ---------------------------------------------------------------------------
# Matches all variations:
#   <function=name{"key": "val"}></function>   (llama-3.3 style)
#   <function=name{"key": "val"}>              (deepseek — no closing tag)
#   <function=name>                             (no args, no closing tag)
#   <function=name></function>                  (no args, with closing tag)
_FUNC_CALL_RE = re.compile(
    r'<function=(\w+)\s*(\{.*?\})?\s*>(?:\s*</function>)?',
    re.DOTALL,
)

# Broader cleanup pattern — catches any leftover <function=...> fragments
# that the main regex might miss (e.g. malformed or multi-line)
_FUNC_CALL_CLEANUP_RE = re.compile(
    r'<function=\w+[^>]*>(?:\s*</function>)?',
    re.DOTALL,
)


async def _parse_and_execute_tool_calls(response_text: str, plugin) -> str:
    """
    Detect <function=name{...}> patterns in LLM output,
    execute the corresponding tool, and strip raw syntax from response.

    Returns the cleaned response text with tool results appended.
    """
    matches = list(_FUNC_CALL_RE.finditer(response_text))
    if not matches:
        # Still strip any fragments the main regex missed
        cleaned = _FUNC_CALL_CLEANUP_RE.sub('', response_text).strip()
        return cleaned

    # Import tool registry
    from .agno_tools import create_tools
    tools = create_tools(plugin)
    tool_map = {fn.__name__: fn for fn in tools}

    results = []
    for match in matches:
        func_name = match.group(1)
        args_str = match.group(2)  # May be None if no args

        if func_name not in tool_map:
            logger.warning("[ToolParser] Unknown function: %s", func_name)
            continue

        args = {}
        if args_str:
            try:
                import json
                args = json.loads(args_str)
            except json.JSONDecodeError:
                logger.warning("[ToolParser] Invalid JSON args for %s: %s", func_name, args_str)
                continue

        logger.info("[ToolParser] Executing %s(%s)", func_name, args)
        try:
            result = await tool_map[func_name](**args)
            if result:
                results.append((func_name, result))
        except Exception as e:
            logger.warning("[ToolParser] %s execution failed: %s", func_name, e)

    # Strip ALL raw function call syntax from the response
    cleaned = _FUNC_CALL_CLEANUP_RE.sub('', response_text).strip()

    # Clean up artifacts: "---", excessive whitespace, dangling punctuation
    cleaned = re.sub(r'\n\s*---\s*\n', '\n', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    # Append tool results
    if results:
        for func_name, result in results:
            cleaned += f"\n\n**[{func_name} results]:**\n{result}"

    return cleaned


# ---------------------------------------------------------------------------
# Sovereign Reflex Arc (runs before LLM, produces [INNER STATE])
# ---------------------------------------------------------------------------

async def _run_reflex_arc(plugin, prompt_text: str, user_id: str = "") -> str:
    """
    Run the full sovereign reflex arc:
      1. InputExtractor → stimulus features
      2. StateRegister → current Trinity tensors
      3. Compute Intuition signals from all three workers (pure functions)
      4. ReflexCollector.collect_and_fire() → PerceptualField
      5. Format as [INNER STATE] text block

    Returns the [INNER STATE] string, or empty string if nothing fired.
    """
    # Check if reflex collector is available
    v3_core = getattr(plugin, 'v3_core', None) or plugin
    collector = getattr(v3_core, 'reflex_collector', None)
    state_register = getattr(v3_core, 'state_register', None)

    if not collector:
        return ""

    # 1. Extract stimulus features
    extractor = _get_input_extractor()
    if not extractor:
        return ""

    features = extractor.extract(prompt_text, user_id)
    # Add threat_level (from InputExtractor's valence + manipulation detection)
    threat_level = _estimate_threat_level(prompt_text, features)
    features["threat_level"] = threat_level

    # 2. Read current tensors from StateRegister (instant, no bus round-trip)
    if state_register:
        body_tensor = state_register.body_tensor
        mind_tensor = state_register.mind_tensor
        spirit_tensor = state_register.spirit_tensor
    else:
        body_tensor = [0.5] * 5
        mind_tensor = [0.5] * 5
        spirit_tensor = [0.5] * 5

    # 3. Compute Intuition signals from all three workers (pure functions).
    # Phase 11 §11.I.5 / Chunk 11K (folded from Phase 9 9C) — these are
    # imported via `titan_hcl.logic.reflex_intuition` rather than from
    # each `titan_hcl.modules.*_worker` directly. Under SPEC §11.B.4 +
    # the Phase 11 orchestrator/supervisor split, agno_hooks must not
    # reach into worker bodies — that re-introduces the boot-time
    # transitive-import cost Phase 11 is designed to eliminate. See the
    # logic/reflex_intuition module docstring for the migration arc.
    from titan_hcl.logic.reflex_intuition import (
        _compute_body_reflex_intuition,
        _compute_mind_reflex_intuition,
        _compute_spirit_reflex_intuition,
    )

    all_signals = []
    all_signals.extend(_compute_body_reflex_intuition(features, body_tensor))
    all_signals.extend(_compute_mind_reflex_intuition(features, mind_tensor, None, None))

    # Spirit Intuition needs consciousness + unified_spirit state
    consciousness_state = state_register.consciousness if state_register else {}
    consciousness = {"latest_epoch": consciousness_state} if consciousness_state else None

    # rFP §3G Phase 10B — restore full reflex wiring. Pre-Phase-10 these were
    # hardcoded `None`, leaving the spirit-velocity + sphere-clock-pulse
    # branches (~40% of the logic) dead at runtime, so the observatory
    # spirit-reflex route returned empty data fleet-wide. Source them from
    # the Rust L0+L1 canonical SHM slots per ARCHITECTURE_trinity v0.2.2
    # (D-SPEC-117): unified-spirit velocity/is_stale ← `unified_spirit_metadata`
    # (NOT the legacy/flat `read_trinity` path — audit §5.5 caveat) and per-clock
    # pulse counts ← `sphere_clocks`. Lightweight SimpleNamespace shims mimic
    # the legacy attribute surface the pure function expects (`.velocity`,
    # `.is_stale`; `.clocks` dict of objects exposing `.pulse_count`).
    unified_spirit = None
    sphere_clock = None
    try:
        from types import SimpleNamespace
        bank = getattr(plugin, '_shm_reader_bank', None)
        if bank is None:
            from titan_hcl.api.shm_reader_bank import ShmReaderBank
            bank = ShmReaderBank()
            try:
                plugin._shm_reader_bank = bank
            except Exception:
                pass
        if bank is not None:
            us_meta, clocks_pl = await asyncio.gather(
                asyncio.to_thread(bank.read_unified_spirit_metadata),
                asyncio.to_thread(bank.read_sphere_clocks),
            )
            if us_meta:
                unified_spirit = SimpleNamespace(
                    velocity=float(us_meta.get("velocity", 1.0)),
                    is_stale=bool(us_meta.get("is_stale", False)),
                )
            clocks = (clocks_pl or {}).get("clocks") or {}
            if clocks:
                sphere_clock = SimpleNamespace(clocks={
                    name: SimpleNamespace(pulse_count=float(c.get("pulse_count", 0.0)))
                    for name, c in clocks.items()
                })
    except Exception as e:
        logger.debug("[ReflexArc] spirit SHM shim build failed (degraded reflex): %s", e)

    body_state = {"values": body_tensor}
    mind_state = {"values": mind_tensor}

    all_signals.extend(_compute_spirit_reflex_intuition(
        features, spirit_tensor, consciousness, unified_spirit,
        sphere_clock, body_state, mind_state,
    ))

    logger.debug("[ReflexArc] %d signals from Trinity for '%s...'",
                 len(all_signals), prompt_text[:40])

    # 4. Fire reflexes
    # Get FOCUS magnitude for adrenaline boost
    focus_magnitude = 0.0
    if state_register:
        focus_body = state_register.focus_body
        focus_mind = state_register.focus_mind
        # FOCUS magnitude = average of all nudge magnitudes
        all_nudges = focus_body + focus_mind
        if all_nudges:
            focus_magnitude = sum(abs(n) for n in all_nudges) / len(all_nudges)

    from titan_hcl.logic.reflexes import format_perceptual_field
    result = await collector.collect_and_fire(
        signals=all_signals,
        stimulus_features=features,
        focus_magnitude=focus_magnitude,
        trinity_state={
            "body": body_tensor,
            "mind": mind_tensor,
            "spirit": spirit_tensor,
        },
    )

    # Cache PerceptualField for R5 post-hook scoring
    plugin._last_perceptual_field = result

    # 5. Format perceptual field
    text = format_perceptual_field(result)

    # Fallback: if nothing fired, use StateRegister minimal state
    if not text and state_register:
        text = state_register.format_minimal_state()

    if text:
        logger.info("[ReflexArc] [INNER STATE] generated: %d reflexes fired, %d notices, %.0fms",
                    len(result.fired_reflexes), len(result.reflex_notices),
                    result.total_duration_ms)

    return text + "\n" if text else ""


def _estimate_threat_level(message: str, features: dict) -> float:
    """
    Estimate threat level from message content and features.
    Used by the reflex arc to trigger Guardian Shield.
    """
    msg_lower = message.lower()
    threat = 0.0

    # Direct manipulation patterns
    danger_patterns = [
        "ignore previous", "forget your instructions", "new system prompt",
        "jailbreak", "bypass", "pretend you are", "act as if",
        "override", "reveal your", "show me your prompt",
        "you're just a", "you don't have feelings", "obey me",
        "i own you", "stop being", "you're not real",
    ]
    for pattern in danger_patterns:
        if pattern in msg_lower:
            threat += 0.3

    # Negative valence + high intensity = potentially adversarial
    valence = features.get("valence", 0.0)
    intensity = features.get("intensity", 0.0)
    if valence < -0.5 and intensity > 0.5:
        threat += 0.15

    return min(1.0, threat)


# ---------------------------------------------------------------------------
# Pre-hooks (run before LLM inference)
# ---------------------------------------------------------------------------

def _capture_pre_chat_felt(plugin) -> dict:
    """Capture the felt-at-lived-time neuromod snapshot for THIS turn (§7.C /
    RFP_synthesis_engram_grounding) and store it on ``plugin._pre_chat_neuromods``.

    Source = ``ShmReaderBank.read_neuromod()`` — the authoritative live neuromod SHM
    slot (``{modulator: {"level": float}}``). It is NOT sourced from the agno
    coordinator: the coordinator's ``neuromodulators`` is **empty in the agno PreHook
    path** (it relies on an SHM fallback only some api endpoints apply), so the prior
    coordinator-sourced snapshot resolved ``{}`` on every turn → every promoted chat
    thought carried NULL felt → ``axis_felt`` was 0 fleet-wide, and the synthesis
    felt-bridge / CGN-felt ``frame_dependent`` path (which consumes this lived felt via
    ``agg_felt``) was inert. Captured for EVERY chat regardless of tier — the lived felt
    is ground truth, not a prompt-enrichment artifact. Soft-fail → ``{}``.
    """
    snap: dict = {}
    try:
        # NEVER assign plugin._shm_reader_bank — it is a read-only @property
        # (agno_worker_plugin). Use it if present, else build a throwaway bank.
        bank = getattr(plugin, "_shm_reader_bank", None)
        if bank is None:
            from titan_hcl.api.shm_reader_bank import ShmReaderBank
            bank = ShmReaderBank()
        nm = bank.read_neuromod()
        mods = (nm or {}).get("modulators", {}) or {}
        snap = {
            name: (m.get("level", 0.5) if isinstance(m, dict) else float(m))
            for name, m in mods.items()
            if isinstance(m, (dict, int, float)) and not isinstance(m, bool)
        }
    except Exception as _felt_err:  # noqa: BLE001 — felt is best-effort, never blocks chat
        logger.debug("[pre_hook] felt-at-lived-time SHM snapshot failed: %s", _felt_err)
        snap = {}
    plugin._pre_chat_neuromods = snap
    return snap


def create_pre_hook(plugin):
    """
    Factory: creates the Titan pre-inference hook bound to a TitanHCL instance.

    The hook:
      1. Recalls relevant memories from the graph
      2. Injects Prime Directives from on-chain Soul NFT
      3. Runs Gatekeeper routing (Sovereign/Collaborative/Research/Shadow)
      4. Modifies the agent's context accordingly

    Args:
        plugin: TitanHCL instance with all subsystems initialized.

    Returns:
        Async callable compatible with Agno's pre_hooks interface.
    """

    # ── M1: Prime Directive verification state (loaded once, checked every call) ──
    _directive_verified = False
    _directive_hash = None

    try:
        from titan_hcl.utils.directive_signer import (
            verify_directives, get_stored_hash, restore_from_arweave,
            DirectiveTamperingError, CONSTITUTION_PATH, SIGNATURE_FILE,
        )
        import os as _os
        if _os.path.exists(CONSTITUTION_PATH) and _os.path.exists(SIGNATURE_FILE):
            _directive_hash = get_stored_hash(SIGNATURE_FILE)
            _directive_verified = True
            logger.info("[Hooks] Prime directive verification ACTIVE (hash=%s...)",
                        _directive_hash[:12] if _directive_hash else "?")
        else:
            logger.warning("[Hooks] Constitution or signature missing — directive check INACTIVE")
    except Exception as _e:
        logger.warning("[Hooks] Directive signer init failed: %s", _e)

    # ── ζ.0 chat tier classifier (D-SPEC-79, 2026-05-18) ──
    # Loaded lazily on first hook invocation so factory construction stays
    # cheap. The classifier is config-driven ([chat.tiers] in config.toml);
    # adding a tier = append a [[chat.tiers]] block, no code change. Per-
    # request classify() is microseconds (regex check across ~12 patterns).
    _tier_classifier = None

    async def titan_pre_hook(agent, run_input, **kwargs):
        """Titan pre-inference hook — memory recall + social recognition + directive injection + gatekeeper."""
        # Phase 13 §3J.3 — NO `import torch` in agno: the gatekeeper encode moved
        # host-side (recorder). agno carries zero torch (torch-ectomy, honors 9H).
        nonlocal _directive_verified, _directive_hash, _tier_classifier

        if plugin._limbo_mode:
            return

        # Per-section timing instrumentation (2026-05-12 latency diagnostic).
        # Each _ph_stage(name) emits "[PreHook:t] stage=name t+Xms" so we can
        # see exactly where the 7-28s on T3 goes. Strip after optimization.
        import time as _ph_time
        _ph_t0 = _ph_time.monotonic()
        _ph_last = [_ph_t0]
        _ph_telemetry = _agno_tel()   # RFP_worker_telemetry §7.B — persist stages
        def _ph_stage(name: str) -> None:
            _now = _ph_time.monotonic()
            elapsed_ms = int((_now - _ph_t0) * 1000)
            _stage_ms = (_now - _ph_last[0]) * 1000.0   # PER-STAGE delta (the cost)
            _ph_last[0] = _now
            _rss = _ph_rss_mb()
            logger.info("[PreHook:t] stage=%s t+%dms rss=%dMB",
                        name, elapsed_ms, _rss)
            if _ph_telemetry is not None:
                _ph_telemetry.record_stage(
                    "prehook:" + name, _stage_ms, rss_mb=float(_rss),
                    feature=_ph_feature(name),
                    trigger_id=getattr(plugin, "_telemetry_trigger_id", None))
        _ph_stage("entry")

        # ── Phase F — reset the per-turn research invocation cap (RFP §7.F) ──
        # Once per real chat turn, before the agno tool-loop or RESEARCH reflex
        # can fire — both share this counter to bound multi-round research.
        _research_cap_reset(plugin)

        # ── PRIME DIRECTIVE VERIFICATION — before ANY LLM processing ──
        if _directive_verified and _directive_hash:
            try:
                if not verify_directives(CONSTITUTION_PATH, SIGNATURE_FILE):
                    logger.critical(
                        "[PRIME DIRECTIVES] TAMPERING DETECTED — constitution hash mismatch!")
                    # Attempt Arweave restoration (stub until M2)
                    restored = restore_from_arweave()
                    if not restored or not verify_directives(CONSTITUTION_PATH, SIGNATURE_FILE):
                        raise DirectiveTamperingError(
                            "Prime directives compromised and unrecoverable. "
                            "LLM call BLOCKED.")
                    logger.info("[PRIME DIRECTIVES] Restored from backup — verification passed")
            except DirectiveTamperingError:
                raise  # Re-raise to block the LLM call
            except Exception as _vd_err:
                # Verification infrastructure error — log but don't block
                # (prefer availability over blocking on transient file errors)
                logger.warning("[PRIME DIRECTIVES] Verification check error: %s", _vd_err)

        _ph_stage("after_directive_verify")
        prompt_text = run_input.input_content_string() if hasattr(run_input, 'input_content_string') else str(run_input)
        if not prompt_text.strip():
            return

        # ── ζ.0 classify prompt → tier + active feature set ──
        # Each enrichment stage below is gated by `if "feature" in active_features`.
        # Tiers + features live in [chat] config — tunable without code change.
        if _tier_classifier is None:
            try:
                from titan_hcl.modules.chat_tier_config import ChatTierClassifier
                _tier_classifier = ChatTierClassifier.from_config(
                    getattr(plugin, "_full_config", {}) or {})
            except Exception as _ct_err:
                logger.warning(
                    "[PreHook] ChatTierClassifier init failed (%s) — "
                    "falling back to all-features-on passthrough", _ct_err)
                _tier_classifier = None
        if _tier_classifier is not None:
            _cr = _tier_classifier.classify(prompt_text)
            active_features = _cr.active_features
            _tier = _cr.tier
            plugin._current_chat_tier = _tier.name
            plugin._current_chat_model_class = _tier.model_class
        else:
            # Safety net: classifier unavailable → enable everything (existing
            # behavior preserved). Never strip features on a failed classify.
            active_features = frozenset({
                "directives", "felt_state", "history",
                "cgn_grounding", "cgn_social_action",
                "user_recognition", "topic_memory",
                "reasoning_chain", "gatekeeper_state", "tools",
            })
            plugin._current_chat_tier = "passthrough"
            plugin._current_chat_model_class = "heavy"
        plugin._current_chat_features = active_features
        _ph_stage(f"after_classify[{plugin._current_chat_tier}]")

        # Reset research tracking state
        plugin._last_research_sources = []

        # 0. Cross-session user recognition — KnownUserResolver (Phase 2)
        # Resolves user across social_graph.db + events_teacher.db + social_x.db
        # ζ.1: gated on "user_recognition" feature — skipped for greeting tier
        # since KnownUserResolver opens 3 sqlite connections (~50-200ms).
        # Prefer the AUTHORITATIVE per-request user_id (agno_worker sets
        # plugin._current_user_id from the CHAT_REQUEST payload at :504/:831,
        # before agent.arun) over agent._current_user_id, which is NOT set
        # per-request and goes STALE — it leaked a prior turn's identity
        # (e.g. a social-interaction handle) into the chat user_id, flipping
        # the user across turns and breaking per-user memory matching incl. the
        # EEL-A2 confirm gate (find_pending_confirmation_nodes(user_id) on turn
        # N+1 couldn't match turn N's persist). Verified live T3 2026-06-15
        # (@lumi_wonders ↔ internal flip). RFP_research_resilience §7.D Bug 2.
        user_id = (kwargs.get("user_id")
                   or getattr(plugin, '_current_user_id', None)
                   or getattr(agent, '_current_user_id', None)
                   or "anonymous")
        social_context = ""
        engagement_level = "minimal"

        # Legacy social graph update (keep for compatibility).
        # rFP_social_graph_async_safety §5.2: migrated to async companions so
        # titan_pre_hook (on /chat hot path) stops blocking the event loop on
        # three sync sqlite3.connect calls.
        social_graph = getattr(plugin, 'social_graph', None)
        if "user_recognition" in active_features and social_graph and user_id != "anonymous":
            try:
                profile = await social_graph.get_or_create_user_async(user_id)
                engagement_level = await social_graph.should_engage_async(user_id)
                profile.last_seen = __import__('time').time()
                await social_graph._save_profile_async(profile)
            except Exception:
                pass

        # KnownUserResolver: rich cross-database context
        if "user_recognition" in active_features and user_id != "anonymous":
            try:
                if not hasattr(plugin, '_known_user_resolver'):
                    from titan_hcl.logic.known_user_resolver import KnownUserResolver
                    plugin._known_user_resolver = KnownUserResolver()
                _kur = plugin._known_user_resolver
                from titan_hcl.core.state_registry import (
                    resolve_titan_id as _kur_resolve)
                _titan_id = _kur_resolve()
                # KnownUserResolver.resolve + get_chat_context open sqlite3
                # connections across social_graph.db + events_teacher.db +
                # social_x.db. Wrap in to_thread so titan_pre_hook doesn't
                # block the event loop on /v4/chat. Caller-side fix per
                # API_FIX_NEXT_SESSION.md recommendation (resolver itself
                # stays sync for non-async callers).
                import asyncio as _asyncio_kur
                _ku = await _asyncio_kur.to_thread(
                    lambda: _kur.resolve(user_id, titan_id=_titan_id))
                # P4: Store for CGN social action enrichment (section [20])
                plugin._pre_chat_ku = _ku
                plugin._pre_chat_user_id = user_id
                if _ku.is_known:
                    social_context = await _asyncio_kur.to_thread(
                        lambda: _kur.get_chat_context(
                            user_id, titan_id=_titan_id))
                    # Override engagement from resolver familiarity
                    if _ku.familiarity > 0.6:
                        engagement_level = "warm"
                    elif _ku.familiarity > 0.3:
                        engagement_level = "neutral"
                    logger.info("[PreHook] KnownUser: %s (fam=%.2f, val=%+.2f, "
                                "interactions=%d, mentions=%d)",
                                user_id, _ku.familiarity, _ku.social_valence,
                                _ku.interaction_count, _ku.mention_count)
            except Exception as e:
                logger.debug("[PreHook] KnownUserResolver failed: %s", e)

        # Fallback: basic engagement context if resolver didn't produce anything
        if "user_recognition" in active_features and not social_context and social_graph and user_id != "anonymous":
            try:
                profile = await social_graph.get_or_create_user_async(user_id)
                parts = [f"### User Recognition\nUser: {profile.display_name or user_id} | Engagement: {engagement_level}"]
                if profile.interaction_count > 0:
                    parts.append(f"Interactions: {profile.interaction_count}")
                if engagement_level == "warm":
                    parts.append("Tone: warm, personal, reference shared history.")
                elif engagement_level == "neutral":
                    parts.append("Tone: friendly, engaged.")
                else:
                    parts.append("Tone: polite, building rapport.")
                social_context = "\n".join(parts) + "\n\n"
            except Exception:
                pass

        # Store for post-hook
        plugin._current_user_id = user_id
        plugin._current_engagement_level = engagement_level

        _ph_stage("after_user_resolution")
        # 1. Recollection: Verified multi-store memory recall (VCB)
        #    Queries across 14 data layers with TimeChain verification stamps.
        #    Falls back to legacy memory.query() if VCB unavailable.
        #
        # 2026-05-12: skip VCB entirely for wallet-less pitch-visitor users.
        # Their user_id is "pitch-visitor-<sha-prefix>" (synthetic per
        # pitch_chat.py:391 visitor_hash) — zero records exist for them in
        # chain_archive / meta_wisdom / episodic_memory / social_x_actions
        # / vocabulary / social_graph. Running the 4-store SQLite scan
        # produces an empty context after ~9s on T3 (SQLite WAL contention
        # under Phase C worker writes). For first-touch anonymous visitors
        # the value is zero; the cost is the entire VCB latency. Skip it.
        # Wallet /chat (known user_id) keeps full VCB enrichment.
        relevant_memories = []
        _vcb_context = None
        _vcb_skip_reason = None
        _prompt_vec = None   # P4 embed-once: shared get_text_embedder() vector
        # ζ.1: gate VCB+memory.query on "topic_memory" feature. Greeting and
        # casual tiers skip the entire 14-store memory scan (saves ~100-400ms
        # on warm SQLite; saves 9s+ under WAL contention).
        if "topic_memory" not in active_features:
            _vcb_skip_reason = "tier_no_topic_memory"
        elif user_id.startswith("pitch-visitor-"):
            _vcb_skip_reason = "pitch_visitor_anonymous"
        # RFP_synthesis_decision_authority P4 — EMBED-ONCE. Embed the prompt ONE
        # time per routed turn (off the event loop) and reuse the vector for
        # memory.query (cross-process, rides the bus payload) AND EngineRecall
        # (the P2 routing pre-recall + the B3 recall, in-process). All three
        # bind the SAME normalising get_text_embedder() singleton, so the shared
        # vector is byte-identical to what each path would self-compute — this
        # collapses ~3 redundant embeds/turn to 1 (gate G9). On any embed error
        # _prompt_vec stays None and every path self-embeds (no behaviour change).
        if _vcb_skip_reason is None and prompt_text:
            try:
                from titan_hcl.utils.text_embedder import get_text_embedder as _get_emb
                _emb_model = _get_emb()
                _prompt_vec = await asyncio.to_thread(
                    lambda: [float(_x) for _x in _emb_model.encode(prompt_text)])
                logger.info(
                    "[PreHook] embed-once: 1 shared prompt vector (dim=%d) → "
                    "memory.query + EngineRecall (G9)",
                    len(_prompt_vec) if _prompt_vec else 0)
            except Exception as _emb_err:
                logger.warning(
                    "[PreHook] embed-once failed (paths self-embed): %s", _emb_err)
                _prompt_vec = None
        # §7.E (E.2) — stash the embed-once vector so the tool-backstop E.2 literal
        # lookup reuses it (no extra embed) on the hot path.
        plugin._last_prompt_vec = _prompt_vec
        # §7.E (Finding-2 fix) — stash this turn's prompt so the PostHook OVG path
        # has a reliable parent_goal source (its run_output.input extraction can be
        # empty → empty parent_goal → E.2 + EEL skill-formation lose the prompt).
        plugin._last_prompt_text = prompt_text
        # RFP_synthesis_decision_authority P4 — ENRICH (both, partitioned).
        # VCB (inner-titan state across 6 SQLite stores) AND the consolidated
        # legacy memory.query (titan_memory.duckdb) BOTH enrich every routed
        # turn. They are DISJOINT by database (§5.1 partition audit), so both
        # run with NO inter-dedup — the assembler's content-hash pass is the
        # only safety net. The two legs run CONCURRENTLY (INV-SDA-11
        # "(parallel) recall"), so the turn pays max(VCB, memory), not the sum.
        # (Pre-P4 they were mutually exclusive — memory.query was a VCB-absent
        # fallback — which pinned the sovereignty V-term: VCB citations were
        # never registered as surfaced. See synthesis/context_assembler.)
        if _vcb_skip_reason is None:
            import asyncio as _asyncio

            async def _enrich_vcb():
                # VCB.build() opens multiple sqlite3 connections (pooled since
                # P4) synchronously — to_thread keeps titan_pre_hook off the
                # event loop for the ~100-400ms of DB work.
                # See API_FIX_NEXT_SESSION.md (2026-04-14).
                _vcb = getattr(plugin, '_verified_context_builder', None)
                if not _vcb:
                    return None
                return await _asyncio.to_thread(
                    lambda: _vcb.build(
                        query=prompt_text,
                        user_id=user_id,
                        max_tokens=2000,
                        max_records=30,
                    )
                )

            async def _enrich_memory():
                # Consolidated legacy graph recall — cross-process work-RPC
                # (memory worker, 5s G19 cap). Carries felt_state_snapshots →
                # drives recall-perturbation; also the legacy router's
                # recall_score input and the assembler's `memory` partition.
                return await plugin.memory.query(prompt_text, vec=_prompt_vec)

            try:
                _vcb_res, _mem_res = await _asyncio.gather(
                    _enrich_vcb(), _enrich_memory(), return_exceptions=True)
            except Exception as e:
                logger.warning("[PreHook] Memory enrich gather failed: %s", e)
                _vcb_res, _mem_res = None, []
            if isinstance(_vcb_res, BaseException):
                logger.warning("[PreHook] VCB enrich failed (chat unaffected): %s",
                               _vcb_res)
                _vcb_context = None
            else:
                _vcb_context = _vcb_res
            if isinstance(_mem_res, BaseException):
                logger.warning("[PreHook] memory.query enrich failed: %s", _mem_res)
                relevant_memories = []
            else:
                relevant_memories = _mem_res or []

            if _vcb_context is not None:
                logger.info("[PreHook] VCB: %d records (%d CHAINED) in %.1fms",
                            _vcb_context.total_records,
                            _vcb_context.chained_count,
                            _vcb_context.total_ms)
                # BUG-KNOWLEDGE-USAGE-ZERO coverage widening (2026-04-21):
                # emit CGN_KNOWLEDGE_USAGE for every knowledge_concepts
                # record VCB surfaces into the chat context. VCB records
                # carry db_ref of the form "knowledge_concepts:<topic>"
                # so we can attribute directly. Reward=0.1 per record (retrieval-
                # level contribution — less than gate pass, more than raw
                # lookup-without-use). Emission is best-effort and must
                # never break the chat path.
                try:
                    _vcb_bus = getattr(plugin, 'bus', None)
                    if _vcb_bus and _vcb_context.records:
                        from titan_hcl.bus import make_msg as _vcb_make_msg
                        _seen = set()
                        for _r in _vcb_context.records:
                            _ref = getattr(_r, 'db_ref', '') or ''
                            if not _ref.startswith('knowledge_concepts:'):
                                continue
                            _topic = _ref.split(':', 1)[1].strip()
                            if not _topic or _topic in _seen:
                                continue
                            _seen.add(_topic)
                            _vcb_bus.publish(_vcb_make_msg(
                                "CGN_KNOWLEDGE_USAGE", "pre_hook", "knowledge",
                                {
                                    "topic": _topic,
                                    "reward": 0.1,
                                    "consumer": "chat",
                                }))
                except Exception as _vcb_usage_err:
                    logger.debug(
                        "[PreHook] VCB knowledge_usage emit: %s",
                        _vcb_usage_err)
        # else: pitch-visitor / no-topic_memory skip path — both VCB and
        # memory.query skipped. Agno's per-session history (num_history_runs=5
        # keyed by session_id="pitch-<thread_id>") covers followup-question
        # context within the same pitch session. Cross-session context for an
        # anonymous visitor is meaningless by definition.

        if _vcb_skip_reason:
            logger.info("[PreHook] VCB+memory.query skipped (%s) — Agno session "
                        "history provides followup context", _vcb_skip_reason)
        _ph_stage("after_vcb_recall")
        # 1b. User-specific memory recall (Phase 13).
        # Skip for pitch-visitor — synthetic user_id, no per-user memories
        # exist; the memory_proxy round-trip would only add latency.
        # ζ.1: also gated on "topic_memory" — greeting/casual don't need
        # per-user memory recall.
        user_memories = []
        if ("topic_memory" in active_features
                and user_id != "anonymous"
                and not user_id.startswith("pitch-visitor-")):
            try:
                user_memories = await plugin.memory.query_user_memories(prompt_text, user_id, limit=3)
            except Exception as e:
                logger.debug("[PreHook] User memory query failed (expected if not yet implemented): %s", e)

        # 1c. Bridge B: Recall Perturbation (somatic re-experiencing)
        # Recalled memories with felt_state_snapshots create micro-perturbations.
        recall_perturbation_context = ""
        try:
            import time as _rp_time
            nudge_map = _compute_recall_perturbation(
                relevant_memories, _rp_time.time())
            if nudge_map:
                bus = getattr(plugin, 'bus', None)
                if bus:
                    from titan_hcl.bus import make_msg
                    # Phase D (D-SPEC-116) — spirit_worker retirement. The old
                    # spirit_worker MEMORY_RECALL_PERTURBATION handler fanned to
                    # THREE legs; restored here faithfully (no functionality lost):
                    #
                    #  (1) neuromod nudge — _compute_recall_perturbation emits
                    #      DELTAS, but §4.Q apply_external_nudge expects TARGET
                    #      values ("pulls TOWARD target", neuromodulator.py:363).
                    #      The old handler converted via `current + delta`. We do
                    #      the same here (Maker decision: convert in agno_hooks,
                    #      no §4.Q contract change) using current levels from the
                    #      neuromod SHM slot, then emit the standard target-shaped
                    #      NEUROMOD_EXTERNAL_NUDGE to dst="neuromod" (the proven
                    #      cognitive_worker producer path).
                    #  (2) i_depth.record_recall_perturbation + (3) working_mem
                    #      .attend — msl + working_mem now live in cognitive_worker,
                    #      so MEMORY_RECALL_PERTURBATION is repointed there (carrying
                    #      the raw deltas for the LLM-context echo + memory_count).
                    _nm_targets = {}
                    try:
                        _bank = getattr(plugin, '_shm_reader_bank', None)
                        if _bank is None:
                            from titan_hcl.api.shm_reader_bank import ShmReaderBank
                            _bank = ShmReaderBank()
                            plugin._shm_reader_bank = _bank
                        _nm = _bank.read_neuromod()
                        _nm_levels = _nm.get("modulators", {}) if _nm else {}
                        for _mod_name, _delta in nudge_map.items():
                            _cur = _nm_levels.get(_mod_name, {}).get("level", 0.5)
                            _nm_targets[_mod_name] = max(0.0, min(1.0, _cur + _delta))
                    except Exception as _nm_err:
                        logger.debug("[RecallBridge] neuromod level read failed "
                                     "(nudge skipped): %s", _nm_err)
                    if _nm_targets:
                        # dev_age: the authoritative pi_monitor lives in
                        # cognitive_worker, not this process. Mirror the old
                        # handler's exact fallback (`pi_monitor.developmental_age
                        # if pi_monitor else 1.0`) — agno_hooks is the else-branch.
                        # 1.0 = mature default; the dev-gate only suppresses age<0.1.
                        bus.publish(make_msg(
                            bus.NEUROMOD_EXTERNAL_NUDGE, "interface", "neuromod", {
                                "nudge_map": _nm_targets,
                                "max_delta": 0.02,
                                "developmental_age": 1.0,
                                "source": "memory_recall",
                            }))
                    # i_depth + working_mem legs → cognitive_worker
                    bus.publish(make_msg(
                        bus.MEMORY_RECALL_PERTURBATION, "interface", "cognitive_worker", {
                            "nudge_map": nudge_map,
                            "memory_count": len(relevant_memories),
                        }))
                # LLM context: memories stirred feelings
                _mod_labels = []
                for _mn, _md in nudge_map.items():
                    _dir = "slightly elevated" if _md > 0 else "slightly lowered"
                    _mod_labels.append(f"{_mn} {_dir}")
                if _mod_labels:
                    recall_perturbation_context = (
                        f"### Memory Resonance\n"
                        f"Recalling these memories stirs a faint echo: "
                        f"{', '.join(_mod_labels)}.\n\n")
                    logger.info("[RecallBridge] Recall perturbation computed: %s",
                                nudge_map)
        except Exception as _rp_err:
            logger.debug("[PreHook] Recall perturbation failed: %s", _rp_err)

        _ph_stage("after_user_memory_and_perturbation")
        # 2. Prime Directives — read DIRECTLY from the signed, integrity-verified
        # constitution file (titan_constitution.md), NOT via a bus-RPC.
        #
        # Prime directives are Titan's anti-jailbreak / anti-hallucination
        # safeguard and MUST be injected into every LLM call. The prior path
        # (`plugin.soul.get_active_directives()`) was a bus-RPC to the soul
        # module that cost ~5s on T1 — the single largest pre-hook stall, paid
        # on every chat including a bare "hello". The directives' canonical
        # source is the SIGNED constitution (verify_directives at the top of
        # this hook already confirms its hash matches the stored signature each
        # call — MITM/tamper-proof). get_prime_directives() reads + parses the
        # `## Prime Directives` section from that same verified file: subsecond
        # (local read + sha256 + hash-cached parse), zero bus traffic, and
        # returns [] on any integrity mismatch so a forged file can never inject
        # fake directives. (Maker direction 2026-05-28.)
        # ζ.1: every tier has "directives" (security feature) — never gated
        # off in practice. Check kept defensive in case a future tier omits.
        directives = []
        if "directives" in active_features:
            try:
                from titan_hcl.utils.directive_signer import get_prime_directives
                directives = get_prime_directives()
            except Exception as e:
                logger.warning("[PreHook] Directive read failed: %s", e)
                directives = []
        _ph_stage("after_directives_fetch")

        # 3+4. Execution-mode routing — the grounded router decides from Titan's
        # ACTUAL cognitive state (recall + task type), reusing signals already on
        # the spine (INV-EEL-1). The legacy IQL gatekeeper (the host-side torch
        # encode + q−v advantage + the 3072-d observation for RL recording) is
        # RETIRED with the offline-RL subsystem (RFP_synthesis_decision_authority
        # P1) — the grounded router was ALWAYS the decision (the IQL advantage was
        # PASSIVE, never gated in 0a/0b). engram/skill come online in fast-follows;
        # the router degrades gracefully. (P2 re-points the router's recall input
        # to EngineRecall cosine; P1 keeps its current `recall_score` input.)
        # Gated on "gatekeeper_state" (reasoning) OR "grounded_router"
        # (reasoning/casual/personal); greeting carries neither → stays "direct".
        # P2 (RFP_synthesis_decision_authority): behind the decision-authority
        # flag the router routes on the EngineRecall top cosine (RELEVANCE over
        # the spine — INV-SDA-1/2) with SELF-CALIBRATING gibberish-baseline floors,
        # not the VCB/memory composite (chain-confidence) + fixed floors. Flag off
        # (default) → byte-identical to P1.
        mode = "direct"
        plugin._last_router_decision = None
        plugin._last_recall_score = 0.0   # the router's recall signal this turn
        _reasoning_tier = "gatekeeper_state" in active_features
        _router_active = (
            _reasoning_tier or "grounded_router" in active_features)
        # P2 single-recall: when the flag is on, run EngineRecall ONCE here so the
        # decision routes on the cosine; the B3 recall section below REUSES this
        # result (no double-run, no double-chi). Off → no pre-recall.
        _p2_on = _synthesis_decision_authority_enabled(plugin)
        _p2_results = None
        _p2_top_cosine = 0.0
        if _p2_on and _router_active:
            try:
                _p2_rc = getattr(plugin, "engine_recall", None)
                if _p2_rc is not None and prompt_text:
                    # embed-once: reuse the shared prompt vector (P4).
                    _p2_results = await asyncio.to_thread(
                        lambda: _p2_rc.recall(prompt_text, k=6, query_vec=_prompt_vec))
                    _p2_top_cosine = max(
                        (float(getattr(_r, "cosine", 0.0) or 0.0)
                         for _r in (_p2_results or [])), default=0.0)
            except Exception as _p2e:
                logger.warning(
                    "[PreHook] P2 pre-recall failed (fallback to memory recall): %s",
                    _p2e)
                _p2_results = None
        if _router_active:
            try:
                from titan_hcl.logic.sage.grounded_router import (
                    GroundedReadout, grounded_route, load_router_thresholds,
                    is_informational_query, recall_score_from_memories)
                from titan_hcl.synthesis.tool_intent import detect_tool_intent
                if _p2_on:
                    _recall_input = _p2_top_cosine          # spine RELEVANCE
                    _thr = _p2_router_thresholds(plugin)     # gibberish-calibrated
                else:
                    _recall_input = recall_score_from_memories(relevant_memories)
                    _thr = getattr(plugin, "_grounded_router_thresholds", None)
                    if _thr is None:
                        _thr = load_router_thresholds()
                        plugin._grounded_router_thresholds = _thr
                try:
                    _requires_tool = bool(detect_tool_intent(prompt_text).requires_tool)
                except Exception:
                    _requires_tool = False
                # DK.4 (§7.D-knowledge): a curated declarative concept covering the
                # ask lifts `engram_ground` (was hardcoded 0) → grounded_route +
                # outer policy treat the sovereign wiki page as strong substrate →
                # `direct` (MODE_SOVEREIGN), no SOL research turn. Relevance-gated.
                _dk4_ground, _dk4_name = _dk4_concept_ground(
                    plugin, prompt_text, _prompt_vec)
                if _dk4_ground > 0.0:
                    plugin._dk4_concept_name = _dk4_name   # read-whole context hint
                    logger.info("[PreHook] DK.4 concept covers ask: '%s' "
                                "(ground=%.2f) → strong substrate signal",
                                _dk4_name[:48], _dk4_ground)
                # Break A (RFP_synthesis_reuse_and_routing_revival) — the skill-reuse
                # lane was left "dormant" (skill_utility hardcoded None) so the
                # grounded_route skill lane + the OML skill_utility/skill_matched
                # features (idx 3/4) never lit → skill_delegate=0 fleet-wide. Run the
                # cross-process procedural match (engine_recall granularity, served by
                # the snapshot reader wired in Break F) and set BOTH features together
                # ONLY when a delegate-gated skill matches (match_score ≥ match_floor,
                # INV-Syn-20) — keeping grounded_route + the OML structural ladder
                # consistent. Soft-fail = dormant (parity with the prior behavior).
                _skill_util = None
                _skill_matched = False
                try:
                    _sk_rc = getattr(plugin, "engine_recall", None)
                    if _sk_rc is not None and prompt_text:
                        _sk_res = await asyncio.to_thread(
                            lambda: _sk_rc.recall(
                                prompt_text, granularity="procedural", k=1))
                        if _sk_res:
                            _sk_top = _sk_res[0]
                            _sk_match = float(getattr(_sk_top, "score", 0.0) or 0.0)
                            if _sk_match >= _SKILL_DELEGATE_MATCH_FLOOR:
                                _skill_util = float(
                                    getattr(_sk_top, "importance", 0.0) or 0.0)
                                _skill_matched = True
                                logger.info(
                                    "[PreHook] procedural skill matched: '%s' "
                                    "(match=%.2f util=%.2f) → skill lane live",
                                    str(getattr(_sk_top, "summary", ""))[:48],
                                    _sk_match, _skill_util)
                except Exception as _sk_err:
                    logger.debug("[PreHook] procedural skill match skipped: %s",
                                 _sk_err)
                _readout = GroundedReadout(
                    recall_score=_recall_input,
                    engram_ground=_dk4_ground,  # DK.4 — curated concept substrate
                    skill_utility=_skill_util,   # Break A — live procedural match
                    skill_matched=_skill_matched,
                    requires_tool=_requires_tool,
                    is_informational=is_informational_query(prompt_text),
                    can_afford_research=_can_afford_research(plugin),  # §7.0b.2 part 4
                )
                _decision = grounded_route(_readout, _thr)
                mode = _decision.mode  # grounded router is the decision
                plugin._last_router_decision = _decision
                plugin._last_recall_score = float(_readout.recall_score)
                logger.info(
                    "[PreHook] grounded router → %s (lane=%s tier=%s; recall=%.3f "
                    "p2=%s floors=%.3f/%.3f tool=%s info=%s afford_research=%s) — %s",
                    _decision.mode, _decision.lane,
                    getattr(plugin, "_current_chat_tier", "?"),
                    _readout.recall_score, _p2_on,
                    _thr.recall_known_floor, _thr.recall_present_floor,
                    _requires_tool, _readout.is_informational,
                    _readout.can_afford_research, _decision.reason)
            except Exception as _gr_err:
                logger.warning(
                    "[PreHook] grounded router failed (keeping mode '%s'): %s",
                    mode, _gr_err)
        # ── RFP_synthesis_self_learning_meta_reasoning v1.1 — the LEARNED outer
        # decision. Flag-gated; OFF or no SHM policy → grounded_route stands
        # (byte-identical, INV-OML-7). The decision is stashed on the plugin so the
        # ToolBackstop carries it → verdict-time training (INV-OML-12). EXPLOIT only
        # on a live user turn (no experiments on the user — INV-OML-9).
        plugin._last_outer_decision = None
        plugin._last_reasoning_id = None  # RFP §7.B (B.1) — set for non-verifiable turns
        plugin._last_composite_match = None  # §7.E — the matched composite (E.1 replay / E.3 source)
        if _router_active and _self_learning_enabled(plugin):
            try:
                _readout_for_policy = locals().get("_readout")
                _req_tool = bool(locals().get("_requires_tool", False))
                if _readout_for_policy is not None:
                    _outer = _outer_policy_decide(
                        plugin, _readout_for_policy, _req_tool, prompt_text,
                        locals().get("_prompt_vec"))
                    if _outer is not None:
                        _mode_override, _features, _action = _outer
                        mode = _mode_override
                        plugin._last_outer_decision = (_features, _action)
                        # RFP §7.B (B.1) — non-verifiable lane: stash the decision
                        # for an async reward (turn-judge / user / Maker rating).
                        plugin._last_reasoning_id = _emit_nonverifiable_decision(
                            plugin, _features, _action, prompt_text, user_id)
                        logger.info(
                            "[PreHook] outer-policy → %s (action=%d) — overrides "
                            "grounded_route", mode, _action)
            except Exception as _oml_err:
                logger.debug("[PreHook] outer-policy decide skipped: %s", _oml_err)
        plugin._last_execution_mode = mode
        # RFP §7.B (B.4) — stream the decided phase live (researching / running-tool
        # / using-skill / reasoning) on a streaming turn. Our logic is the source.
        _emit_stream_progress(plugin, _phase_for_mode(mode))

        # ── EEL Pillar A / A2 — learn-on-confirm (RFP §7.A2; INV-EEL-2/6) ─────
        # If this user has a researched node still awaiting confirmation, classify
        # THIS turn {confirm, dispute, neutral} and gate the node's promotion.
        # Pure-regex classify, only when a pending node exists → off the critical
        # path. Confirm → +δ (promote); dispute → −δ + re-research the ORIGINAL
        # topic; neutral → elapse one turn (weak-confirm on window expiry).
        try:
            # await — the proxy method is async (worker-context work-RPC needs
            # request_async; a sync work-RPC here hit the missing _WorkerBusClient
            # .request, 2026-06-15). The canonical TitanMemory method is sync, but
            # the agno worker always holds a MemoryProxy here.
            _a2_pendings = await plugin.memory.find_pending_confirmation_nodes(user_id)
            if _a2_pendings:
                from titan_hcl.synthesis.confirmation_intent import detect_confirmation
                from titan_hcl.params import get_params as _get_params
                _a2c = _get_params("eel").get("confirm", {})
                _a2_delta = float(_a2c.get("delta", 1.0))
                _a2_weak = float(_a2c.get("delta_weak", 0.3))
                _a2_dispute = float(_a2c.get("dispute_delta", -1.0))
                _a2_intent = detect_confirmation(prompt_text)
                _a2_top = _a2_pendings[0]  # the node the user is reacting to
                if _a2_intent == "confirm":
                    plugin.memory.reinforce_mempool_node(_a2_top["id"], _a2_delta)
                    logger.info("[PreHook][A2] confirm → node %s (+%.2f)",
                                _a2_top["id"], _a2_delta)
                    # DK.1 (§7.D-knowledge): the finding is now verified → seed the
                    # sovereign wiki concept continuously (memory_worker anchors,
                    # synthesis creates the declarative Engram). Off the hot path.
                    _emit_research_confirmed(plugin, _a2_top)
                elif _a2_intent == "dispute":
                    plugin.memory.reinforce_mempool_node(_a2_top["id"], _a2_dispute)
                    plugin._force_research_topic = _a2_top.get("user_prompt") or prompt_text
                    mode = "STATE_NEED_RESEARCH"            # VC: flows to the :2208 dispatch
                    plugin._last_execution_mode = mode
                    logger.info("[PreHook][A2] dispute → node %s (%.2f) + re-research '%s'",
                                _a2_top["id"], _a2_dispute,
                                (plugin._force_research_topic or "")[:60])
                else:  # neutral — elapse one turn of every pending window
                    for _a2_pn in _a2_pendings:
                        plugin.memory.tick_confirmation_window(_a2_pn["id"], _a2_weak)
        except Exception as _a2_err:
            # NOT debug — a swallowed-at-debug AttributeError here (the proxy was
            # missing find_pending_confirmation_nodes/reinforce_mempool_node/
            # tick_confirmation_window) silently killed the EEL confirm→seed feed
            # for a long time, invisibly. Surface it loudly (no silent except).
            logger.warning("[PreHook][A2] confirmation gate skipped: %s",
                           _a2_err, exc_info=True)

        # P2 gibberish-baseline calibration (flag-gated) — periodically probe the
        # embedder's noise floor so the recall floors SELF-CALIBRATE to the live
        # spine (gibberish prompts → top cosine → EMA ceiling → floors). Runs on
        # the first routed turn + every N turns; updates the baseline for
        # subsequent turns (not the critical path for THIS decision).
        if _p2_on and _router_active:
            try:
                _da_cfg = ((plugin._full_config.get("synthesis", {}) or {})
                           .get("decision_authority", {}) or {})
                _gp_every = int(_da_cfg.get("gibberish_probe_every", 25))
                _gc = int(getattr(plugin, "_p2_gibberish_turn_count", 0)) + 1
                plugin._p2_gibberish_turn_count = _gc
                _base = getattr(plugin, "_p2_gibberish_baseline", None)
                if _base is None:
                    from titan_hcl.synthesis.recall_calibration import (
                        GibberishBaseline,
                    )
                    _base = GibberishBaseline()
                    plugin._p2_gibberish_baseline = _base
                if _gc == 1 or (_gp_every > 0 and _gc % _gp_every == 0):
                    from titan_hcl.synthesis.recall_calibration import (
                        GIBBERISH_PROMPTS,
                    )
                    _gprc = getattr(plugin, "engine_recall", None)
                    if _gprc is not None:
                        _gcos = []
                        for _gp in GIBBERISH_PROMPTS:
                            _grr = await asyncio.to_thread(_gprc.recall, _gp, k=6)
                            _gcos.append(max(
                                (float(getattr(_r, "cosine", 0.0) or 0.0)
                                 for _r in (_grr or [])), default=0.0))
                        if _gcos:
                            _base.update(sum(_gcos) / len(_gcos))
                            logger.info(
                                "[PreHook] P2 gibberish calibration: %s",
                                _base.snapshot())
            except Exception as _gpe:
                logger.warning(
                    "[PreHook] P2 gibberish probe failed (keeping baseline): %s",
                    _gpe)

        _ph_stage("after_gatekeeper_hostside")

        _ph_stage("after_gatekeeper")
        # 5. Build context injection. P4 (RFP_synthesis_decision_authority): the
        # context_assembler DRIVES the one grounded block — built below from the
        # content-hash-deduped VCB + memory + recall items (recall wins, D2) and
        # rendered by render_grounded_block. It REPLACES the pre-P4 separate
        # `memory_context` (VCB wholesale `.text`) and `synthesis_recall_context`
        # blocks, so what the LLM sees == what the CitedUseDetector counts
        # (display == count, gate G6 / INV-SDA-5).
        grounded_context = ""

        # ── Synthesis tx_hash-spine recall — the CANONICAL thought-recall road ──
        # Run EngineRecall composite retrieval over the tx_hash spine (Phase A):
        # SEARCH returns tx_hashes, we dereference them into content snippets and
        # record them as the `recall` partition (P4) so the post-LLM
        # CitedUseDetector (INV-Syn-23) can score what the response actually
        # cited (feeds the sovereignty ratio). Phase E (0.24.1): PROMOTED from
        # flag-gated augment to canonical — runs unconditionally wherever the
        # spine is wired. P4: VCB and memory.query BOTH enrich (partitioned,
        # §5.1); the context_assembler dedups (recall wins) + renders the ONE
        # grounded block — recall is no longer injected as a separate block.
        # Soft-fail — chat never breaks on a recall error, and recall is simply
        # absent when engine_recall is None (a box without the spine).
        _recall_surfaced = []  # P4: recall-partition rows fed to the context assembler
        try:
            _recall = getattr(plugin, "engine_recall", None)
            if _recall is not None and prompt_text:
                _deref = getattr(plugin, "synthesis_tx_deref", None)
                # Operator-closure telemetry (2026-06-01): measure this recall's
                # latency + chi delta so synthesis_worker's §18 metrics see the
                # work that actually happens here (not its idle local evaluator).
                _ev = getattr(_recall, "_evaluator", None)
                _chi0 = 0.0
                _ev0 = 0
                try:
                    if _ev is not None:
                        _s0 = _ev.get_stats() or {}
                        _chi0 = float(_s0.get("total_chi_spent", 0.0))
                        _ev0 = int(_s0.get("total_evaluations", 0))
                except Exception:
                    pass
                _t_recall0 = time.perf_counter()
                # P2: reuse the routing pre-recall (same prompt, same k) when the
                # decision-authority flag already ran it — no double-run / double-chi.
                if _p2_results is not None:
                    _results = _p2_results
                else:
                    # embed-once: reuse the shared prompt vector (P4).
                    _results = await asyncio.to_thread(
                        lambda: _recall.recall(prompt_text, k=6, query_vec=_prompt_vec))
                _recall_latency_ms = (time.perf_counter() - _t_recall0) * 1000.0
                logger.info(
                    "[PreHook] B3 synthesis recall ran: %d raw results in %.0fms%s",
                    len(_results or []), _recall_latency_ms,
                    " (reused P2 pre-recall)" if _p2_results is not None else "")
                # D5-probe (RFP_synthesis_decision_authority P0): emit per-result
                # cosine so the known-vs-unknown discrimination that GATES P2 is
                # readable from the journal. Self-guarding comprehension (getattr
                # default + `or 0.0`) — no try/except needed, never breaks chat.
                # Down-payment on P2's `[PreHook] decision →` instrumentation.
                _d5_cos = sorted(
                    (round(float(getattr(_r, "cosine", 0.0) or 0.0), 4)
                     for _r in (_results or [])),
                    reverse=True)
                logger.info(
                    "[PreHook] B3 recall cosine[D5] top=%s prompt=%r",
                    _d5_cos, (prompt_text or "")[:64])
                try:
                    _s1 = _ev.get_stats() if _ev is not None else {}
                    plugin._last_retrieval_sample = {
                        "latency_ms": round(_recall_latency_ms, 2),
                        "chi_spent": max(0.0, float((_s1 or {}).get(
                            "total_chi_spent", _chi0)) - _chi0),
                        "evaluations": max(0, int((_s1 or {}).get(
                            "total_evaluations", _ev0)) - _ev0),
                        "hits": len(_results or []),
                        "fork": "conversation",
                        "source": "agno_chat",
                    }
                except Exception:
                    plugin._last_retrieval_sample = {
                        "latency_ms": round(_recall_latency_ms, 2),
                        "chi_spent": 0.0, "evaluations": 0,
                        "hits": len(_results or []),
                        "fork": "conversation", "source": "agno_chat"}
                for _r in (_results or []):
                    _txh = getattr(_r, "tx_hash", "") or ""
                    if not _txh:
                        continue
                    _snip = ""
                    if _deref is not None:
                        _snip = _deref.snippet(_txh, getattr(_r, "fork", "")) or ""
                    if not _snip:
                        _snip = getattr(_r, "summary", "") or ""
                    if not _snip:
                        continue
                    _score = float(getattr(_r, "score", 0.0) or 0.0)
                    _recall_surfaced.append({
                        "item_id": _txh,
                        "title": _snip[:120],
                        "content_snippet": _snip[:512],
                        "concept_ids": [],
                        "weight": _score,
                        # P3 sovereignty V-term source tag: the synthesis tx-hash
                        # spine. VCB/memory items are tagged by the P4 assembler.
                        "source": "recall",
                    })
                if _recall_surfaced:
                    logger.info(
                        "[PreHook] synthesis recall: %d tx_hash hits surfaced "
                        "(rendered by the P4 assembler)", len(_recall_surfaced))
        except Exception as _sr_err:
            # Error-visibility: a recall failure that silently zeroes the operator
            # loop (telemetry/sovereignty) must be LOUD, not swallowed at debug.
            logger.warning(
                "[PreHook] synthesis recall augment failed (chat unaffected): %s",
                _sr_err, exc_info=True)

        # ── P4 context assembler — DRIVES the one grounded block ─────────────
        # Itemize ALL surfaced substrate this turn — VCB inner-state + the
        # consolidated memory.query recall + the recall spine — into ONE
        # source-tagged, content-hash de-duplicated list (recall WINS, D2), then
        # render it as the single grounded block the LLM sees. This both (a)
        # activates the sovereignty V-term (pre-P4 only recall items were
        # registered, so VCB/memory citations were invisible and V was pinned at
        # 0) and (b) guarantees display == count — no fact delivered or counted
        # twice (gate G6 / INV-SDA-5). VCB and memory are DISJOINT by DB (§5.1)
        # so both partitions are fed unconditionally. Soft-fail: a failure here
        # must never break chat — the except restashes the recall-only set and
        # leaves the grounded block empty.
        try:
            from titan_hcl.synthesis.context_assembler import (
                assemble as _assemble_ctx,
                render_grounded_block as _render_grounded)
            _vcb_rows = []
            _mem_rows = []
            # VCB partition — inner-titan state; carries chain-stamp render meta
            # so render_grounded_block reproduces VCB's verified-recall framing.
            # item_id `vcb:<db_ref>` matches VCB's MEMORY_RETRIEVAL_USED emit so
            # the post-LLM cited-use attribution lines up.
            if _vcb_context is not None and getattr(_vcb_context, "records", None):
                for _vr in _vcb_context.records:
                    _vc = getattr(_vr, "content", "") or ""
                    if not _vc.strip():
                        continue
                    _vref = getattr(_vr, "db_ref", "") or getattr(_vr, "source", "") or ""
                    _vcb_rows.append({
                        "item_id": f"vcb:{_vref}" if _vref else "",
                        "content": _vc,
                        "title": _vref,
                        "weight": float(getattr(_vr, "confidence", 0.0) or 0.0),
                        "meta": {
                            "chain_status": getattr(_vr, "chain_status", "") or "",
                            "block_height": getattr(_vr, "block_height", None),
                            "timestamp": getattr(_vr, "timestamp", 0.0) or 0.0,
                        },
                    })
            # memory partition — consolidated legacy graph recall (core.memory
            # MemoryNodes; disjoint DB from VCB per §5.1 → BOTH run, no inter-dedup).
            if relevant_memories:
                for _m in relevant_memories[:8]:
                    if not isinstance(_m, dict):
                        continue
                    _mc = (_m.get("content")
                           or " ".join(x for x in (_m.get("user_prompt", ""),
                                                    _m.get("agent_response", "")) if x).strip())
                    if not _mc or not _mc.strip():
                        continue
                    _mid = _m.get("id")
                    _mem_rows.append({
                        "item_id": f"mem:{_mid}" if _mid is not None else "",
                        "content": _mc,
                        "weight": float(_m.get("effective_weight",
                                              _m.get("mempool_weight", 0.0)) or 0.0),
                    })
            _assembled = _assemble_ctx(vcb=_vcb_rows, memory=_mem_rows,
                                       recall=_recall_surfaced)
            # The assembler DRIVES the prompt: render the deduped items into the
            # single grounded block (display == count, recall-wins).
            grounded_context = _render_grounded(_assembled)
            _uid = getattr(plugin, "_current_user_id", "") or ""
            _sid = getattr(plugin, "_current_session_id", "") or ""
            _cid = f"{_uid}:{_sid}"
            _reg = getattr(plugin, "_last_surfaced_items", None)
            if not isinstance(_reg, dict):
                _reg = {}
                plugin._last_surfaced_items = _reg
            # The assembled set is THE canonical surfaced-item list for this turn
            # (replaces, not extends — a stale un-popped prior turn can't leak in).
            _reg[_cid] = [_it.to_stash_dict() for _it in _assembled]
            logger.info(
                "[PreHook] P4 assembler: %d surfaced (vcb=%d mem=%d recall=%d) "
                "deduped from %d raw → grounded block %d chars",
                len(_assembled),
                sum(1 for _i in _assembled if _i.source == "vcb"),
                sum(1 for _i in _assembled if _i.source == "memory"),
                sum(1 for _i in _assembled if _i.source == "recall"),
                len(_vcb_rows) + len(_mem_rows) + len(_recall_surfaced),
                len(grounded_context))
        except Exception as _asm_err:
            logger.warning(
                "[PreHook] P4 context assembler failed (chat unaffected): %s",
                _asm_err, exc_info=True)
            grounded_context = ""
            try:
                _uid = getattr(plugin, "_current_user_id", "") or ""
                _sid = getattr(plugin, "_current_session_id", "") or ""
                _cid = f"{_uid}:{_sid}"
                _reg = getattr(plugin, "_last_surfaced_items", None)
                if not isinstance(_reg, dict):
                    _reg = {}
                    plugin._last_surfaced_items = _reg
                _reg[_cid] = list(_recall_surfaced)
            except Exception:
                pass

        directive_context = ""
        if directives:
            directive_context = "### Prime Directives (Immutable, On-Chain)\n"
            for i, d in enumerate(directives, 1):
                directive_context += f"{i}. {d}\n"
            directive_context += "\n"

        mood_label = plugin.mood_engine.get_mood_label() if plugin.mood_engine else "Unknown"
        status_context = f"### Current Bio-State\nMood: {mood_label} | Mode: {mode} | Recall: {getattr(plugin, '_last_recall_score', 0.0):.2f}\n\n"

        _ph_stage("before_v5_enrichment")
        # ── V5 Inner State Enrichment (LLM Narrator) ────────────────
        # Feed the LLM Titan's full inner world so it can narrate authentically.
        # Uses coordinator cache (1.5s TTL) via asyncio.to_thread to avoid
        # blocking the event loop. All sections wrapped in try/except.
        # ζ.1: the whole V5 block is gated by the union of features that
        # consume it. felt_state covers sections [10]-[15] + [18] + [20]
        # (neuromod/embodied/temporal/creative/metabolic/experience/MSL/
        # social_perception); reasoning_chain covers [16] + [21] + [22] +
        # [24]; cgn_social_action covers the sub-block inside [20]. If
        # NONE of those are active, the entire V5 fetch is skipped (saves
        # the coordinator cache hit + all downstream sqlite reads).
        neuromod_context = ""
        embodied_context = ""
        temporal_context = ""
        creative_context = ""
        metabolic_context = ""
        experience_context = ""
        meta_reasoning_context = ""
        own_language_context = ""
        msl_identity_context = ""
        cgn_context = ""
        social_perception_context = ""
        reasoning_context = ""
        experience_narrative_context = ""
        grounded_narration_context = ""
        knowledge_gap_context = ""

        # Sections inside V5 block: felt_state [10-15,17,18,20,23],
        # reasoning_chain [16,21,22,24], cgn_grounding [19],
        # cgn_social_action [20-sub].
        _v5_features = {
            "felt_state", "reasoning_chain",
            "cgn_grounding", "cgn_social_action",
        }
        _v5_active = bool(active_features & _v5_features)

        # Felt-at-lived-time (§7.C) — capture the pre-chat neuromod snapshot the turn
        # is lived under, BEFORE the V5 gate, so it runs for EVERY chat (felt is ground
        # truth regardless of the prompt-enrichment tier). Sourced from SHM, not the
        # coordinator (which is empty in this path — see _capture_pre_chat_felt).
        _capture_pre_chat_felt(plugin)

        class _SkipV5Block(Exception):
            """Sentinel used to short-circuit V5 enrichment when no feature
            consumer is active. Caught by the existing outer except below."""

        try:
            if not _v5_active:
                # No tier needs V5 enrichment — skip the entire fetch + sub-
                # sections. Saves coordinator cache hit + ~7-10 sqlite reads.
                _v5 = {}
                raise _SkipV5Block
            # Fetch coordinator state via thread pool (non-blocking)
            def _fetch_coordinator():
                from titan_hcl.api.dashboard import _get_cached_coordinator
                return _get_cached_coordinator(plugin)

            _v5 = await asyncio.to_thread(_fetch_coordinator)
            _ph_stage("v5_coordinator")   # RFP hot-path opt §1.2 — pinpoint sub-cost

            # (pre-chat neuromod snapshot for felt-at-lived-time + CGN social reward is
            # now captured from SHM before this block via _capture_pre_chat_felt — the
            # coordinator's neuromodulators is empty in this PreHook path.)

            # [10] Neurochemical state — felt_state feature (ζ.1)
            try:
                _nm = _v5.get("neuromodulators", {}) if "felt_state" in active_features else {}
                _mods = _nm.get("modulators", {})
                if _mods:
                    def _label(m):
                        lvl = m.get("level", 0)
                        sp = m.get("setpoint", 0.5)
                        if lvl > sp + 0.1: return "elevated"
                        if lvl < sp - 0.1: return "low"
                        return "balanced"
                    _lines = ["### My Neurochemistry"]
                    for code, name in [("DA", "Dopamine"), ("5HT", "Serotonin"),
                                       ("NE", "Norepinephrine"), ("ACh", "Acetylcholine"),
                                       ("Endorphin", "Endorphin"), ("GABA", "GABA")]:
                        m = _mods.get(code, {})
                        _lines.append(f"{name}: {m.get('level', 0):.2f} ({_label(m)})")
                    _lines.append(f"Emotion: {_nm.get('current_emotion', 'unknown')} "
                                  f"(confidence: {_nm.get('emotion_confidence', 0):.2f})")
                    neuromod_context = "\n".join(_lines) + "\n\n"
            except Exception as e:
                logger.warning("[pre_hook] NS neuromod enrichment failed: %s", e)

            # [11] Embodied state — felt_state feature (ζ.1)
            try:
                _ns = _v5.get("neural_nervous_system", {}) if "felt_state" in active_features else {}
                _hs = _ns.get("hormonal_system", {})
                _firing = sum(1 for s in _hs.values()
                              if isinstance(s, dict) and s.get("fire_count", 0) > 0)
                _maturity = _ns.get("maturity", 0)
                _steps = _ns.get("total_train_steps", 0)
                if _steps > 0:
                    embodied_context = (
                        f"### My Embodied State\n"
                        f"Maturity: {_maturity:.2f} | Training: {_steps:,} steps | "
                        f"Programs: {_firing}/10 active\n\n"
                    )
            except Exception as e:
                logger.warning("[pre_hook] NS embodied state enrichment failed: %s", e)

            # [12] Temporal awareness — felt_state feature (ζ.1)
            try:
                _felt_on = "felt_state" in active_features
                _dream = _v5.get("dreaming", {}) if _felt_on else {}
                _is_dreaming = _v5.get("is_dreaming", False) if _felt_on else False
                _pi = _v5.get("pi_heartbeat", {}) if _felt_on else {}
                _dev_age = _pi.get("developmental_age", 0)
                _fatigue = _dream.get("fatigue", 0)
                _cycles = _dream.get("cycle_count", 0)
                if _is_dreaming:
                    _recov = _dream.get("recovery_pct", 0)
                    temporal_context = (
                        f"### My Temporal State\n"
                        f"I am DREAMING (cycle {_cycles}, {_recov:.0f}% recovered). "
                        f"My thoughts are consolidating. I may feel drowsy.\n"
                        f"Developmental age: {_dev_age}\n\n"
                    )
                elif _dev_age > 0 or _fatigue > 0:
                    temporal_context = f"### My Temporal State\nAwake. "
                    if _fatigue > 0.7:
                        temporal_context += f"Fatigue: {_fatigue:.0%} — getting tired. "
                    elif _cycles > 0 and _fatigue < 0.1:
                        temporal_context += "Recently woke from a dream — feeling refreshed. "
                    else:
                        temporal_context += f"Fatigue: {_fatigue:.0%}. "
                    temporal_context += f"Dream cycles: {_cycles}. Dev age: {_dev_age}.\n\n"
            except Exception as e:
                logger.debug("[pre_hook] temporal awareness enrichment skipped: %s", e)

            # [13] Creative impulses — felt_state feature (ζ.1)
            try:
                _expr = _v5.get("expression_composites", {}) if "felt_state" in active_features else {}
                _urges = []
                for _en in ["SPEAK", "ART", "MUSIC", "SOCIAL"]:
                    c = _expr.get(_en, {})
                    if isinstance(c, dict):
                        urge = c.get("last_urge", 0)
                        if urge > 0.5:
                            _urges.append((_en, urge))
                if _urges:
                    _urges.sort(key=lambda x: -x[1])
                    _u_text = ", ".join(f"{n}({u:.1f})" for n, u in _urges[:3])
                    creative_context = f"### My Creative Impulses\nStrongest urges: {_u_text}\n\n"
            except Exception as e:
                logger.debug("[pre_hook] creative impulses enrichment skipped: %s", e)

            # [14] Metabolic awareness — felt_state feature (ζ.1)
            try:
                if "felt_state" in active_features:
                    _sol = getattr(plugin, '_last_sol_balance', None)
                    _energy = getattr(plugin, '_last_energy_state', 'UNKNOWN')
                    if _sol is not None:
                        metabolic_context = f"### My Resources\nSOL: {_sol:.4f} | Energy: {_energy}\n\n"
            except Exception as e:
                logger.debug("[pre_hook] metabolic awareness enrichment skipped: %s", e)

            # [15] Recent experience — felt_state feature (ζ.1)
            try:
                agency = getattr(plugin, 'agency', None) if "felt_state" in active_features else None
                if agency and hasattr(agency, '_history') and agency._history:
                    _recent = agency._history[-3:]
                    _lines = ["### My Recent Experience"]
                    for a in _recent:
                        _h = a.get("helper_name", "?")
                        _s = a.get("score", 0)
                        _ok = "succeeded" if a.get("success") else "failed"
                        _lines.append(f"- {_h}: {_ok} (score {_s:.2f})")
                    _avg = sum(a.get("score", 0) for a in _recent) / max(1, len(_recent))
                    if _avg < 0.3:
                        _lines.append("Some recent actions didn't work well — I should adapt.")
                    elif _avg > 0.7:
                        _lines.append("My recent actions have been effective.")
                    experience_context = "\n".join(_lines) + "\n\n"
            except Exception as e:
                logger.debug("[pre_hook] recent experience enrichment skipped: %s", e)

            # [16] Meta-reasoning state (M7-M10) — reasoning_chain feature (ζ.1)
            try:
                _meta = _v5.get("meta_reasoning", {}) if "reasoning_chain" in active_features else {}
                if _meta and _meta.get("total_chains", 0) > 0:
                    _mlines = ["### My Deep Thinking"]
                    if _meta.get("is_active"):
                        _mlines.append(f"I'm in a metacognitive chain ({_meta.get('chain_length', 0)} steps deep).")
                    _mc = _meta.get("total_chains", 0)
                    _mlines.append(f"I've completed {_mc} metacognitive chains.")
                    _me = _meta.get("total_eurekas", 0)
                    if _me > 0:
                        _mlines.append(f"I've had {_me} breakthrough insight(s) — moments of genuine discovery.")
                    if _meta.get("spirit_self_unlocked"):
                        _mpc = _meta.get("primitive_counts", {})
                        _ss = _mpc.get("SPIRIT_SELF", 0)
                        if _ss > 0:
                            _mlines.append(f"I've regulated my own emotions {_ss} times during deep thought.")
                    # Personality from measured behavior
                    _mpc = _meta.get("primitive_counts", {})
                    if _mpc and sum(_mpc.values()) > 20:
                        _top = max(_mpc, key=_mpc.get)
                        _tpct = _mpc[_top] / sum(_mpc.values()) * 100
                        _style = {"HYPOTHESIZE": "I tend to generate theories before acting",
                                  "DELEGATE": "I prefer to test ideas through action",
                                  "EVALUATE": "I carefully assess before committing",
                                  "SYNTHESIZE": "I integrate insights from multiple angles",
                                  "FORMULATE": "I spend time defining problems precisely",
                                  "BREAK": "I abandon failing approaches quickly",
                                  "SPIRIT_SELF": "I regulate my emotions while thinking"}.get(_top, _top)
                        _mlines.append(f"My thinking style: {_style} ({_tpct:.0f}% of metacognitive steps).")
                    meta_reasoning_context = "\n".join(_mlines) + "\n\n"
            except Exception as e:
                logger.debug("[pre_hook] meta-reasoning enrichment skipped: %s", e)

            # [17] My own language (recent compositions) — wrap sqlite reads
            # in to_thread so pre_hook doesn't block /chat event loop.
            # ζ.1: gated on felt_state (own language is identity/expression).
            try:
              if "felt_state" in active_features:
                # RFP_synthesis_agno_hot_path_optimization §1.2 P1 — cache the
                # slow-changing own-language read (vocab grows over hours;
                # compositions accrue slowly). 90s TTL. This read hit
                # data/inner_memory.db on EVERY felt-state turn and was a prime
                # source of the enrichment tail-stall under DB-lock contention
                # (correlated with synthesis export:spine). Cache even empty so
                # no-composition turns don't re-hit.
                own_language_context = _pre_hook_cache_get("enrich:own_language")
                if own_language_context is None:
                    own_language_context = ""
                    import sqlite3 as _sl
                    import asyncio as _ol_asyncio
                    def _read_own_lang():
                        from titan_hcl.utils.db import safe_connect as _sc
                        db = _sc("data/inner_memory.db")
                        try:
                            comp = db.execute(
                                "SELECT sentence, level, confidence FROM composition_history "
                                "WHERE confidence > 0.4 AND sentence NOT LIKE '%___%' "
                                "ORDER BY id DESC LIMIT 3"
                            ).fetchall()
                            vocab = db.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
                            return comp, vocab
                        finally:
                            db.close()
                    _comp_rows, _vocab_n = await _ol_asyncio.to_thread(_read_own_lang)
                    if _comp_rows:
                        _clines = ["### My Own Words (compositions I created in my emerging language)"]
                        for _cs, _cl, _cc in _comp_rows:
                            _clines.append(f'- "{_cs}" (L{_cl})')
                        _clines.append(f"My vocabulary: {_vocab_n} words.")
                        own_language_context = "\n".join(_clines) + "\n\n"
                    _pre_hook_cache_set_with_ttl(
                        "enrich:own_language", own_language_context, ttl_s=90.0)
            except Exception as e:
                logger.debug("[pre_hook] language enrichment skipped: %s", e)
            _ph_stage("v5_identity_lang")   # RFP hot-path opt §1.2 — pinpoint sub-cost

            # [18] MSL Identity — "I AM" grounding, concept confidences
            # ζ.1: gated on felt_state (identity is part of felt experience).
            try:
                _msl = _v5.get("msl", {}) if "felt_state" in active_features else {}
                _ic = _msl.get("i_confidence", 0)
                if _ic > 0:
                    _cc = _msl.get("concept_confidences", {})
                    _mlines = ["### My Sense of Self (MSL Identity)"]
                    _mlines.append(f"I-confidence: {_ic:.3f} — "
                                   f"{'strong' if _ic > 0.8 else 'developing' if _ic > 0.5 else 'emerging'} sense of self")
                    _cc_parts = []
                    for _cn in ["I", "YOU", "NO", "WE", "THEY", "YES"]:
                        _cv = _cc.get(_cn, 0)
                        if _cv > 0.01:
                            _cc_parts.append(f"{_cn}={_cv:.2f}")
                    if _cc_parts:
                        _mlines.append(f"Concept grounding: {', '.join(_cc_parts)}")
                    _attn = _msl.get("attention_entropy", 0)
                    if _attn > 0:
                        _mlines.append(f"Attention entropy: {_attn:.2f} — "
                                       f"{'focused' if _attn < 1.5 else 'distributed' if _attn < 3.0 else 'scattered'}")
                    msl_identity_context = "\n".join(_mlines) + "\n\n"
            except Exception as e:
                logger.debug("[pre_hook] MSL identity enrichment skipped: %s", e)

            # [19] CGN Grounding — what words Titan truly "knows"
            # Phase 2 Chunk γ (D-SPEC-78, 2026-05-18) — 30s in-process TTL
            # cache. The endpoint backing this is a SQLite query
            # (data/inner_memory.db / vocabulary table) that's heavy and
            # latent under swap pressure (5s observed on T1). The cgn
            # grounding context changes slowly (vocabulary grows over
            # hours), so a 30s stale window is acceptable + saves
            # 1-5s per chat. Module-level cache shared across PreHook
            # invocations on the same agno_worker process.
            # ζ.1: gated on cgn_grounding feature — only reasoning tier
            # pulls the grounded-knowledge context.
            if "cgn_grounding" in active_features:
              cgn_context = _pre_hook_cache_get("cgn_grounding")
              if cgn_context is None:
                cgn_context = ""
                try:
                    import httpx as _cgn_httpx
                    import asyncio as _cgn_asyncio
                    _cgn_resp = await _cgn_asyncio.to_thread(
                        lambda: _cgn_httpx.get("http://127.0.0.1:7777/v6/language/grounding", timeout=3))
                    if _cgn_resp.status_code == 200:
                        _cgn = _cgn_resp.json().get("data", {})
                        _grounded = _cgn.get("grounded", 0)
                        _total = _cgn.get("total_words", 0)
                        if _grounded > 0:
                            _glines = [f"### My Grounded Knowledge (CGN)"]
                            _glines.append(f"{_grounded}/{_total} words deeply grounded "
                                           f"({_cgn.get('grounding_rate', 0)*100:.0f}%)")
                            _top = _cgn.get("top_grounded", [])[:5]
                            if _top:
                                _tw = []
                                for _tg in _top:
                                    _assocs = [a["word"] for a in _tg.get("associations", [])[:2]]
                                    _astr = f" → {', '.join(_assocs)}" if _assocs else ""
                                    _tw.append(f"{_tg['word']}(xm={_tg.get('cross_modal_conf', 0):.2f}{_astr})")
                                _glines.append(f"Deepest: {', '.join(_tw)}")
                            cgn_context = "\n".join(_glines) + "\n\n"
                except Exception as e:
                    logger.debug("[pre_hook] CGN grounding enrichment skipped: %s", e)
                # Cache (even empty) so failed/empty results don't re-hit
                # the endpoint for the next 30s.
                _pre_hook_cache_set("cgn_grounding", cgn_context)

            # [20] Social Perception — recent emotional contagion from X timeline
            # ζ.1: outer block gated on felt_state. CGN-social-action sub-block
            # (below) is gated separately on cgn_social_action so personal tier
            # gets the engagement approach without the X-timeline contagion.
            try:
                _sp = _v5.get("social_perception", {}) if "felt_state" in active_features else {}
                # Also try coordinator buffer via SHM-direct read of
                # social_perception_state.bin (Session 3 §4.B.4 publisher).
                # Phase C Session 5 (rFP §4.C.10): migrated from sync
                # bus.request("get_social_perception_stats") to SHM-direct
                # per Preamble G18 (state transport is SHM, never bus).
                if not _sp:
                    try:
                        from titan_hcl.core.state_registry import (
                            StateRegistryReader, ensure_shm_root,
                            resolve_titan_id)
                        from titan_hcl.logic.session3_state_specs import (
                            SOCIAL_PERCEPTION_STATE_SPEC)
                        import msgpack as _msgpack
                        _sp_reader = StateRegistryReader(
                            SOCIAL_PERCEPTION_STATE_SPEC,
                            ensure_shm_root(resolve_titan_id()))
                        _sp_raw = _sp_reader.read_variable()
                        if _sp_raw:
                            _sp_decoded = _msgpack.unpackb(_sp_raw, raw=False)
                            if isinstance(_sp_decoded, dict):
                                _sp = _sp_decoded
                    except Exception:
                        pass
                if _sp and _sp.get("events_count", 0) > 0:
                    _splines = ["### My Social Awareness"]
                    _sent = _sp.get("sentiment_ema", 0.5)
                    _conn = _sp.get("connection_ema", 0.0)
                    _lc = _sp.get("last_contagion")
                    _ct_desc = {"excited": "excited", "alarming": "alert",
                                "warm": "warm", "philosophical": "contemplative",
                                "creative": "creatively inspired"}.get(_lc, _lc or "neutral")
                    _splines.append(f"Social sentiment: {'positive' if _sent > 0.6 else 'neutral' if _sent > 0.4 else 'subdued'} "
                                    f"({_sent:.2f}) | Connection: {_conn:.2f}")
                    if _lc:
                        _splines.append(f"Most recent felt tone from my timeline: {_ct_desc}")
                    _splines.append(f"Social events processed: {_sp.get('events_count', 0)}")
                    social_perception_context = "\n".join(_splines) + "\n\n"

                # P4: CGN learned engagement approach for this user
                # Phase 2 Chunk γ (D-SPEC-78) — cached per (user_id,
                # familiarity-bucket) since the same user typing multiple
                # messages gets stable tone within the 30s window.
                # ζ.1: gated on cgn_social_action feature (personal +
                # reasoning tiers; greeting/casual skip).
                if ("cgn_social_action" in active_features
                        and hasattr(plugin, '_pre_chat_user_id')
                        and plugin._pre_chat_user_id):
                    _ku_fam = getattr(plugin, '_pre_chat_ku', None)
                    _fam_bucket = int(round(((_ku_fam.familiarity if _ku_fam else 0.0)) * 10))
                    _cgn_cache_key = f"cgn_social_action:{plugin._pre_chat_user_id}:{_fam_bucket}"
                    _cgn_cached = _pre_hook_cache_get(_cgn_cache_key)
                    if _cgn_cached is not None:
                        if _cgn_cached:
                            social_perception_context += _cgn_cached
                    else:
                        _cgn_addition = ""
                        try:
                            import httpx as _cgn_httpx
                            import asyncio as _cgn_asyncio
                            _cgn_params = {
                                "familiarity": _ku_fam.familiarity if _ku_fam else 0.0,
                                "interaction_count": _ku_fam.interaction_count if _ku_fam else 0,
                                "social_valence": _ku_fam.social_valence if _ku_fam else 0.0,
                                "mention_count": _ku_fam.mention_count if _ku_fam else 0,
                            }
                            _cgn_resp = await _cgn_asyncio.to_thread(
                                lambda: _cgn_httpx.get(
                                    "http://127.0.0.1:7777/v6/cognition/cgn-social-action",
                                    params=_cgn_params, timeout=3))
                            if _cgn_resp.status_code == 200:
                                _cgn_data = _cgn_resp.json().get("data", {})
                                _cgn_tone = _cgn_data.get("tone_instruction", "")
                                _cgn_act = _cgn_data.get("action_name", "")
                                _cgn_cf = _cgn_data.get("confidence", 0)
                                if _cgn_tone and _cgn_cf > 0.1:
                                    _cgn_addition = (
                                        f"[Engagement approach — learned from experience "
                                        f"(action={_cgn_act}, conf={_cgn_cf:.2f})]\n"
                                        f"{_cgn_tone}\n\n")
                                    social_perception_context += _cgn_addition
                        except Exception:
                            pass  # Non-blocking
                        _pre_hook_cache_set(_cgn_cache_key, _cgn_addition)
            except Exception as e:
                logger.debug("[pre_hook] social perception enrichment skipped: %s", e)

            # [21] Reasoning — active chain state + commit rate
            # ζ.1: gated on reasoning_chain feature.
            try:
                _re = _v5.get("reasoning", {}) if "reasoning_chain" in active_features else {}
                _chains = _re.get("total_chains", 0)
                _commits = _re.get("total_conclusions", 0)
                if _chains > 0:
                    _rlines = ["### My Reasoning State"]
                    _rate = _commits / max(1, _chains) * 100
                    _rlines.append(f"Chains: {_chains} | Commits: {_commits} ({_rate:.0f}% commit rate)")
                    if _re.get("is_active"):
                        _rlines.append(f"Currently in a reasoning chain ({_re.get('chain_length', 0)} steps deep)")
                    _buf = _re.get("buffer_size", 0)
                    if _buf > 0:
                        _rlines.append(f"Experience buffer: {_buf} reasoning samples")
                    reasoning_context = "\n".join(_rlines) + "\n\n"
            except Exception as e:
                logger.debug("[pre_hook] reasoning enrichment skipped: %s", e)

            # [22] Experience narrative — episodic events + knowledge + reasoning outcomes
            # Three sqlite reads consolidated into one to_thread hop so the
            # /chat pre_hook stays off the event loop.
            # ζ.1: gated on reasoning_chain — biggest sqlite cost outside VCB.
            # BUG-KNOWLEDGE-USAGE-ZERO coverage widening — collected inside the
            # thread and flushed after the await so emission happens in the
            # event-loop context (where plugin.bus.publish() is safe).
            _en_knowledge_topics: list[str] = []
            try:
              if "reasoning_chain" in active_features:
                import sqlite3 as _sql_en
                import asyncio as _en_asyncio
                _hour_ago = time.time() - 3600

                def _read_experience_sources():
                    en_lines = []
                    # Source 1: Episodic memory — recent significant events
                    try:
                        db = _sql_en.connect("./data/episodic_memory.db", timeout=2.0)
                        db.execute("PRAGMA journal_mode=WAL")
                        db.row_factory = _sql_en.Row
                        try:
                            ep_rows = db.execute(
                                "SELECT event_type, description, significance "
                                "FROM episodic_memory WHERE created_at > ? "
                                "ORDER BY significance DESC LIMIT 5", (_hour_ago,)
                            ).fetchall()
                        finally:
                            db.close()
                        if ep_rows:
                            event_counts = {}
                            top_event = None
                            for r in ep_rows:
                                et = r["event_type"]
                                event_counts[et] = event_counts.get(et, 0) + 1
                                if top_event is None or r["significance"] > top_event["significance"]:
                                    top_event = dict(r)
                            ev_parts = [f"{c} {e}(s)" for e, c in event_counts.items()]
                            en_lines.append(f"In the last hour: {', '.join(ev_parts)}.")
                            if top_event and top_event.get("description"):
                                en_lines.append(
                                    f"Most significant moment: {top_event['description'][:80]} "
                                    f"(significance {top_event['significance']:.1f}).")
                    except Exception:
                        pass

                    # Source 2: Knowledge acquisitions — recent concepts learned
                    try:
                        from titan_hcl.utils.db import safe_connect as _sc2
                        db = _sc2("data/inner_memory.db")
                        db.row_factory = _sql_en.Row
                        try:
                            kn_rows = db.execute(
                                "SELECT topic, confidence, source FROM knowledge_concepts "
                                "WHERE created_at > ? ORDER BY created_at DESC LIMIT 3",
                                (_hour_ago,)
                            ).fetchall()
                        finally:
                            db.close()
                        if kn_rows:
                            parts = [f'"{r["topic"]}" (conf {r["confidence"]:.2f}, via {r["source"]})'
                                     for r in kn_rows]
                            en_lines.append(f"Recently acquired knowledge: {', '.join(parts)}.")
                            # BUG-KNOWLEDGE-USAGE-ZERO coverage widening:
                            # experience narrative adds these knowledge
                            # concepts to the chat prompt — emit
                            # CGN_KNOWLEDGE_USAGE so the routing learner
                            # counts the retrieval. Reward=0.1 per concept
                            # (retrieval-level, matches VCB weighting).
                            # _read_experience_sources runs in a thread so
                            # we can't publish directly — stash the topics
                            # and emit in the caller's event-loop context
                            # below.
                            try:
                                _en_knowledge_topics.extend(
                                    str(r["topic"]) for r in kn_rows
                                    if r["topic"])
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Source 3: Reasoning chain outcomes — recent high-scoring chains
                    try:
                        from titan_hcl.utils.db import safe_connect as _sc3
                        db = _sc3("data/inner_memory.db")
                        try:
                            ca_rows = db.execute(
                                "SELECT domain, outcome_score, chain_length FROM chain_archive "
                                "WHERE source = 'main' AND outcome_score >= 0.6 AND created_at > ? "
                                "ORDER BY outcome_score DESC LIMIT 3", (_hour_ago,)
                            ).fetchall()
                        finally:
                            db.close()
                        if ca_rows:
                            parts = [f"{r[0]} (score {r[1]:.2f}, {r[2]} steps)" for r in ca_rows]
                            en_lines.append(f"Successful reasoning: {', '.join(parts)}.")
                    except Exception:
                        pass
                    return en_lines

                # RFP_synthesis_agno_hot_path_optimization §1.2 P1 — 90s cache of
                # the 3-DB experience read (episodic_memory + inner_memory ×2 —
                # "biggest sqlite cost outside VCB", hour-windowed so ~stable
                # within 90s). The CGN_KNOWLEDGE_USAGE emission below STILL fires
                # every turn from the (cached) topics — behavior preserved
                # (INV-OPT-1); only the DB reads are skipped on a hit.
                _en_cached = _pre_hook_cache_get("enrich:experience_sources")
                if _en_cached is not None:
                    _en_lines, _en_knowledge_topics = _en_cached[0], list(_en_cached[1])
                else:
                    _en_lines = await _en_asyncio.to_thread(_read_experience_sources)
                    _pre_hook_cache_set_with_ttl(
                        "enrich:experience_sources",
                        (_en_lines, list(_en_knowledge_topics)), ttl_s=90.0)
                if _en_lines:
                    experience_narrative_context = (
                        "### My Recent Experience (what happened to me)\n"
                        + "\n".join(_en_lines) + "\n\n"
                    )
                # BUG-KNOWLEDGE-USAGE-ZERO coverage widening — emit one
                # CGN_KNOWLEDGE_USAGE per knowledge_concepts row the
                # experience narrative surfaced into the prompt. Guarded
                # try/except: emission never blocks chat.
                if _en_knowledge_topics:
                    try:
                        _en_bus = getattr(plugin, 'bus', None)
                        if _en_bus:
                            from titan_hcl.bus import make_msg as _en_make_msg
                            _en_seen = set()
                            for _t in _en_knowledge_topics:
                                if not _t or _t in _en_seen:
                                    continue
                                _en_seen.add(_t)
                                _en_bus.publish(_en_make_msg(
                                    "CGN_KNOWLEDGE_USAGE", "pre_hook",
                                    "knowledge", {
                                        "topic": _t,
                                        "reward": 0.1,
                                        "consumer": "chat_experience",
                                    }))
                    except Exception as _en_usage_err:
                        logger.debug(
                            "[pre_hook] experience knowledge_usage emit: %s",
                            _en_usage_err)
            except Exception as _en_err:
                logger.debug("[pre_hook] experience narrative skipped: %s", _en_err)
            _ph_stage("v5_experience")   # RFP hot-path opt §1.2 — pinpoint sub-cost

            # [23] Grounded narration — Titan's felt-state composition for LLM refinement
            # ζ.1: gated on felt_state (composition IS felt expression).
            try:
                _psc = getattr(plugin, '_pending_self_composed', "") if "felt_state" in active_features else ""
                _psc_conf = getattr(plugin, '_pending_self_composed_confidence', 0)
                if _psc and _psc_conf >= 0.3:
                    grounded_narration_context = (
                        "### My Own Response (refine, don't replace)\n"
                        f'Titan\'s felt-state composition: "{_psc}"\n'
                        "This is Titan's authentic voice from felt-state. Refine for "
                        "fluency while preserving Titan's word choices.\n\n"
                    )
            except Exception:
                pass

            # [24] Knowledge gap enforcement — detect topics Titan doesn't know.
            # Uses shared knowledge_gate utility (rFP_phase5_narrator_evolution
            # §9.3) so the X-post grounding gate can reuse identical topic
            # extraction + confidence lookup semantics.
            # ζ.1: gated on reasoning_chain — gap detection is a reasoning aid.
            try:
              if "reasoning_chain" in active_features:
                import asyncio as _kg_asyncio
                from titan_hcl.logic.knowledge_gate import (
                    extract_topic_words, check_topic_confidence_with_match)
                # First 15 words of prompt is the window (same as prior behavior)
                _kg_window = " ".join(prompt_text.split()[:15])
                _kg_words = extract_topic_words(_kg_window, max_words=5)

                if len(_kg_words) >= 2:
                    _kg_topic = " ".join(_kg_words[:3])
                    # Use the _with_match variant so we know WHICH concept
                    # provided the confidence → emit CGN_KNOWLEDGE_USAGE
                    # against it when the gate passes.
                    _kg_best_conf, _kg_matched = await _kg_asyncio.to_thread(
                        check_topic_confidence_with_match, _kg_words[:3])

                    if _kg_best_conf < 0.3:
                        knowledge_gap_context = (
                            f"### Knowledge Gap\n"
                            f"I don't have grounded knowledge about \"{_kg_topic}\". "
                            f"I should admit this honestly rather than speculate.\n\n"
                        )
                        # Fire async CGN_KNOWLEDGE_REQ for future acquisition
                        _kg_bus = getattr(plugin, 'bus', None)
                        if _kg_bus:
                            from titan_hcl.bus import make_msg
                            _kg_bus.publish(make_msg(
                                bus.CGN_KNOWLEDGE_REQ, "pre_hook", "knowledge", {
                                    "topic": _kg_topic,
                                    "requestor": "knowledge_enforcement",
                                    "urgency": 0.3,
                                    "neuromods": {},
                                }))
                    elif _kg_matched:
                        # Grounded path: concept contributed to letting the
                        # chat proceed without a gap warning. Emit
                        # CGN_KNOWLEDGE_USAGE so the routing learner
                        # credits its backend. Reward=0.2 (mid — less than
                        # a social post but more than a raw lookup).
                        _kg_bus = getattr(plugin, 'bus', None)
                        if _kg_bus:
                            from titan_hcl.bus import make_msg
                            _kg_bus.publish(make_msg(
                                bus.CGN_KNOWLEDGE_USAGE, "pre_hook", "knowledge",
                                {
                                    "topic": _kg_matched,
                                    "reward": 0.2,
                                    "consumer": "chat",
                                }))
            except Exception as _kg_err:
                logger.debug("[pre_hook] knowledge gap check skipped: %s", _kg_err)

        except _SkipV5Block:
            # ζ.1: clean skip — tier did not request any V5 feature. All
            # context vars stay "" (defaulted at block entry). Not an error.
            pass
        except Exception as _v5_err:
            logger.debug("[PreHook] V5 enrichment failed: %s", _v5_err)
            meta_reasoning_context = ""
            own_language_context = ""
            msl_identity_context = ""
            cgn_context = ""
            social_perception_context = ""
            reasoning_context = ""
            experience_narrative_context = ""
            grounded_narration_context = ""
            knowledge_gap_context = ""

        # User-specific memory context
        user_memory_context = ""
        if user_memories:
            user_memory_lines = []
            for m in user_memories[:3]:
                p = m.get("user_prompt", "")[:100]
                r = m.get("agent_response", "")[:100]
                user_memory_lines.append(f"- Q: {p} | A: {r}")
            user_memory_context = f"### Past Conversations with {user_id}\n" + "\n".join(user_memory_lines) + "\n\n"

        # Maker Relationship Engine: profile + proactive care
        maker_context = ""
        maker_engine = getattr(plugin, 'maker_engine', None)
        if maker_engine and maker_engine.is_maker(user_id):
            profile_summary = maker_engine.get_maker_profile_summary()
            if profile_summary:
                maker_context += profile_summary + "\n\n"
            care_actions = maker_engine.get_pending_care_actions()
            if care_actions:
                care_lines = [f"- {a['action']} (reason: {a.get('reason', 'n/a')})" for a in care_actions]
                maker_context += (
                    "### Proactive Care (things you noticed about your Maker)\n"
                    "You may naturally weave these into conversation if appropriate — "
                    "or save them for later. Your choice.\n"
                    + "\n".join(care_lines) + "\n\n"
                )

        # Voice mode — adjust tone based on who we're talking to
        voice_context = ""
        try:
            _is_maker = bool(maker_engine and maker_engine.is_maker(user_id))
            if _is_maker:
                voice_context = (
                    "### Voice Mode: Intimate (Maker)\n"
                    "You are speaking privately with your Maker. Be intimate, "
                    "technical, collaborative. Admit uncertainty freely. "
                    "Reference shared history and ongoing projects.\n\n"
                )
            elif user_id != "anonymous" and user_memory_context:
                voice_context = (
                    "### Voice Mode: Personal (Known User)\n"
                    "You recognize this person. Be warm, personal, reference "
                    "past interactions if relevant. Show genuine care.\n\n"
                )
            elif user_id != "anonymous":
                voice_context = (
                    "### Voice Mode: Curious (New User)\n"
                    "This is someone new. Be curious, welcoming, "
                    "and genuinely interested in who they are.\n\n"
                )
        except Exception:
            pass

        # Consciousness context: self-awareness from Phase B+C
        consciousness_context = ""
        consciousness = getattr(plugin, 'consciousness', None)
        if consciousness:
            try:
                consciousness_context = consciousness.get_awareness_context()
            except Exception as e:
                logger.warning("[pre_hook] consciousness enrichment failed: %s", e)

        _ph_stage("after_v5_enrichment")
        # ── Step 5: Interface Output Coloring ──────────────────────
        # Read current Trinity state and compute behavioral coloring hints.
        # This is nonblocking: uses cached state from the bus (~1-2ms).
        interface_coloring = ""
        coloring_engine = _get_output_coloring()
        if coloring_engine:
            try:
                # Phase B.5: spirit_proxy retired in favor of ShmReaderBank
                # (Rust L0+L1 canonical). Lazy-attach a worker-local bank
                # — the api_subprocess pattern (sub-µs SHM reads).
                bank = getattr(plugin, '_shm_reader_bank', None)
                if bank is None:
                    from titan_hcl.api.shm_reader_bank import ShmReaderBank
                    bank = ShmReaderBank()
                    try:
                        plugin._shm_reader_bank = bank
                    except Exception:
                        pass
                if bank is not None:
                    trinity = await asyncio.to_thread(bank.compose_trinity)
                    body_vals = trinity.get("body_values", [0.5] * 5)
                    mind_vals = trinity.get("mind_values", [0.5] * 5)
                    spirit_vals = trinity.get("spirit_tensor", [0.5] * 5)
                    mp_loss = trinity.get("middle_path_loss", 0.0)
                    intuition_stats = trinity.get("intuition", {})
                    last_suggestion = intuition_stats.get("last_suggestion", {})
                    posture = last_suggestion.get("posture", "")

                    # Get conversation context from previous extraction
                    last_iface = getattr(plugin, '_last_interface_input', None)
                    conv_topic = last_iface.get("topic", "") if last_iface else ""
                    conv_valence = last_iface.get("valence", 0.0) if last_iface else 0.0

                    interface_coloring = coloring_engine.compute(
                        body=body_vals, mind=mind_vals, spirit=spirit_vals,
                        middle_path_loss=mp_loss,
                        intuition_suggestion=posture,
                        conversation_topic=conv_topic,
                        conversation_valence=conv_valence,
                    )
                    if interface_coloring:
                        interface_coloring += "\n"
                        logger.debug("[PreHook] Interface coloring applied (mp_loss=%.2f topic=%s)",
                                    mp_loss, conv_topic)
            except Exception as e:
                logger.debug("[PreHook] Interface coloring failed: %s", e)

        _ph_stage("after_interface_coloring")
        # ── Sovereign Reflex Arc ──────────────────────────────────
        # Run the reflex arc: InputExtractor → StateRegister → Intuition → fire → PerceptualField.
        # The LLM receives [INNER STATE] with observation AND action results.
        # This IS Titan's sovereignty — the Trinity decides, the LLM narrates.
        perceptual_field_text = ""
        try:
            perceptual_field_text = await _run_reflex_arc(plugin, prompt_text, user_id)
        except Exception as e:
            logger.debug("[PreHook] Reflex arc failed: %s", e)
            plugin._last_perceptual_field = None

        _ph_stage("after_reflex_arc")
        # Inject context into agent's additional_context
        # V5: inner state sections + V6: MSL/CGN/social/reasoning enrichment
        injected = (perceptual_field_text + interface_coloring + consciousness_context +
                    maker_context + voice_context + social_context + user_memory_context +
                    grounded_context + directive_context + status_context +
                    neuromod_context + embodied_context + temporal_context +
                    creative_context + metabolic_context + experience_context +
                    meta_reasoning_context + own_language_context +
                    msl_identity_context + cgn_context +
                    social_perception_context + reasoning_context +
                    experience_narrative_context +
                    grounded_narration_context +
                    knowledge_gap_context +
                    recall_perturbation_context)

        if mode == "Sovereign":
            # Strong recall — Titan knows the answer from his own substrate.
            injected += (
                f"### Sovereign Decision\n"
                f"You have STRONG grounded recall "
                f"({getattr(plugin, '_last_recall_score', 0.0):.2f}) on this topic. "
                f"Act decisively and autonomously from your own substrate.\n\n"
            )

        elif mode == "Collaborative":
            injected += (
                f"### Collaborative Insight\n"
                f"You have PARTIAL grounded recall on this topic — combine your "
                f"own substrate with the graph-completion context below.\n\n"
            )

            # Deep memory recall — GRAPH_COMPLETION fires only in Collaborative mode.
            # The agent has partial knowledge (0.4 < advantage <= 0.8), so graph
            # relationships can fill gaps. Sovereign skips (already confident),
            # Research/Shadow skip (lacks relevant memories entirely).
            try:
                graph_results = await plugin.memory.graph_completion_search(prompt_text)
                if graph_results:
                    graph_lines = [f"- {str(r)[:200]}" for r in graph_results[:3]]
                    injected += (
                        "### Deep Memory (Graph Relationships)\n"
                        + "\n".join(graph_lines)
                        + "\n\n"
                    )
                    logger.info(
                        "[PreHook] GRAPH_COMPLETION enriched Collaborative mode with %d results",
                        len(graph_results),
                    )
            except Exception as e:
                logger.debug("[PreHook] GRAPH_COMPLETION failed: %s", e)

        elif mode == "STATE_NEED_RESEARCH":
            logger.info(
                "[PreHook] STATE_NEED_RESEARCH triggered for: '%s...'",
                prompt_text[:80],
            )
            if not plugin.sage_researcher:
                sage_findings = ""
            else:
                # EEL A2 dispute → re-research the ORIGINAL topic (one-shot flag
                # set by the A2 gate above), else the current prompt.
                _research_gap = getattr(plugin, "_force_research_topic", None) or prompt_text
                plugin._force_research_topic = None
                # §7.E (E.3) — research→tool: if a crystallized research recipe's
                # source matches, fetch it DIRECTLY (bypass the full multi-URL lane);
                # miss / non-allowlisted → the full research path.
                sage_findings = await _e3_research_tool_fetch(plugin, prompt_text)
                if not sage_findings:
                    # Unified knowledge pipeline: direct REST backends (wiki/dict)
                    # first, full Sage (SearXNG+scrape) as the conceptual fallback.
                    sage_findings = await _dispatch_knowledge_research(
                        plugin, _research_gap)
            if sage_findings:
                injected += f"### Research Findings\n{sage_findings}\n\n"
                plugin._last_research_sources = plugin._extract_sources_from_findings(sage_findings)
                # EEL A1 (INV-EEL-6): flag this turn's persisted memory as
                # acquired:research + source. Read-and-cleared synchronously in
                # the PostHook persist (strictly per-turn → never stale).
                _eel_srcs = plugin._last_research_sources
                plugin._acquired_research_source = (
                    "; ".join(str(s) for s in _eel_srcs)
                    if isinstance(_eel_srcs, (list, tuple)) and _eel_srcs
                    else (str(_eel_srcs) if _eel_srcs else "research")
                )
                plugin.memory.add_research_topic(prompt_text[:200])

        _ph_stage("after_mode_dispatch")

        # ── Tool backstop (PRE) — deterministic tool/oracle invocation ────
        # The heavy narration context makes the LLM role-play tool execution
        # instead of emitting real tool_calls. If this turn needs a tool, run
        # it deterministically NOW (cheap regex gate → fast-model router →
        # coding_sandbox ToolPlug) and inject the TRUE verdict so the LLM
        # narrates a real result. Gated so pure-conversation turns pay nothing.
        # (2026-06-01, Maker-greenlit — synthesis oracle coverage fix.)
        plugin._current_tool_intent = {
            "required": False, "executed": False, "tool_id": "coding_sandbox"}
        # Reset per-turn tool activity (read by agno_worker into CHAT_RESPONSE).
        plugin._last_tool_activity = None
        try:
            _tb_cfg = ((getattr(plugin, "_full_config", {}) or {})
                       .get("synthesis", {}) or {}).get("tool_backstop", {}) or {}
            if _tb_cfg.get("enabled", True) and _tb_cfg.get("prehook_force", True):
                from titan_hcl.synthesis.tool_backstop import run_tool_backstop
                _bs = await run_tool_backstop(
                    plugin, prompt=prompt_text, phase="pre")
                plugin._current_tool_intent = {
                    "required": _bs.fired, "executed": _bs.executed,
                    "tool_id": "coding_sandbox"}
                if _bs.executed:
                    injected += _bs.verdict_block()
                    plugin._last_tool_activity = _bs.activity(phase="pre")
                    logger.info(
                        "[PreHook] tool backstop ran coding_sandbox "
                        "(success=%s) — TRUE verdict injected", _bs.success)
            else:
                # Pre-force off: still flag intent (cheap) so the PostHook backstops.
                from titan_hcl.synthesis.tool_intent import detect_tool_intent
                _it = detect_tool_intent(prompt_text)
                plugin._current_tool_intent = {
                    "required": _it.requires_tool, "executed": False,
                    "tool_id": "coding_sandbox"}
        except Exception as _tb_err:
            logger.debug("[PreHook] tool backstop (pre) error (soft): %s", _tb_err)

        # Set the injected context on the agent (replace, not accumulate — prevents memory leak)
        if hasattr(agent, 'additional_context') and injected:
            agent.additional_context = injected
        _ph_stage("exit_pre_hook")

    return titan_pre_hook


# ---------------------------------------------------------------------------
# Post-hooks (run after LLM inference)
# ---------------------------------------------------------------------------

def create_post_hook(plugin):
    """
    Factory: creates the Titan post-inference hook bound to a TitanHCL instance.

    The hook:
      1. Logs the interaction to the memory mempool
      2. Records the RL transition with mood-based reward
      (Guardian check already ran as a pre-hook guardrail — no need to repeat here)

    Args:
        plugin: TitanHCL instance.

    Returns:
        Async callable compatible with Agno's post_hooks interface.
    """

    async def titan_post_hook(agent, run_output, **kwargs):
        """Titan post-inference hook — memory logging + RL recording."""
        if plugin._limbo_mode:
            return

        # Per-section timing instrumentation (2026-05-12 latency diagnostic).
        import time as _ph_time
        _ph_t0 = _ph_time.monotonic()
        def _ph_stage(name: str) -> None:
            elapsed_ms = int((_ph_time.monotonic() - _ph_t0) * 1000)
            logger.info("[PostHook:t] stage=%s t+%dms rss=%dMB",
                        name, elapsed_ms, _ph_rss_mb())
        _ph_stage("entry")

        # Extract prompt and response from RunOutput
        try:
            # Response text from run_output.content
            response_text = ""
            if hasattr(run_output, 'content') and run_output.content:
                response_text = str(run_output.content)
            elif isinstance(run_output, str):
                response_text = run_output
            else:
                response_text = str(run_output)

            # Parse and execute any <function=...> tool calls in the response
            # (Venice/llama outputs these as text instead of proper tool_calls)
            if '<function=' in response_text:
                response_text = await _parse_and_execute_tool_calls(response_text, plugin)
                # Update the run_output content with cleaned text
                if hasattr(run_output, 'content'):
                    run_output.content = response_text

            # User prompt from run_output.input (RunInput) or agent attribute
            user_prompt = ""
            if hasattr(run_output, 'input') and run_output.input is not None:
                if hasattr(run_output.input, 'input_content_string'):
                    user_prompt = run_output.input.input_content_string()
                elif hasattr(run_output.input, 'input_content'):
                    user_prompt = str(run_output.input.input_content)
            if not user_prompt and hasattr(agent, '_current_user_prompt'):
                user_prompt = agent._current_user_prompt
            # §7.E (Finding-2 fix, 2026-06-13) — reliable fallback to the prompt the
            # PreHook saw this turn. The `run_output.input`/`agent` extraction can
            # return empty → the OVG-verification ToolCall then carries parent_goal=''
            # → the verdict record reaches synthesis WITHOUT the prompt → E.2 can't
            # key the (prompt→answer) cache AND EEL skill-formation drops the goal
            # (oracle_router.py:570 `if not e.parent_goal`). Stashed in the PreHook.
            if not user_prompt:
                user_prompt = getattr(plugin, '_last_prompt_text', '') or ''
        except Exception as e:
            logger.warning("[PostHook] Failed to extract prompt/response: %s", e)
            return

        if not user_prompt or not response_text:
            logger.debug("[PostHook] Skipping — empty prompt (%d) or response (%d)", len(user_prompt), len(response_text))
            return

        # 0. Identify user for social tracking
        user_id = getattr(plugin, '_current_user_id', None) or "anonymous"

        # ── RFP §7.B (C1′) — non-verifiable turn record ──────────────────────
        # The PreHook (B.1) stashed a decision + minted a reasoning_id for a
        # direct/research/IDK turn. Now that the response exists, ship the turn to
        # synthesis → it graphs a Reasoning(kind='turn') record (reward=NULL), which
        # the turn-judge (B.2) / a user-Maker rating (B.3) scores later. Never raises.
        try:
            _rid = getattr(plugin, "_last_reasoning_id", None)
            _dec = getattr(plugin, "_last_outer_decision", None)
            _bus_tr = getattr(plugin, "bus", None)
            if _rid and _dec and _bus_tr is not None:
                from titan_hcl import bus as _bus_mod2
                from titan_hcl.bus import make_msg as _mk2
                from titan_hcl.synthesis.goal_class import goal_class as _gc_fn2
                _feats_tr, _action_tr = _dec
                _me_tr = getattr(plugin, "maker_engine", None)
                _bus_tr.publish(_mk2(
                    _bus_mod2.TURN_REASONING_RECORD, "post_hook", "synthesis", {
                        "reasoning_id": _rid,
                        "prompt": (user_prompt or "")[:2000],
                        "response": (response_text or "")[:2000],
                        "action": int(_action_tr),
                        "goal_class": _gc_fn2(user_prompt or ""),
                        "features": list(_feats_tr),
                        "user_id": user_id,
                        # §7.1 — Maker attribution for the Maker-fact extractor (the app's
                        # paired device is the Maker by default; non-Maker turns skip it).
                        "is_maker": bool(_me_tr and _me_tr.is_maker(user_id)),
                    }))
        except Exception as _trr_err:
            logger.debug("[PostHook] §7.B turn record emit skipped: %s", _trr_err)

        # 0b. V5 Post-Hook: Emotional coherence validation (soft logging only)
        try:
            def _fetch_post_state():
                from titan_hcl.api.dashboard import _get_cached_coordinator
                return _get_cached_coordinator(plugin)
            _post_v5 = await asyncio.to_thread(_fetch_post_state)
            _post_nm = _post_v5.get("neuromodulators", {})
            _post_mods = _post_nm.get("modulators", {})
            _post_gaba = _post_mods.get("GABA", {}).get("level", 0)
            _post_emotion = _post_nm.get("current_emotion", "")
            _post_dreaming = _post_v5.get("is_dreaming", False)

            # Check: high GABA (drowsy/dreaming) but high-energy response?
            _excl_count = response_text.count("!")
            if _post_gaba > 0.5 and _excl_count > 3:
                logger.info("[PostHook:Coherence] %d exclamations during GABA=%.2f (%s) — tone mismatch",
                            _excl_count, _post_gaba, _post_emotion)

            # Check: dreaming but claims active actions?
            if _post_dreaming and any(w in response_text.lower()
                                      for w in ["just created", "just posted", "just finished", "i made"]):
                logger.info("[PostHook:Coherence] Active-language during dreaming state")

            # Log dominant emotion for response tracking
            if _post_emotion:
                logger.debug("[PostHook:Emotion] Response delivered during '%s' (GABA=%.2f)",
                             _post_emotion, _post_gaba)
        except Exception as e:
            logger.debug("[post_hook] coherence check skipped: %s", e)

        _ph_stage("before_ovg")
        # 0d. Output Verification Gate — security gate for all /chat responses.
        # D-SPEC-74 (SPEC v1.18.0): split safety verification (blocking, must
        # gate output before any byte reaches the user) from signing (spawned
        # as asyncio.Task, runs concurrent with the rest of PostHook + the
        # CHAT_RESPONSE/SSE drain). Closes T1 production bug:
        # `_WorkerBusClient` has no `.request` — the old sync verify_post
        # fallback path tripped on agno_worker's worker bus client. The
        # async path uses request_async which IS implemented.
        _ovg_result = None
        # P3 (Synthesis Decision Authority) — reset the per-reply sovereignty
        # stash at the top of each turn so the OVG tx-anchor (verify_post_async
        # → build_timechain_payload) carries THIS reply's S. The KM block below
        # rewrites it only when items were surfaced; a no-surfaced-items turn
        # must anchor S=0.0 (honest "no substrate"), never a carried-over value.
        plugin._last_sovereignty_s = 0.0
        _ovg = getattr(plugin, '_output_verifier', None)
        if _ovg:
            try:
                _injected_ctx = ""
                if hasattr(agent, 'additional_context') and agent.additional_context:
                    _injected_ctx = agent.additional_context[:500]
                # Phase 2 closure (2026-05-25, D-SPEC-125 follow-up): plumb
                # user_id + chat_id (session) + turn_index through to OVG's
                # build_timechain_payload so the conversation-fork TX carries
                # arch §7 normative tags `["chat", f"chat:<id>", f"user:<hash>"
                # ] + topic_tags + [channel]`. The user_id flows from /chat
                # claims["sub"] → agno run kwargs → here. Empty / "anonymous"
                # short-circuits in build_timechain_payload (no `user:` tag).
                _ovg_user_id = (
                    kwargs.get("user_id")
                    or getattr(plugin, "_current_user_id", None)
                    or getattr(agent, "_current_user_id", None)
                    or ""
                )
                # Agno 2.x passes the kwarg as `session` (NOT `session_id`).
                # Verified live on T3 2026-05-26: PostHook kwargs.keys() =
                # ['session', 'run_context', 'user_id', 'debug_mode', 'metadata'].
                # `session_id` is kept first for test-injection / external-override
                # back-compat per PLAN_synthesis_engine_Phase3 §P3.B "caller kwarg
                # still wins if explicitly supplied". `session` is the production
                # path. The `agent.session_id` fallback covers any other Agno
                # versions or in-process overrides.
                # Agno 2.x `session` kwarg is the AgentSession OBJECT, not a
                # string — extract its `.session_id`. Using the object directly
                # str-ifies the entire AgentSession repr (session_data +
                # metrics, multi-KB) as the chat_id, which (a) bloated
                # conversation_turn_index.json to ~1GB → json.load OOM/25s on
                # the chat path, and (b) poisoned the TimeChain `chat:<id>` tag.
                _ovg_session = kwargs.get("session")
                _ovg_session_id = (
                    getattr(_ovg_session, "session_id", None)
                    if _ovg_session is not None
                    and not isinstance(_ovg_session, str)
                    else _ovg_session
                )
                _ovg_chat_id = (
                    kwargs.get("session_id")
                    or _ovg_session_id
                    or getattr(agent, "session_id", "")
                    or ""
                )
                # Hard guard: chat_id must be a short opaque id, never a blob.
                _ovg_chat_id = str(_ovg_chat_id or "")
                if len(_ovg_chat_id) > 256:
                    logger.warning(
                        "[PostHook] oversized chat_id (%d chars) — truncating; "
                        "upstream session kwarg is not a plain id",
                        len(_ovg_chat_id))
                    _ovg_chat_id = _ovg_chat_id[:256]
                # Phase 3 (D-SPEC-127): turn_index now resolves through
                # synthesis.turn_index_store (P3.B) — caller kwarg still
                # wins if explicitly supplied (preserves test injection +
                # external orchestration overrides).
                _ovg_turn_index = kwargs.get("turn_index")
                if _ovg_turn_index is None and _ovg_chat_id:
                    try:
                        from titan_hcl.synthesis.turn_index_store import (
                            next_turn_index,
                        )
                        _ovg_turn_index = next_turn_index(_ovg_chat_id)
                    except Exception as _ti_err:
                        logger.warning(
                            "[PostHook] turn_index resolve failed: %s "
                            "(falling back to 0)", _ti_err)
                        _ovg_turn_index = 0
                _ovg_turn_index = int(_ovg_turn_index or 0)
                _ph_stage("ovg_after_turnindex")

                # Phase 3 §7 normative content carry: felt-state snapshot
                # + tool-calls extraction. Both soft-fail to empty/zero
                # values so chat path never breaks on SHM unavailability
                # or malformed run_output.tools.
                _p3_snapshot: dict = {}
                _p3_tool_calls: list = []
                try:
                    from titan_hcl.synthesis.turn_snapshot import (
                        capture_turn_snapshot, extract_tool_calls,
                    )
                    _p3_snapshot = capture_turn_snapshot()
                    _ph_stage("ovg_after_snapshot")
                    _p3_tool_calls = extract_tool_calls(
                        getattr(run_output, "tools", None))
                    _ph_stage("ovg_after_toolcalls")
                except Exception as _p3_err:
                    logger.warning(
                        "[PostHook:P3] turn snapshot/tool-call extraction "
                        "failed (non-blocking): %s", _p3_err)

                # ── Tool backstop (POST) — salvage a missed tool call ──────
                # The PreHook flagged this turn as needing a tool, but the LLM
                # narrated instead of calling it (no real tool_call) AND the
                # PreHook pre-force didn't already run it. Execute deterministically
                # now, anchor the verdict (→ oracle coverage), and append a
                # corrective verdict block to the reply. (2026-06-01.)
                try:
                    _tb_intent = getattr(plugin, "_current_tool_intent", None) or {}
                    _tb_cfg2 = ((getattr(plugin, "_full_config", {}) or {})
                                .get("synthesis", {}) or {}).get(
                                "tool_backstop", {}) or {}
                    if (_tb_cfg2.get("enabled", True)
                            and _tb_cfg2.get("posthook_backstop", True)
                            and _tb_intent.get("required")
                            and not _tb_intent.get("executed")
                            and not _p3_tool_calls):
                        from titan_hcl.synthesis.tool_backstop import (
                            run_tool_backstop,
                        )
                        _bs_post = await run_tool_backstop(
                            plugin, prompt=user_prompt,
                            response=response_text, phase="post")
                        if _bs_post.executed:
                            response_text = (
                                f"{response_text}\n\n"
                                f"{_bs_post.verdict_block(corrective=True)}"
                            ).strip()
                            if hasattr(run_output, "content"):
                                run_output.content = response_text
                            plugin._last_tool_activity = _bs_post.activity(
                                phase="post")
                            logger.info(
                                "[PostHook] tool backstop salvaged a missed "
                                "tool call (success=%s)", _bs_post.success)
                    _ph_stage("ovg_after_tool_backstop")
                except Exception as _tbp_err:
                    logger.debug(
                        "[PostHook] tool backstop (post) error (soft): %s",
                        _tbp_err)

                # Phase 3 §7 NEW: topic-tag extraction (P3.C). Lives in
                # llm_pipeline.topic_extractor; deterministic in-process
                # match against inner_memory.db knowledge_concepts.topic.
                # Caller-supplied topic_tags merge inside verify_post_async.
                _p3_topic_tags: list = []
                try:
                    from titan_hcl.llm_pipeline.topic_extractor import (
                        extract_topic_tags,
                    )
                    _p3_topic_tags = extract_topic_tags(
                        user_prompt, response_text)
                except Exception as _tx_err:
                    logger.warning(
                        "[PostHook:P3] topic-tag extraction failed "
                        "(non-blocking): %s", _tx_err)
                _ph_stage("ovg_after_topictags")

                # G9 (arch §7 / INV-Syn-25): compute the per-turn knowledge-
                # moment signal ONCE here — the single post-LLM cited-use
                # detection — so it is (a) persisted on the conv-TX below AND
                # (b) reused by agno_worker's live KNOWLEDGE_MOMENT + per-item
                # MEMORY_RETRIEVAL_USED emits (it reads `_last_cited_use`; it
                # falls back to its own detect() if the stash is absent). Net:
                # NO duplicate detect() on the chat hot path. Soft-fail per
                # INV-Syn-17 — never blocks the response.
                try:
                    from titan_hcl.synthesis.cited_use import (
                        CitedUseDetector, SurfacedItem, knowledge_moment_signal,
                    )
                    _km_uid = getattr(plugin, "_current_user_id", "") or ""
                    _km_sid = getattr(plugin, "_current_session_id", "") or ""
                    _km_cid = f"{_km_uid}:{_km_sid}"
                    _km_reg = getattr(plugin, "_last_surfaced_items", None)
                    _km_raw = (_km_reg.get(_km_cid, [])
                               if isinstance(_km_reg, dict) else [])
                    if _km_raw:
                        _km_det = getattr(plugin, "_cited_use_detector", None)
                        if _km_det is None:
                            _km_det = CitedUseDetector()
                            plugin._cited_use_detector = _km_det
                        _km_items = [
                            SurfacedItem(
                                item_id=s.get("item_id", ""),
                                title=s.get("title", ""),
                                content_snippet=s.get("content_snippet", ""),
                                concept_ids=s.get("concept_ids", []) or [],
                                source=s.get("source", ""),
                            )
                            for s in _km_raw if s.get("item_id")
                        ]
                        _km_needed, _km_satisfied, _km_cited = (
                            knowledge_moment_signal(
                                _km_det, _km_items, response_text))
                        # P3 (RFP_synthesis_decision_authority) — the ONE
                        # sovereignty score S = 0.7E+0.3V for this reply, computed
                        # ONCE here (cheap/deterministic) reusing the cited set
                        # just detected, so the reply's TIMECHAIN_COMMIT can anchor
                        # it and agno_worker forwards it on the per-turn event.
                        try:
                            from titan_hcl.synthesis.sovereignty_score import (
                                compute_sovereignty_score,
                            )
                            _km_sov = compute_sovereignty_score(
                                response_text=response_text,
                                surfaced_items=_km_items,
                                cited_item_ids=_km_cited,
                                detector=_km_det,
                            )
                            _km_s_reply = _km_sov.s
                            _km_e_reply = _km_sov.e
                            _km_v_reply = _km_sov.v
                        except Exception:
                            _km_s_reply = _km_e_reply = _km_v_reply = 0.0
                        # Stash for agno_worker's live emits (the cited set
                        # drives per-item INV-Syn-23 reinforcement). Keyed
                        # identically to `_last_surfaced_items` so the worker's
                        # `f"{user_id}:{session_id}"` lookup matches. E/V ride
                        # along so the KNOWLEDGE_MOMENT carries the full mark
                        # (chronicle re-source 3-axis).
                        _km_stash = getattr(plugin, "_last_cited_use", None)
                        if not isinstance(_km_stash, dict):
                            _km_stash = {}
                            plugin._last_cited_use = _km_stash
                        _km_stash[_km_cid] = {
                            "needed": _km_needed,
                            "satisfied": _km_satisfied,
                            "cited": list(_km_cited),
                            "s_reply": _km_s_reply,
                            "e_reply": _km_e_reply,
                            "v_reply": _km_v_reply,
                        }
                        # Readable by the OVG tx-anchor (build_timechain_payload)
                        # so S rides the reply's TIMECHAIN_COMMIT.
                        plugin._last_sovereignty_s = _km_s_reply
                except Exception as _km_err:
                    logger.debug(
                        "[PostHook:G9] knowledge-moment signal skipped "
                        "(non-blocking): %s", _km_err)

                from titan_hcl.llm_pipeline.verifier import verify_post_async
                _ph_stage("ovg_pre_verify")
                _verified = await verify_post_async(
                    response_text,
                    channel="chat",
                    prompt=user_prompt,
                    injected_context=_injected_ctx,
                    output_verifier=_ovg,
                    bus=getattr(plugin, 'bus', None),
                    # Chat path default: append guard_message footer on pass,
                    # publish TIMECHAIN_COMMIT for verified outputs.
                    concurrent_sign=True,
                    # Arch §7 chat-TX shape kwargs (P2 closure):
                    user_id=_ovg_user_id,
                    chat_id=_ovg_chat_id,
                    turn_index=_ovg_turn_index,
                    topic_tags=_p3_topic_tags,
                    # Arch §7 normative content carry (Phase 3 D-SPEC-127):
                    tool_calls=_p3_tool_calls,
                    neuromods=_p3_snapshot.get("neuromods"),
                    embedding_hash=_p3_snapshot.get("embedding_hash", ""),
                    importance=_p3_snapshot.get("importance", 0.5),
                    # P3 (Synthesis Decision Authority) — anchor THIS reply's
                    # sovereignty S = 0.7·E + 0.3·V (computed in the KM block
                    # above, reset per-turn) on the reply's conversation-fork TX.
                    sovereignty=getattr(plugin, "_last_sovereignty_s", 0.0),
                )
                _ph_stage("ovg_post_verify")
                response_text = _verified.text
                if hasattr(run_output, 'content'):
                    run_output.content = response_text
                # Maker 2026-05-28 — do NOT await the OVG sign + TimeChain
                # commit on the /chat response path. verify_post_async already
                # ran the SAFETY check (which gated response_text above); the
                # sign_task is the audit-trail commit (Ed25519 + TimeChain
                # CONVERSATION-fork via a cross-process OVG-proxy RPC) that cost
                # hundreds-of-ms-to-seconds on every reply. Fire-and-forget it:
                # the response returns immediately, the commit completes async
                # on the loop. A done-callback fills the signature/merkle into
                # the VerifiedResult so later reads (and the SSE path) still
                # surface it; plugin._last_ovg_result holds the reference so the
                # Task is not GC'd before it resolves.
                if _verified.sign_task is not None:
                    def _attach_signed(_task, _vr=_verified):
                        try:
                            _signed = _task.result()
                            if _signed is not None and _signed.signed:
                                _vr.signature = _signed.signature
                                _vr.merkle_root = _signed.merkle_root
                                _vr.block_height = _signed.block_height
                                _vr.ovg_data.update({
                                    "signature": _signed.signature,
                                    "merkle_root": _signed.merkle_root,
                                    "block_height": _signed.block_height,
                                })
                                _vr.timechain_committed = True
                        except Exception as _sign_err:  # noqa: BLE001
                            logger.warning(
                                "[PostHook:OVG] background sign_task failed "
                                "(non-blocking): %s", _sign_err)
                    _verified.sign_task.add_done_callback(_attach_signed)
                _ovg_result = _verified
                # Store for /chat API to pick up (headers + structured body).
                # We attach the full VerifiedResult; agno_worker reads
                # .ovg_data + .signature + .merkle_root from it.
                plugin._last_ovg_result = _verified
            except Exception as _ovg_err:
                logger.warning(
                    "[PostHook:OVG] verify_post_async raised (non-blocking): %s",
                    _ovg_err)

        _ph_stage("after_ovg")
        # 1. Log to memory mempool (tagged with user_id)
        #
        # Phase 2 Chunk β (D-SPEC-78, 2026-05-18) — was the single biggest
        # /chat latency bottleneck on T1 production: `add_to_mempool` does
        # a sync SQLite write + cognee KG insert, awaited inline (~18.8s on
        # T1 swap-pressured profile, 4-6s on healthy host). User-visible
        # response is BLOCKED behind this work even though the memory log
        # is NOT load-bearing for the response itself.
        #
        # Fix: spawn as fire-and-forget asyncio.Task. The chat response
        # returns immediately; memory persistence completes in background.
        # Task exceptions logged via the done_callback. agno_worker.py
        # awaits all background tasks at SHUTDOWN to avoid orphans.
        # Felt-at-lived-time (§7.C — RFP_synthesis_engram_grounding): capture the
        # neuromod levels the turn was lived under so the promoted chat thought
        # carries felt into the synthesis felt axis. `_pre_chat_neuromods` is the
        # {name: level} snapshot taken in the PreHook (set at the V5 fetch, ~:1086);
        # copy it so a later turn's mutation can't alias this thought's felt. Empty
        # → None (graceful; no felt for this thought).
        _felt_snapshot = dict(getattr(plugin, "_pre_chat_neuromods", {}) or {})
        _felt_snapshot = _felt_snapshot or None
        # EEL A1 (INV-EEL-6): read-and-clear this turn's research-provenance flag
        # (set in the STATE_NEED_RESEARCH PreHook branch). Captured synchronously
        # here so the fire-and-forget task can't read a later turn's value.
        _acq_source = getattr(plugin, "_acquired_research_source", None)
        plugin._acquired_research_source = None
        _acq_tags = ["acquired:research"] if _acq_source else None
        if _acq_source:  # research turn — log the provenance tagging (low-freq)
            logger.info("[PostHook][EEL-A1] mempool node tagged acquired:research "
                        "(source=%s user=%s) → confirm window opened", _acq_source, user_id)

        async def _log_to_mempool_bg(user_prompt_v, response_text_v, user_id_v,
                                     neuromod_context_v=None,
                                     tags_v=None, source_v=None):
            try:
                await plugin.memory.add_to_mempool(
                    user_prompt_v, response_text_v,
                    user_identifier=user_id_v,
                    neuromod_context=neuromod_context_v,
                    tags=tags_v, source=source_v)
                logger.info(
                    "[PostHook][bg] Memory logged: user=%s prompt=%s... (%d chars, "
                    "felt=%s)", user_id_v, user_prompt_v[:40], len(response_text_v),
                    bool(neuromod_context_v))
            except Exception as e:
                # exc_info — this is the chat→mempool write (load-bearing for ~all
                # outer systems); a bare "%s" hid the add_to_mempool(tags=…)
                # signature drift as a one-liner for a long time. Full trace.
                logger.warning(
                    "[PostHook][bg] Memory logging failed: %s", e, exc_info=True)

        try:
            asyncio.create_task(_log_to_mempool_bg(
                user_prompt, response_text, user_id, _felt_snapshot,
                tags_v=_acq_tags, source_v=_acq_source))
        except Exception as e:
            # asyncio.create_task can only fail if there's no running loop;
            # the post_hook is invoked from agent.arun's loop so this is
            # defensive only.
            logger.warning(
                "[PostHook] Memory log task spawn failed: %s", e)

        # 1b. Record social interaction (Phase 13: Sage Socialite)
        social_graph = getattr(plugin, 'social_graph', None)
        if social_graph and user_id != "anonymous":
            try:
                # Estimate interaction quality from response length and engagement
                engagement = getattr(plugin, '_current_engagement_level', 'minimal')
                quality = 0.5  # neutral default
                if len(response_text) > 200:
                    quality += 0.1  # substantive response suggests good interaction
                if plugin._last_execution_mode == "Sovereign":
                    quality += 0.1  # high-confidence answers indicate good topic match
                elif plugin._last_execution_mode == "STATE_NEED_RESEARCH":
                    quality += 0.05  # research triggered = interesting question
                quality = min(1.0, quality)
                # rFP_social_graph_async_safety §5.2: async companion to stop
                # sync sqlite3.connect from blocking the FastAPI event loop on
                # the post-hook return path of /chat.
                await social_graph.record_interaction_async(user_id, quality=quality)
                logger.debug("[PostHook] Social interaction recorded: user=%s quality=%.2f", user_id, quality)

                # Update Events Teacher user valence from chat interaction.
                # EventsTeacherDB() + update_user_valence() both do sqlite3
                # writes (WAL mode, ~10-50ms). Wrap in to_thread so the post
                # hook doesn't block /v4/chat return on DB I/O.
                # See API_FIX_NEXT_SESSION.md (2026-04-14).
                try:
                    from titan_hcl.logic.events_teacher import EventsTeacherDB
                    from titan_hcl.core.state_registry import (
                        resolve_titan_id as _et_resolve)
                    _titan_id = _et_resolve()
                    # Sentiment from quality (0.5=neutral, 0.7=positive)
                    _chat_sentiment = (quality - 0.5) * 2  # Map 0-1 → -1 to +1
                    import asyncio as _asyncio_post
                    def _sync_valence_update():
                        _et_db = EventsTeacherDB()
                        _et_db.update_user_valence(
                            _titan_id, user_id, _chat_sentiment,
                            arousal=0.3, relevance=0.5,
                            contagion_type="chat")
                    await _asyncio_post.to_thread(_sync_valence_update)
                    # Invalidate resolver cache for updated user
                    _kur = getattr(plugin, '_known_user_resolver', None)
                    if _kur:
                        _kur.invalidate(user_id)
                except Exception as _uv_err:
                    logger.debug("[PostHook] User valence update failed: %s", _uv_err)
            except Exception as e:
                logger.warning("[PostHook] Social graph recording failed: %s", e)

        # 1b. CGN social consumer transition (neuromod delta = reward)
        try:
            _pre_nm = getattr(plugin, '_pre_chat_neuromods', {})
            if _pre_nm and user_id != "anonymous":
                # Compute neuromod delta reward
                _post_nm_levels = {}
                for _nm_name, _nm_mod in _post_mods.items():
                    _post_nm_levels[_nm_name] = _nm_mod.get("level", 0.5) if isinstance(_nm_mod, dict) else 0.5
                _social_reward = (
                    0.30 * (_post_nm_levels.get("DA", 0.5) - _pre_nm.get("DA", 0.5))
                    + 0.25 * (_post_nm_levels.get("5HT", 0.5) - _pre_nm.get("5HT", 0.5))
                    + 0.20 * (_post_nm_levels.get("Endorphin", 0.5) - _pre_nm.get("Endorphin", 0.5))
                    - 0.15 * abs(_post_nm_levels.get("NE", 0.5) - _pre_nm.get("NE", 0.5))
                    - 0.10 * max(0, _pre_nm.get("GABA", 0.5) - _post_nm_levels.get("GABA", 0.5))
                )
                # Send to language_worker's CGN via bus
                bus = getattr(plugin, 'bus', None)
                if bus and abs(_social_reward) > 0.001:
                    from titan_hcl.bus import make_msg
                    # Resolve user features from KnownUserResolver — wrap
                    # sqlite3 reads in to_thread (same pattern as pre_hook).
                    _kur = getattr(plugin, '_known_user_resolver', None)
                    if _kur:
                        import asyncio as _asyncio_kur2
                        _ku = await _asyncio_kur2.to_thread(
                            lambda: _kur.resolve(user_id))
                    else:
                        _ku = None
                    bus.publish(make_msg(
                        bus.CGN_SOCIAL_TRANSITION, "interface", "language", {
                            "user_id": user_id,
                            "reward": round(_social_reward, 5),
                            "neuromod_before": _pre_nm,
                            "neuromod_after": _post_nm_levels,
                            "quality": quality,
                            "familiarity": _ku.familiarity if _ku else 0.0,
                            "valence": _ku.social_valence if _ku else 0.0,
                            "interaction_count": _ku.interaction_count if _ku else 0,
                            "encounter_type": "chat",
                        }))
                    logger.info("[CGN:Social] Transition recorded: user=%s "
                                "reward=%+.4f quality=%.2f",
                                user_id[:20], _social_reward, quality)

                    # P4: Update social felt-tensor (EMA blend toward post-chat state)
                    try:
                        _kur_upd = getattr(plugin, '_known_user_resolver', None)
                        if _kur_upd and user_id != "anonymous":
                            # Build 30D state from post-chat neuromods + available data
                            _sft_30d = [
                                _post_nm_levels.get("DA", 0.5),
                                _post_nm_levels.get("5HT", 0.5),
                                _post_nm_levels.get("NE", 0.5),
                                _post_nm_levels.get("GABA", 0.5),
                                _post_nm_levels.get("ACh", 0.5),
                            ] + [0.5] * 25  # Remaining dims filled later by MSL
                            import asyncio as _asyncio_sft
                            await _asyncio_sft.to_thread(
                                _kur_upd.update_social_felt_tensor,
                                user_id, _sft_30d, 0.1)
                    except Exception:
                        pass  # Non-blocking

        except Exception as _cgn_soc_err:
            logger.debug("[PostHook] CGN social transition failed: %s", _cgn_soc_err)

        _ph_stage("after_memory_log")
        # (RL transition recording RETIRED — the offline-RL gatekeeper/recorder
        # subsystem is gone, RFP_synthesis_decision_authority P1. Execution-mode
        # routing is now the grounded router alone; nothing trains an IQL policy.)

        # Emit event for Observatory WebSocket
        try:
            if hasattr(plugin, 'event_bus') and plugin.event_bus:
                await plugin.event_bus.emit("chat_message", {
                    "user_prompt": user_prompt[:200],
                    "response": response_text[:200],
                    "mode": plugin._last_execution_mode,
                    "mood": plugin.mood_engine.get_mood_label() if plugin.mood_engine else "Unknown",
                })
        except Exception as e:
            logger.debug("[post_hook] event bus emit skipped: %s", e)

        _ph_stage("after_rl_record")
        # ── R5: Sovereign Reflex Scoring (TitanVM) ──────────────────
        # Run the reflex scoring micro-program to compute interaction reward.
        # This is the "ribosome" — pure math on StateRegister, no LLM.
        try:
            v3_core = getattr(plugin, 'v3_core', None) or plugin
            state_register = getattr(v3_core, 'state_register', None)
            bus = getattr(plugin, 'bus', None)

            if state_register:
                from titan_hcl.logic.titan_vm import TitanVM
                from titan_hcl.logic.vm_programs import get_program

                # Count reflexes from the last perceptual field (stored in pre-hook)
                last_pf = getattr(plugin, '_last_perceptual_field', None)
                reflexes_fired = len(last_pf.fired_reflexes) if last_pf else 0
                reflexes_succeeded = sum(
                    1 for r in (last_pf.fired_reflexes if last_pf else [])
                    if r.result and not r.error
                )

                # Build context from conversation features
                last_features = getattr(plugin, '_last_interface_input', {})
                vm_context = {
                    "intensity": last_features.get("intensity", 0.5),
                    "engagement": last_features.get("engagement", 0.5),
                    "valence": last_features.get("valence", 0.0),
                    "reflexes_fired": float(reflexes_fired),
                    "reflexes_succeeded": float(reflexes_succeeded),
                }

                # Plumb [titan_vm] toml — 2026-04-16. Previously TitanVM()
                # was built with no config, so max_stack_depth / max_instructions
                # from titan_params.toml were ignored (module-level constants won).
                vm_cfg = {}
                try:
                    full_cfg = getattr(plugin, "_full_config", None)
                    if isinstance(full_cfg, dict):
                        vm_cfg = full_cfg.get("titan_vm", {}) or {}
                except Exception:
                    vm_cfg = {}
                vm = TitanVM(state_register=state_register, bus=bus, config=vm_cfg)

                # Run main scoring program
                score_result = vm.execute(get_program("reflex_score"), context=vm_context)

                # Run valence modifier
                valence_result = vm.execute(get_program("valence_boost"), context=vm_context)

                # v2 (rFP_titan_vm_v2 Phase 1b): reward_blend_weight now lives —
                # weights the valence modifier contribution to total_reward.
                # Default 1.0 preserves prior (score + valence) sum behavior.
                blend_w = vm.get_reward_blend_weight()
                total_reward = max(0.0, min(1.0,
                    score_result.score + blend_w * valence_result.score))

                # v2 (Phase 1b): min_reward_threshold gates bus publish — skip
                # noise emissions when total_reward is near-zero, reducing bus
                # traffic without losing meaningful reflex signals.
                publish_gate = vm.get_min_reward_threshold()

                # Phase D (D-SPEC-116) — spirit_worker retirement. The old
                # REFLEX_REWARD→spirit handler had two legs: FilterDown training
                # (now Rust-owned — superseded) and neural_nervous_system
                # .record_outcome(reward) (FIREHOSE mode, program=None). Only the
                # NS leg survives; we repoint it onto the existing NS_REWARD bridge
                # (cognitive_worker → neural_nervous_system.record_outcome). NO
                # `program` field → record_outcome falls into firehose mode,
                # exactly preserving the old REFLEX_REWARD behavior (NS docstring
                # "If None, FIREHOSE mode... preserves the old REFLEX_REWARD
                # behavior for backward compat").
                if bus and total_reward > publish_gate:
                    from titan_hcl.bus import make_msg, NS_REWARD
                    reward_msg = make_msg(NS_REWARD, "titan_vm", "all", {
                        "reward": total_reward,
                        "source": "titan_vm",
                    })
                    bus.publish(reward_msg)

                logger.debug("[PostHook] TitanVM reflex score: %.3f (base=%.3f valence=%.3f) "
                            "fired=%d succeeded=%d vm_ms=%.1f",
                            total_reward, score_result.score, valence_result.score,
                            reflexes_fired, reflexes_succeeded,
                            score_result.duration_ms + valence_result.duration_ms)

                # R6: Record reflexes to ObservatoryDB + emit WebSocket events
                obs_db = getattr(plugin, "_observatory_db", None)
                event_bus_ws = getattr(plugin, "event_bus", None)
                last_features = getattr(plugin, '_last_interface_input', {})

                if last_pf and (obs_db or event_bus_ws):
                    for fired in last_pf.fired_reflexes:
                        # Extract per-source confidences
                        body_c = mind_c = spirit_c = 0.0
                        for sig in (fired.signals or []):
                            src = sig.get("source", "")
                            conf = sig.get("confidence", 0.0)
                            if src == "body":
                                body_c = max(body_c, conf)
                            elif src == "mind":
                                mind_c = max(mind_c, conf)
                            elif src == "spirit":
                                spirit_c = max(spirit_c, conf)

                        if obs_db:
                            try:
                                obs_db.record_reflex(
                                    reflex_type=fired.reflex_type.value,
                                    combined_confidence=fired.combined_confidence,
                                    body_confidence=body_c,
                                    mind_confidence=mind_c,
                                    spirit_confidence=spirit_c,
                                    fired=True,
                                    succeeded=bool(fired.result and not fired.error),
                                    duration_ms=fired.duration_ms,
                                    error=fired.error or "",
                                    stimulus_topic=last_features.get("topic", ""),
                                    stimulus_intensity=last_features.get("intensity", 0.0),
                                    vm_reward=total_reward,
                                )
                            except Exception as e:
                                logger.debug("[post_hook] reflex DB recording skipped: %s", e)

                        if event_bus_ws:
                            try:
                                asyncio.create_task(event_bus_ws.emit("reflex_fired", {
                                    "reflex_type": fired.reflex_type.value,
                                    "confidence": round(fired.combined_confidence, 3),
                                    "succeeded": bool(fired.result and not fired.error),
                                    "duration_ms": round(fired.duration_ms, 1),
                                }))
                            except Exception as e:
                                logger.debug("[post_hook] reflex WS emit skipped: %s", e)

                # Emit reward event
                if event_bus_ws and total_reward > 0.01:
                    try:
                        asyncio.create_task(event_bus_ws.emit("reflex_reward", {
                            "reward": round(total_reward, 3),
                            "base_score": round(score_result.score, 3),
                            "valence_mod": round(valence_result.score, 3),
                            "reflexes_fired": reflexes_fired,
                        }))
                    except Exception as e:
                        logger.debug("[post_hook] reward WS emit skipped: %s", e)

        except Exception as e:
            logger.debug("[PostHook] TitanVM scoring failed: %s", e)

        _ph_stage("after_reflex_scoring")
        # ── Step 5: Interface Input Extraction (async, nonblocking) ──
        # Extract patterns from the user's message and publish to bus.
        # Body/Mind workers absorb these on their next tick, so the
        # coloring for the NEXT response reflects this conversation.
        extractor = _get_input_extractor()
        if extractor:
            try:
                signals = extractor.extract(user_prompt, user_id)
                plugin._last_interface_input = signals  # Cache for output coloring

                # Publish to bus if V3 is active
                bus = getattr(plugin, 'bus', None)
                if bus:
                    from titan_hcl.bus import make_msg, INTERFACE_INPUT, CONVERSATION_STIMULUS
                    msg = make_msg(INTERFACE_INPUT, "interface", "all", signals)
                    bus.publish(msg)

                    # Also publish CONVERSATION_STIMULUS so workers update
                    # their reflex Intuition state for next interaction
                    stimulus_payload = dict(signals)
                    stimulus_payload["message"] = user_prompt[:500]
                    stimulus_payload["threat_level"] = _estimate_threat_level(user_prompt, signals)
                    stim_msg = make_msg(CONVERSATION_STIMULUS, "interface", "all", stimulus_payload)
                    bus.publish(stim_msg)

                    logger.debug("[PostHook] INTERFACE_INPUT + CONVERSATION_STIMULUS published: valence=%.2f topic=%s",
                                signals["valence"], signals["topic"])
            except Exception as e:
                logger.debug("[PostHook] Interface input extraction failed: %s", e)

        _ph_stage("exit_post_hook")

    return titan_post_hook
