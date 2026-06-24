"""
agno_worker_plugin — WorkerPlugin shim for agno_worker subprocess.

D-SPEC-72 / SPEC v1.17.0. The `WorkerPlugin` class exposes the same
attribute surface that `agno_hooks.py` + `agno_tools.py` + `agno_guardrails.py`
expect on the `plugin` parameter, backed by:

  - bus-callable proxies   (memory / social_graph / neuromod / mind /
                            recorder / soul / studio / metabolism / agency /
                            sage_researcher / output_verifier — via bus QUERY)
  - worker-local caches    (replaces parent's plugin._last_X / _pre_chat_X /
                            _current_X / _pending_X / _limbo_mode attrs;
                            flow pre_hook → post_hook within a single chat.
                            ⚠ Chat dispatch is NO LONGER serial: RFP §7.B0
                            (B0-dispatch) runs the chats concurrently on one
                            asyncio loop, so the PER-TURN subset of these
                            fields is request-scoped via a ContextVar bag —
                            see `_REQUEST_SCOPED_FIELDS` / `enter_request_scope`
                            at the top of this module. Singletons + cross-turn
                            accumulators + chat_id-keyed dicts stay plain attrs.)
  - inline stateless tools (maker_engine + _skill_registry — Q4 LOCKED;
                            constructed once at worker boot)

Replaces the in-parent `TitanHCL` reference for the agno_worker context.
The hook code in `agno_hooks.py` is UNCHANGED at the body level — it sees
`plugin` regardless of context (parent TitanHCL OR worker WorkerPlugin).
Minimum-diff carve, maximum architectural correctness.

This pattern mirrors the approach taken in past worker carves (cognitive_worker,
social_graph_worker, studio_worker) where in-process objects are replaced by
bus-callable proxies behind the same attribute surface.
"""
from __future__ import annotations

import contextvars
import logging
import os
from typing import Any, Optional

from titan_hcl.bus import DivineBus
from titan_hcl.params import get_params

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Request-scoped per-turn state (RFP_load_adaptive_inference_routing §7.B0)
# ════════════════════════════════════════════════════════════════════════
#
# B0-dispatch (agno_worker) runs ONE asyncio event loop in ONE daemon thread
# and schedules each chat via `run_coroutine_threadsafe`, so multiple chats
# now overlap. asyncio is single-threaded cooperative concurrency — the hazard
# is NOT a data race but INTERLEAVING across `await agent.arun()`: chat A sets
# `plugin._current_user_id = alice`, yields at the await, chat B sets it to
# `bob`, then A's PostHook reads `bob` → every synthesis record (presence /
# conversation episode / maker-fact / reasoning) tagged with the wrong user.
#
# Fix: the ~38 fields that live for exactly ONE turn (set on entry / in the
# PreHook or a tool, read later the same turn in the PostHook or
# `_handle_chat_request` tail) are backed by a per-task ContextVar bag instead
# of plain plugin attributes. Each chat coroutine installs its OWN bag via
# `enter_request_scope()` before its first await; asyncio.Task copies the
# context at creation so the `.set()` only affects that task. Outside any chat
# task (boot, keepalive thread, tests, non-chat handlers) `_request_bag()`
# returns the process-global fallback bag → byte-identical to the
# pre-concurrency single-flight behavior.
#
# Per-field classification (VERIFIED in code 2026-06-24 — traced, not inferred;
# supersedes the RFP §7.B0 list where they differ, per the trace-over-RFP rule):
#   • MIGRATE (scalar per-turn, read across the await → contaminates): the 11
#     `_current_*`, the 16 scalar `_last_*`, and the per-turn `_pre_chat_*` /
#     telemetry / research / stream fields enumerated below.
#   • LEAVE (already safe-by-key): `_last_surfaced_items` / `_last_cited_use`
#     are dicts keyed by f"{user_id}:{session_id}"; once `_current_user_id` is
#     task-local their keys are correct-per-task and never collide across users.
#     (Migrating them would break the worker's cross-call `.pop(chat_id)`.)
#   • LEAVE (cross-turn accumulators, shared BY DESIGN): `_p2_gibberish_baseline`
#     / `_p2_gibberish_turn_count` (every-N-turns calibration probe) — scoping
#     would reset them each turn and break the probe.
#   • LEAVE (singletons/caches set once at boot): proxies, `engine_recall`,
#     `_tier_classifier_cache`, `_verified_context_builder`, `_chat_ctx`,
#     `_cited_use_detector`, resolvers, `_limbo_mode`, etc.

# name → default value (matches the pre-B0 WorkerPlugin.__init__ literals; any
# field absent from __init__ defaulted None — its readers all use getattr/None).
_REQUEST_SCOPED_FIELDS: dict[str, Any] = {
    # ── the 11 `_current_*` (set in agno_worker._handle_chat_request:642-674
    #    + PreHook) ──
    "_current_user_id": None,
    "_current_session_id": None,
    "_current_channel": None,
    "_current_is_maker": False,
    "_current_did_hash": None,
    "_current_ip_hash": None,
    "_current_engagement_level": 0.0,
    "_current_tool_intent": None,
    "_current_chat_tier": None,
    "_current_chat_model_class": None,
    "_current_chat_features": None,
    # ── the 16 scalar `_last_*` (reset/set each turn, read in PostHook /
    #    handler tail) ──
    "_last_perceptual_field": None,
    "_last_research_sources": [],   # always wholesale-assigned; fresh copy on get
    "_last_prompt_vec": None,
    "_last_prompt_text": None,
    "_last_router_decision": None,
    "_last_recall_score": 0.0,
    "_last_outer_decision": None,
    "_last_reasoning_id": None,
    "_last_composite_match": None,
    "_last_matched_skill_id": None,
    "_last_execution_mode": "",
    "_last_retrieval_sample": None,
    "_last_tool_activity": None,
    "_last_sovereignty_s": 0.0,
    "_last_ovg_result": None,
    "_last_interface_input": None,
    # ── per-turn `_pre_chat_*` / telemetry / research / stream fields the RFP
    #    list missed but which trace as per-turn-set-then-read-across-await ──
    "_pre_chat_user_id": "",
    "_pre_chat_ku": None,
    "_pre_chat_neuromods": {},       # fresh copy on get
    "_pre_chat_recent_presence": None,
    "_pre_chat_presence_record": None,
    "_telemetry_trigger_id": None,
    "_acquired_research_source": None,
    "_stream_progress_ctx": None,
    "_research_call_count": 0,
    "_force_research_topic": None,
    "_dk4_concept_name": None,       # write-only today; scoped for uniformity
}

# The per-task state bag. None → not inside a chat request scope → fall back to
# the process-global bag (single-flight behavior, unchanged).
_REQUEST_STATE: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "agno_request_state", default=None)
_GLOBAL_FALLBACK_BAG: dict[str, Any] = {}


def _request_bag() -> dict:
    bag = _REQUEST_STATE.get()
    return bag if bag is not None else _GLOBAL_FALLBACK_BAG


def enter_request_scope() -> None:
    """Install a FRESH per-task state bag for the current chat coroutine.

    MUST be called at the top of each chat coroutine (`_handle_chat_request`,
    `_handle_chat_stream_request`, `_handle_dream_inbox_replay`) BEFORE its
    first `await`, so concurrent chats never share the per-turn fields. Because
    asyncio.Task copies the context at creation, this `.set()` is visible only
    to the calling task and its descendants. Idempotent per task.
    """
    _REQUEST_STATE.set({})


class _RequestScopedAttr:
    """Data descriptor backing a per-turn WorkerPlugin field with the
    `_REQUEST_STATE` ContextVar bag (RFP §7.B0 B0-state).

    Reads/writes hit the calling task's bag under a chat request scope, else
    the process-global fallback bag. There is exactly one WorkerPlugin per
    worker process, so keying the bag by attribute name alone is sufficient.
    """

    __slots__ = ("name", "default")

    def __init__(self, name: str, default: Any = None):
        self.name = name
        self.default = default

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        bag = _request_bag()
        if self.name in bag:
            return bag[self.name]
        # Return a FRESH copy of mutable defaults so a wholesale-reassign or an
        # in-place mutation can never leak through the shared default object.
        d = self.default
        if isinstance(d, list):
            return list(d)
        if isinstance(d, dict):
            return dict(d)
        return d

    def __set__(self, obj, value):
        _request_bag()[self.name] = value

    def __delete__(self, obj):
        _request_bag().pop(self.name, None)


def extract_sources_from_findings(findings: str) -> list[str]:
    """Heuristic source extraction from [SAGE_RESEARCH_FINDINGS] block.

    Lifted verbatim from `titan_hcl/__init__.py:912 TitanHCL._extract_sources_from_findings`
    so both legacy parent (TitanHCL) and worker (WorkerPlugin) call the
    same canonical implementation. Used to populate research_sources metadata
    in post_resolution_hook without an extra async call or shared mutable state.

    Web (SearXNG + Crawl4AI) is always the base when findings are non-empty;
    X and Document blocks are detected independently — sources are additive.

    Returns: list[str] of source-type strings e.g. ["Web", "Document", "X"].
    """
    if not findings:
        return []
    sources = ["Web"]
    if "[X_SEARCH_RESULTS" in findings:
        sources.append("X")
    if "Document Topic:" in findings:
        sources.append("Document")
    return sources


class WorkerPlugin:
    """Plugin shim for the agno_worker subprocess context.

    Exposes the surface that agno_hooks / agno_tools / agno_guardrails
    expect on the `plugin` parameter. All proxy-backed attributes go
    through the bus; all state-cache attributes are worker-local mutable.
    """

    def __init__(
        self,
        bus_client: DivineBus,
        *,
        config: Optional[dict[str, Any]] = None,
        titan_id: Optional[str] = None,
    ):
        self.bus = bus_client
        self._full_config: dict[str, Any] = config or {}
        self._titan_id = titan_id or os.environ.get("TITAN_ID", "T1")

        # ── Lazy proxy cache (constructed on first .X access) ──
        self._proxy_cache: dict[str, Any] = {}

        # ── Worker-local mutable state (replaces plugin._X parent attrs) ──
        # State that flows pre_hook → post_hook within a single chat run.
        self._limbo_mode: bool = False
        self._current_user_id: Optional[str] = None
        self._current_engagement_level: float = 0.0
        self._pre_chat_user_id: str = ""
        self._pre_chat_ku: Any = None
        self._pre_chat_neuromods: dict[str, float] = {}
        self._last_research_sources: list[str] = []
        self._last_observation_vector: Optional[list[float]] = None
        self._last_execution_mode: str = ""
        self._last_perceptual_field: Any = None
        self._last_transition_id: int = -1
        self._last_interface_input: Any = None
        self._last_ovg_result: Any = None
        self._last_sol_balance: Optional[float] = None
        self._last_energy_state: str = "UNKNOWN"
        self._pending_self_composed: str = ""
        self._pending_self_composed_confidence: float = 0.0
        # Lazy-init resolvers
        self._known_user_resolver: Any = None
        self._verified_context_builder: Any = None

        # ── Inline stateless tools (Q4 LOCKED — constructed at boot) ──
        self.maker_engine: Any = None
        self._skill_registry: Any = None

        # ── v3_core back-compat ──
        # agno_hooks.py:217 + :1821 do `v3_core = getattr(plugin, 'v3_core', None) or plugin`.
        # Pointing v3_core at self preserves the alias semantics — the legacy
        # parent had v3_core as the "Phase C plugin under l0_rust=true" backing
        # object; here we're already that, so the alias is self-referential.
        self.v3_core = self

        # ── Optional collectors (best-effort under WorkerPlugin) ──
        # agno_hooks.py:218 / :235-274 access these via getattr-with-None
        # fallback. The reflex/state-register infra lives in the parent
        # process today; under Phase C worker isolation it's accessed via
        # SHM reads or bus.QUERY. For C2, leave None — hooks degrade
        # gracefully. Wiring these is a Chunk-K post-C2 optimization.
        self.reflex_collector: Any = None
        # state_register RETIRED (RFP_g18 §7.B) — getattr(self, "state_register",
        # None) returns None; reflex/felt readers already degrade gracefully.

    # ────────────────────────────────────────────────────────────────
    # Bus-callable proxies (lazy property accessors)
    # ────────────────────────────────────────────────────────────────

    def _proxy(self, name: str, constructor):
        """Lazy proxy resolution + caching."""
        if name not in self._proxy_cache:
            try:
                self._proxy_cache[name] = constructor(self.bus)
            except Exception as e:
                logger.warning(
                    "[WorkerPlugin] %s proxy construction failed: %s — "
                    "hooks accessing it will see None", name, e,
                )
                self._proxy_cache[name] = None
        return self._proxy_cache[name]

    @property
    def memory(self):
        from titan_hcl.proxies.memory_proxy import MemoryProxy
        return self._proxy("memory", lambda b: MemoryProxy(b, None))

    @property
    def social_graph(self):
        from titan_hcl.proxies.social_graph_proxy import SocialGraphProxy
        return self._proxy("social_graph", lambda b: SocialGraphProxy(b, None))

    @property
    def mind(self):
        from titan_hcl.proxies.mind_proxy import MindProxy
        return self._proxy("mind", lambda b: MindProxy(b, None))

    # recorder proxy RETIRED with the offline-RL subsystem
    # (RFP_synthesis_decision_authority P1).

    # spirit proxy property retired Phase B.5 (2026-05-18) — callers now
    # use plugin._shm_reader_bank.compose_trinity() / .compose_v4_state()
    # (Rust L0+L1 canonical SHM-direct read). Accessing plugin.spirit
    # returns None so any straggler caller in worker context degrades
    # gracefully (same pattern as plugin.consciousness).
    @property
    def spirit(self):
        return None

    @property
    def _shm_reader_bank(self):
        if "_shm_reader_bank" not in self._proxy_cache:
            from titan_hcl.api.shm_reader_bank import ShmReaderBank
            self._proxy_cache["_shm_reader_bank"] = ShmReaderBank()
        return self._proxy_cache["_shm_reader_bank"]

    @property
    def studio(self):
        from titan_hcl.proxies.studio_proxy import StudioProxy
        return self._proxy("studio", lambda b: StudioProxy(b, None))

    @property
    def metabolism(self):
        from titan_hcl.proxies.metabolism_proxy import MetabolismProxy
        # MetabolismProxy.__init__(bus, guardian). The agno subprocess has no
        # in-process Guardian, so pass None — guardian is only touched in
        # _ensure_started(), which degrades gracefully on None (same contract as
        # the sibling proxies above, all constructed `(b, None)`).
        # BUGFIX 2026-06-24: the prior `MetabolismProxy(b)` (1 arg) raised
        # TypeError on EVERY access; `_proxy()` caught it internally and cached
        # None, so the 2-arg `except TypeError` fallback was DEAD CODE and
        # `plugin.metabolism` was permanently None. One correct call, no retry.
        return self._proxy("metabolism", lambda b: MetabolismProxy(b, None))

    @property
    def agency(self):
        from titan_hcl.proxies.agency_proxy import AgencyProxy
        return self._proxy("agency", lambda b: AgencyProxy(b))

    @property
    def soul(self):
        # soul is L0/L1 kernel-level per SPEC §10.A B1 — not extracted; for
        # WorkerPlugin we expose a bus.QUERY-backed shim. agno_hooks.py
        # only accesses `plugin.soul.get_active_directives()` — handled
        # via _SoulShim.
        if "soul" not in self._proxy_cache:
            self._proxy_cache["soul"] = _SoulShim(self.bus)
        return self._proxy_cache["soul"]

    # mood_engine — parent maps this to the mind proxy (get_mood_label()).
    # gatekeeper proxy RETIRED with the offline-RL subsystem
    # (RFP_synthesis_decision_authority P1).
    @property
    def mood_engine(self):
        return self.mind

    # consciousness is NOT a separate L2 worker today — it lives in the parent
    # process. For agno_worker context, hooks accessing it get None and degrade
    # gracefully (access sites in agno_hooks.py use getattr with None fallback).
    @property
    def consciousness(self):
        return None

    @property
    def sage_researcher(self):
        # The chat research lane (agno_hooks STATE_NEED_RESEARCH) runs in THIS
        # agno_worker process, so it needs a real researcher HERE. Construct
        # StealthSageResearcher IN-PROCESS — the same decision as _output_verifier
        # below (worker-context bus-RPC to knowledge_worker is unreliable — RESPONSE
        # delivery to agno_worker's recv_queue times out). Stateless (SearXNG httpx
        # + LLM-distill, no DB) → safe + cheap in-process. Lazy + cached. Returns
        # None on init failure → the research lane degrades to internal-recall-only.
        # 2026-06-15 FIX: this property previously hardcoded `return None`, so every
        # chat research request (router → STATE_NEED_RESEARCH) silently degraded to
        # empty ("research pipeline returned no results") — the web-fetch backend was
        # healthy the whole time; the researcher object was just never wired here.
        if not hasattr(self, "_local_sage_instance"):
            self._local_sage_instance = None
            try:
                from titan_hcl.modules.knowledge_worker import (
                    _build_sage_config, _wire_ollama_cloud)
                from titan_hcl.logic.sage.researcher import StealthSageResearcher
                _cfg: dict[str, Any] = {}
                _cfg.update(get_params("inference") or {})
                _cfg.update(get_params("stealth_sage") or {})
                _sage_cfg = _build_sage_config(_cfg)
                # Research DISTILLATION (turning scraped pages into an answer) goes
                # through /v4/llm-distill, which StealthSage reads from config["api"]
                # (internal_key + port). _build_sage_config drops the api section, so
                # without this the distiller logs "No internal_key … distillation
                # disabled" → research gathers data but returns EMPTY (2026-06-15).
                # The internal_key lives in secrets.toml; this worker's _full_config
                # may not carry the merged secret, so resolve it canonically
                # (load_titan_config merges secrets) when _full_config lacks it.
                _api_cfg = dict(get_params("api") or {})
                if not _api_cfg.get("internal_key"):
                    try:
                        _merged_api = (get_params("api") or {})
                        if _merged_api.get("internal_key"):
                            _api_cfg = _merged_api
                    except Exception:
                        pass
                _sage_cfg["api"] = _api_cfg
                _sage = StealthSageResearcher(_sage_cfg)
                _wire_ollama_cloud(_sage, _cfg)
                self._local_sage_instance = _sage
                logger.info(
                    "[agno_plugin] in-process StealthSage initialized "
                    "(searxng=%s)", getattr(_sage, "_searxng_host", "?"))
            except Exception:
                logger.warning(
                    "[agno_plugin] in-process sage init failed — research degrades "
                    "to internal recall only", exc_info=True)
        return self._local_sage_instance

    # ── output_verifier — LOCAL instance (D-SPEC-75 hotfix 2026-05-18) ──
    #
    # Pre-2026-05-18: this returned an `OutputVerifierProxy` that routed
    # verify_safety / sign_and_commit over the bus to `output_verifier_worker`.
    # Production T3 diagnosis revealed the worker-side bus-RPC path is
    # unreliable from inside agno_worker (the `_WorkerBusClient` does not
    # actually register subscriber names on subscribe; even after fixing
    # the src-override hack, RESPONSE delivery to agno_worker's recv_queue
    # times out at 5s for reasons that need deeper investigation). The
    # output_verifier_worker correctly processed the request (verify_safety
    # PASS in 8.4ms per its journalctl log) — but agno_worker's
    # request_async loop never saw the reply.
    #
    # Decision: agno_worker constructs OutputVerifier IN-PROCESS — same
    # pattern as the inference module (titan_hcl.inference imported
    # directly in agno_worker). OVG is fast (50-300ms safety + ~ms signing)
    # and stateless modulo the keypair file + TimeChain reader — both
    # safe to use in-process. No bus hop, no race conditions, no proxy.
    #
    # The output_verifier_worker subprocess continues to serve OTHER
    # callers (social_x_gateway from social_worker, studio_worker, etc.)
    # via the canonical bus path. Only agno_worker (which is the busiest
    # OVG caller — every chat goes through it) sidesteps the proxy.
    #
    # Future: when worker-context bus-RPC is hardened (separate reply
    # queue per worker, subscribed names actually registered, etc.) we
    # can revisit. Tracked in a follow-up rFP.
    @property
    def _output_verifier(self):
        if not hasattr(self, "_local_ovg_instance"):
            from titan_hcl.logic.output_verifier import OutputVerifier
            from titan_hcl.core.state_registry import resolve_titan_id
            titan_id = resolve_titan_id()
            self._local_ovg_instance = OutputVerifier(titan_id=titan_id)
        return self._local_ovg_instance

    # ── neuromod — no dedicated proxy class in the codebase today; the
    # hooks code accesses plugin.neuromod for boost/get_stats; expose
    # a bus.QUERY-backed shim. ──
    @property
    def neuromod(self):
        if "neuromod" not in self._proxy_cache:
            self._proxy_cache["neuromod"] = _NeuromodShim(self.bus)
        return self._proxy_cache["neuromod"]

    # ── _proxies dict back-compat ──
    # Phase B.5: spirit entry retired (None) — callers migrate to
    # plugin._shm_reader_bank.compose_trinity / .compose_v4_state.
    @property
    def _proxies(self) -> dict[str, Any]:
        return {
            "memory": self.memory,
            "social_graph": self.social_graph,
            "mind": self.mind,
            "spirit": None,  # retired Phase B.5 — see _shm_reader_bank
            "studio": self.studio,
            "metabolism": self.metabolism,
            "agency": self.agency,
            "soul": self.soul,
            "mood_engine": self.mood_engine,
            "neuromod": self.neuromod,
        }

    # ────────────────────────────────────────────────────────────────
    # Methods
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_sources_from_findings(findings: str) -> list[str]:
        """Delegate to the canonical implementation in this module.

        Preserves the legacy `plugin._extract_sources_from_findings(...)`
        call surface used by `agno_hooks.py:1471` + `agno_tools.py:56`.
        """
        return extract_sources_from_findings(findings)


# ── RFP §7.B0 B0-state — install the per-turn request-scoped descriptors ──
# Routes the ~38 per-turn fields through the `_REQUEST_STATE` ContextVar bag so
# concurrent chats (B0-dispatch) never cross-contaminate. Done after the class
# body so every existing `plugin._current_user_id = …` / read site (136 across
# the codebase) is unchanged — the descriptor intercepts get/set transparently.
# `__init__`'s own assignments of these names run OUTSIDE a request scope, so
# they seed the process-global fallback bag with the same defaults.
for _rsf_name, _rsf_default in _REQUEST_SCOPED_FIELDS.items():
    setattr(WorkerPlugin, _rsf_name, _RequestScopedAttr(_rsf_name, _rsf_default))
del _rsf_name, _rsf_default


# ────────────────────────────────────────────────────────────────────
# Bus-RPC shims for objects without dedicated proxy classes
# ────────────────────────────────────────────────────────────────────

class _SoulShim:
    """Bus-RPC shim for plugin.soul.get_active_directives() (used at
    agno_hooks.py:651). Soul lives at L0/L1 (kernel-level per SPEC §10.A B1)
    and is accessible via bus.QUERY action='get_active_directives' to the
    kernel handler. Returns [] gracefully on RPC failure."""

    def __init__(self, bus_client: DivineBus):
        self._bus = bus_client

    async def get_active_directives(self) -> list[dict]:
        try:
            reply = await self._bus.request_async(
                "agno_worker", "soul",
                {"action": "get_active_directives"},
                5.0,
                self._bus.subscribe("agno_worker_soul", reply_only=True),
            )
            if reply is None:
                return []
            return (reply.get("payload") or {}).get("directives", []) or []
        except Exception as e:
            logger.debug("[SoulShim] get_active_directives failed: %s", e)
            return []


class _NeuromodShim:
    """Bus-RPC shim for neuromod proxy access. hooks access surface is
    narrow — mostly get_stats / nudge. Other workers fully migrated to
    neuromod_worker per D-SPEC-54; we route via bus."""

    def __init__(self, bus_client: DivineBus):
        self._bus = bus_client

    def get_stats(self) -> dict:
        # Stats are SHM-read in the real neuromod_proxy; the shim returns
        # empty dict so hooks degrade gracefully. The actual stats lookup
        # in hooks happens via the dashboard.cached_coordinator path which
        # reads SHM directly.
        return {}
