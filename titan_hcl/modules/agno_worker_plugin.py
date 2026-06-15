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
                            shared across pre_hook / post_hook within a
                            single chat — Agno serializes chat dispatch)
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

import logging
import os
from typing import Any, Optional

from titan_hcl.bus import DivineBus

logger = logging.getLogger(__name__)


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
        self.state_register: Any = None

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
        # MetabolismProxy may require a network or guardian arg — pass None
        # and rely on the proxy's defensive init. Adjust signature per the
        # actual MetabolismProxy.__init__ if needed during smoke test.
        try:
            return self._proxy("metabolism", lambda b: MetabolismProxy(b))
        except TypeError:
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
                _cfg.update(self._full_config.get("inference", {}) or {})
                _cfg.update(self._full_config.get("stealth_sage", {}) or {})
                _sage = StealthSageResearcher(_build_sage_config(_cfg))
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
