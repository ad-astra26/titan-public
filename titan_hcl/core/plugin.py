"""
titan_hcl/core/plugin.py — TitanHCL (L1-L3 coordinator).

Thin coordinator holding a `kernel: TitanKernel` reference plus the L2/L3
state and loops:
  - Proxies (memory/rl/llm/body/mind/spirit/media/timechain + 13 @property facades)
  - Agency subsystem (autonomous action pipeline)
  - OutputVerifier (external output gate)
  - Outer trinity collector
  - EventBus + ObservatoryDB + observatory FastAPI app
  - Agno agent
  - Dream inbox (API-side message queue during dream cycles)
  - Parent async loops: agency, v4_event_bridge, trinity_snapshot,
    publish_outer_sources (sovereignty/meditation/social_engagement loops
    retired to workers — D-SPEC-57/60/104)

Uses the kernel for all L0 services (bus, guardian, state_register,
registry_bank, soul, network, identity). The kernel never restarts;
this coordinator may be replaced or re-attached during Phase B shadow
core swap.

Compat @property facade (bus, guardian, soul, _full_config, ...) makes
TitanHCL duck-type-identical to the legacy TitanCore for dashboard +
agent code. Zero dashboard code changes required.

This commit (#3 — plugin skeleton + module registration + proxies)
lands the __init__, compat properties, _register_modules (380-line
module catalog lifted from v5_core.py:317-696), and _create_proxies.
Wire helpers + observatory + agency + async loops + boot orchestration
arrive in commits 4-6.

See:
  - titan-docs/rFP_microkernel_v2_shadow_core.md §A.1
  - titan-docs/PLAN_microkernel_phase_a_s3.md §2.2 + §3 D1+D9+D10
  - titan_hcl/core/kernel.py (the L0 paired class)
"""
import asyncio
import logging
import os
import time
from typing import Optional

from titan_hcl.bus import (
    ACTION_RESULT,
    AGENCY_READY,
    AGENCY_STATS,
    ASSESSMENT_STATS,
    EPOCH_TICK,
    IMPULSE,
    OUTER_OBSERVATION,
    make_msg,
)
from titan_hcl.core.kernel import TitanKernel
from titan_hcl.guardian_hcl import ModuleSpec
from titan_hcl.supervision import (
    Dependency,
    DependencyAction,
    DependencyKind,
    DependencySeverity,
)
from titan_hcl.utils.silent_swallow import swallow_warn
from titan_hcl import bus

logger = logging.getLogger(__name__)


# PERSISTENCE_BY_DESIGN: TitanHCL._proxies / _agency / _*_mode fields
# are runtime bootstrap state — constructed from kernel + config at boot.
# Proxy objects are not self-owned state to persist.
class TitanHCL:
    """
    L1-L3 Coordinator — owns proxies, agency, observatory, agno, dream inbox.

    Usage (from scripts/titan_hcl.py flag-branch per PLAN §4.4):
        kernel = TitanKernel(wallet_path)
        plugin = TitanHCL(kernel)
        await plugin.boot()  # orchestrates kernel.boot() + module wiring

    Compat shape: dashboard.py + agent code treat TitanHCL as the
    "plugin root" with @property accessors (bus, guardian, soul, memory,
    metabolism, ...) that match the legacy TitanCore surface. This is
    intentional — S3 preserves full duck-type compatibility so zero
    downstream code changes are needed during the cutover.
    """

    def __init__(self, kernel: TitanKernel):
        self.kernel = kernel

        # ── Proxy stubs (populated by _create_proxies during boot) ──
        self._proxies: dict[str, object] = {}

        # ── Agency Module (Step 7) ─────────────────────────────────
        self._agency = None
        self._agency_assessment = None
        # _interface_advisor parent instance RETIRED v1.8.5 §4.H (D-SPEC-59,
        # 2026-05-15) — InterfaceAdvisor now lives in interface_advisor_worker
        # subprocess; parent reads rate state from SHM via
        # InterfaceAdvisorStateReader (sub-µs G18, 100ms cache).
        self._interface_advisor_reader = None

        # ── Output Verification Gate (security for all external outputs) ──
        # L3 §A.8.3: when microkernel.a8_output_verifier_subprocess_enabled
        # is true, the OV runs in a subprocess (output_verifier_worker) and
        # parent uses an OutputVerifierProxy that bus.request()s the worker.
        # When false (default), parent retains the local OutputVerifier
        # instance — byte-identical to pre-A.8.3 behavior.
        self._output_verifier = None
        _ov_subproc_enabled = bool(
            kernel.config.get("microkernel", {}).get(
                "a8_output_verifier_subprocess_enabled", False))
        if _ov_subproc_enabled:
            try:
                from titan_hcl.proxies.output_verifier_proxy import OutputVerifierProxy
                self._output_verifier = OutputVerifierProxy(self.bus)
                logger.info("[TitanHCL] OutputVerifier using subprocess proxy "
                            "(A.8.3 flag enabled)")
            except Exception as _ovg_err:
                logger.warning("[TitanHCL] OutputVerifierProxy init failed: %s",
                               _ovg_err)
        else:
            try:
                from titan_hcl.logic.output_verifier import OutputVerifier
                _tc_dir = os.path.join("data", "timechain")
                _titan_id = kernel.config.get("info_banner", {}).get("titan_id") or kernel.titan_id
                _wallet_path = kernel.config.get("network", {}).get(
                    "wallet_keypair_path", "data/titan_identity_keypair.json")
                self._output_verifier = OutputVerifier(
                    titan_id=_titan_id, data_dir=_tc_dir, keypair_path=_wallet_path)
            except Exception as _ovg_err:
                logger.warning("[TitanHCL] OutputVerifier init failed: %s", _ovg_err)

        # ── State ────────────────────────────────────────────────────
        self._last_execution_mode = "Shadow"
        self._is_meditating = False
        self._background_tasks_started = False
        self._observatory_app = None
        self._agent = None

        # ── Phase 2.5.E SocialXGateway reader exposure (2026-05-23 bug-fix) ──
        # The `/v4/community-engagement-stats` endpoint + the kernel_rpc
        # EXPOSED_METHODS allowlist both reference
        # `plugin._social_x_gateway_reader.get_community_engagement_stats`,
        # but this attribute was never bound in __init__. The Maker note in
        # core/kernel.py:409-417 documented the EXPOSED_METHODS gap (added
        # 2026-05-12) but missed that the underlying attribute itself was
        # also never created. Without binding, _resolve_method returns None
        # and kernel_rpc replies with AttributeError, surfacing as
        # `[Dashboard] /v4/community-engagement-stats error: [AttributeError]
        # '_social_x_gateway_reader.get_community_engagement_stats' not
        # resolvable on plugin`. T1 owns the social_x.db locally; T2/T3
        # consume via HTTP (see outer_source_assembly.py:295). Lazy-import
        # to avoid pulling SocialXGateway at boot for Titans that never hit
        # the endpoint.
        self._social_x_gateway_reader = None

        # Phase C v1.8.2 (D-SPEC-56) per rFP_titan_hcl_l2_separation_strategy.md §4.I:
        # `_dream_inbox` deque + `_dream_state` dict DELETED — dream state
        # ownership moved to dream_state_worker (G21 single writer of
        # dream_state.bin SHM slot). Chat handler reads is_dreaming via
        # DreamStateReader (sub-µs G18 SHM-direct, 100ms TTL cache). Chat-during-
        # dream buffering happens in dream_state_worker via DREAM_INBOX_ENQUEUE
        # bus events; drains on dream_end via DREAM_INBOX_REPLAY → chat handler
        # re-processes the buffered messages. See dream_state_worker.py.

        # EventBus / ObservatoryDB — populated by boot()
        self.event_bus = None
        self._observatory_db = None

        # rFP_trinity_130d_awakening Phase 2 — inner perception state owns
        # the AudioPerception / VisualPerception / AmbientChangeMonitor
        # trackers + _last_create_ts. Producers feed inner_mind[5,7,9] +
        # outer_spirit ANANDA[41]. Lazily started by boot() once
        # system_sensor is available.
        self._inner_perception = None

        # rFP_trinity_130d_awakening Phase 2 — outer spirit history aggregator.
        # Owns env_adapt + graceful_rest + circadian_alignment +
        # dream_recall trackers (SPEC §23.9 SAT[11], CHIT[25,26], ANANDA[40]).
        # CHIT[29] self_trajectory is worker-local in outer_spirit_worker.
        self._outer_spirit_history = None

        logger.info(
            "[TitanHCL] Coordinator constructed (kernel_id=%s, limbo=%s)",
            kernel.titan_id, kernel.limbo_mode,
        )

    # ------------------------------------------------------------------
    # L0 compat @property facade (delegate to kernel)
    # ------------------------------------------------------------------
    # Dashboard + agent code accesses these on the plugin root. Routing
    # them through the kernel preserves the legacy TitanCore duck-type
    # without requiring the plugin to own any L0 state.

    @property
    def bus(self):
        return self.kernel.bus

    @property
    def guardian(self):
        return self.kernel.guardian

    @property
    def state_register(self):
        return self.kernel.state_register

    @property
    def registry_bank(self):
        return self.kernel.registry_bank

    @property
    def soul(self):
        return self.kernel.soul

    @property
    def network(self):
        return self.kernel.network

    @property
    def disk_health(self):
        return self.kernel.disk_health

    @property
    def bus_health(self):
        return self.kernel.bus_health

    @property
    def _full_config(self):
        """Legacy name preserved so _register_modules / _wire_* methods
        that reference `self._full_config` work verbatim when lifted
        from v5_core.py. New code should prefer `self.kernel.config`.
        """
        return self.kernel.config

    @property
    def _limbo_mode(self):
        return self.kernel.limbo_mode

    @property
    def _start_time(self):
        """Legacy name — used by any code that previously read
        TitanCore._start_time for uptime calculations.
        """
        return self.kernel._start_time

    # ------------------------------------------------------------------
    # Plugin @property accessors (13 proxy facades — lifted verbatim)
    # ------------------------------------------------------------------
    # Allow existing Observatory API and agent code to work with
    # TitanHCL as if it were TitanHCL. Returns None when module
    # not loaded, so endpoints can degrade gracefully.

    @property
    def memory(self):
        """Lazy access — returns proxy or None."""
        return self._proxies.get("memory")

    @property
    def metabolism(self):
        return self._proxies.get("metabolism")

    @property
    def mood_engine(self):
        return self._proxies.get("mood_engine")

    # recorder / gatekeeper / scholar accessors RETIRED with the offline-RL
    # subsystem (RFP_synthesis_decision_authority P1) — execution-mode routing is
    # the grounded router; sovereignty is the ONE S.

    @property
    def consciousness(self):
        return self._proxies.get("consciousness")

    @property
    def social_graph(self):
        return self._proxies.get("social_graph")

    @property
    def social(self):
        return self._proxies.get("social")

    @property
    def studio(self):
        return self._proxies.get("studio")

    @property
    def maker_engine(self):
        return self._proxies.get("maker_engine")

    @property
    def sage_researcher(self):
        return self._proxies.get("sage_researcher")

    # ------------------------------------------------------------------
    # Module Registration — lifted verbatim from v5_core.py:317-696
    # ------------------------------------------------------------------

    # ─────────────────────────────────────────────────────────────────
    # Module catalog ownership moved out — Phase 6 / D-SPEC-135 / v1.62.0
    # ─────────────────────────────────────────────────────────────────
    # Pre-Phase-6 plugin._register_modules + plugin._register_api_subprocess_module
    # owned the 43-entry ModuleSpec catalog. Phase 6 carves L1 supervision
    # into the standalone guardian_hcl process (scripts/guardian_hcl.py).
    # The catalog now lives in titan_hcl/module_catalog.py:build_catalog
    # and is invoked exclusively by scripts/guardian_hcl.py. Per Maker
    # `feedback_no_shim_old_path_must_be_deleted` the original methods are
    # DELETED here (PURE CUTOVER — no env-var fallback, no in-process
    # Guardian construction left behind).

    # ------------------------------------------------------------------
    # Proxy Creation — lifted verbatim from v5_core.py:697-735
    # ------------------------------------------------------------------

    def _create_proxies(self) -> None:
        """Create proxy objects that bridge V2 API calls to V3 bus-supervised modules.

        Lifted verbatim from v5_core.py:697-735 per PLAN §4.2 Commit 3.
        `self.bus` and `self.guardian` work via compat @property delegates.
        """
        from titan_hcl.proxies.memory_proxy import MemoryProxy
        from titan_hcl.proxies.llm_proxy import LLMProxy
        from titan_hcl.proxies.mind_proxy import MindProxy
        from titan_hcl.proxies.body_proxy import BodyProxy
        from titan_hcl.proxies.media_proxy import MediaProxy
        from titan_hcl.proxies.timechain_proxy import TimechainProxy
        from titan_hcl.proxies.social_graph_proxy import SocialGraphProxy
        from titan_hcl.proxies.agno_proxy import AgnoProxy
        # rFP_phase_c_state_read_unification §B.5 — SpiritProxy retired
        # 2026-05-18 in favor of ShmReaderBank.compose_trinity which reads
        # PURELY from Rust L0+L1 canonical SHM slots per the Maker
        # directive. The bank is the canonical trinity composer fleet-wide.
        from titan_hcl.api.shm_reader_bank import ShmReaderBank

        # Lazy modules — start on first use
        self._proxies["memory"] = MemoryProxy(self.bus, self.guardian)
        self._proxies["llm"] = LLMProxy(self.bus, self.guardian)

        # Always-on modules — already started by Guardian
        self._proxies["mind"] = MindProxy(self.bus, self.guardian)
        self._proxies["body"] = BodyProxy(self.bus, self.guardian)
        # ShmReaderBank is the trinity composer post-Phase-B.5 (no proxy)
        self._shm_reader_bank = ShmReaderBank()

        # Media module (lazy — starts on first use)
        self._proxies["media"] = MediaProxy(self.bus, self.guardian)

        # TimeChain v2 Consumer API proxy
        self._proxies["timechain"] = TimechainProxy(self.bus, self.guardian)

        # V2-compatible aliases (so dashboard/agent code finds what it expects)
        self._proxies["mood_engine"] = self._proxies["mind"]  # mind proxy has get_mood_label()
        # social_graph — dedicated proxy + dedicated subprocess per
        # rFP_titan_hcl_l2_separation_strategy §4.P + D-SPEC-50 (v1.7.1).
        # Replaces the legacy MindProxy alias rot that surfaced as
        # AttributeError 'MindProxy' object has no attribute
        # 'record_interaction_async' on every chat post-hook fleet-wide
        # (Maker 2026-05-12: "no right to be in microkernel Phase C
        # architecture that must be lean and fast").
        self._proxies["social_graph"] = SocialGraphProxy(self.bus, self.guardian)

        # SPEC v1.17.0 / D-SPEC-72 — agno_proxy installation. AgnoProxy wraps
        # the bus-RPC round-trip to agno_worker for /chat + /v4/pitch-chat
        # endpoints. Cached at app.state.agno_proxy by api factory at boot
        # (api/__init__.py reads plugin._proxies["agno"] → assigns to app.state).
        # 90s request timeout matches the existing chat-arun Layer-1 closure
        # for BUG-CHAT-AGENT-ARUN-HANG-T3-PHASE-C; allowlisted in
        # phase_c_rpc_exemptions.yaml as `agno_proxy → agno_worker` work-RPC.
        self._proxies["agno"] = AgnoProxy(self.bus, request_timeout_s=90.0)

        # ── V2 Subsystems (direct instances in Core) ──────────────────
        # _wire_metabolism/studio/social arrive in commit 4.
        # _wire_sovereignty RETIRED v1.8.3 §4.L (D-SPEC-57, 2026-05-15) —
        # GreatCycleTracker now lives in sovereignty_worker subprocess.
        self._wire_metabolism()
        self._wire_meditation()  # §4.D v1.8.3 D-SPEC-57
        self._wire_life_force()  # §4.G v1.8.4 D-SPEC-58
        # _wire_sovereignty RETIRED v1.9.1 §4.L (D-SPEC-60, 2026-05-15) —
        # GreatCycleTracker now lives in sovereignty_worker subprocess.
        self._wire_studio()
        self._wire_social()

        logger.info("[TitanHCL] Created %d proxies", len(self._proxies))

    # ------------------------------------------------------------------
    # V2 Subsystem wiring — lifted verbatim from v5_core.py:737-869
    # ------------------------------------------------------------------

    def _wire_metabolism(self) -> None:
        """Install MetabolismProxy in self._proxies['metabolism'].

        Per rFP_titan_hcl_l2_separation_strategy §4.J + D-SPEC-51
        (SPEC v1.7.2, 2026-05-14). The inline MetabolismController
        instantiation that lived here previously has moved into
        `metabolism_worker` (separate subprocess, registered via the
        ModuleSpec block in `_register_modules`). titan_HCL holds only
        the proxy that exposes the same public surface — hot reads via
        SHM (sub-ms) + work-RPC for evaluate_gate + async state queries.

        The legacy `self.soul.set_metabolism(metabolism)` reverse-
        injection is REMOVED (Maker-locked 2026-05-14 "Replace with SHM
        read from soul"). Soul now constructs its own
        `MetabolismShmReader` for sub-ms tier/feature-flag reads — see
        `titan_hcl/core/soul.py`.
        """
        try:
            from titan_hcl.proxies.metabolism_proxy import MetabolismProxy
            self._proxies["metabolism"] = MetabolismProxy(
                bus=self.bus, guardian=self.guardian,
            )
            logger.info(
                "[TitanHCL] MetabolismProxy installed (rFP §4.J + "
                "D-SPEC-51 — MetabolismController now hosted in "
                "metabolism_worker subprocess)")
        except Exception as e:
            logger.warning(
                "[TitanHCL] MetabolismProxy wiring failed: %s", e,
                exc_info=True)

    def _wire_meditation(self) -> None:
        """Install MeditationProxy in self._proxies['meditation'].

        Per rFP_titan_hcl_l2_separation_strategy §4.D + D-SPEC-57
        (SPEC v1.8.3, 2026-05-15). The dual-process meditation
        orchestration (spirit_worker M3 driver + watchdog + tracker +
        MEDITATION_COMPLETE handler; plugin.py `_meditation_loop` +
        `_meditation_queue` pre-subscription; legacy_core mirror) has
        moved into `meditation_worker` (separate subprocess). titan_HCL
        holds only the proxy that exposes the public surface — get_tracker
        / get_watchdog_health / force_end (G18 SHM reads + fire-and-forget
        MEDITATION_FORCE_END bus publish).
        """
        try:
            from titan_hcl.proxies.meditation_proxy import MeditationProxy
            self._proxies["meditation"] = MeditationProxy(
                bus=self.bus, guardian=self.guardian,
            )
            logger.info(
                "[TitanHCL] MeditationProxy installed (rFP §4.D + "
                "D-SPEC-57 — meditation_tracker + driver + watchdog + "
                "orchestrator now hosted in meditation_worker subprocess)")
        except Exception as e:
            logger.warning(
                "[TitanHCL] MeditationProxy wiring failed: %s", e,
                exc_info=True)

    def _wire_life_force(self) -> None:
        """Install LifeForceProxy in self._proxies['life_force'].

        Per rFP_titan_hcl_l2_separation_strategy §4.G + D-SPEC-58
        (SPEC v1.8.4, 2026-05-15). LifeForceEngine (Chi Λ 3×3 Trinity
        vitality math) was hosted in cognitive_worker chunk 8M.6 as Track 1
        drift since 2026-05-10; v1.8.4 extracts it into a dedicated
        life_force_worker subprocess. titan_HCL holds only the proxy that
        exposes chi state — hot reads via SHM (sub-µs) + work-RPC for
        get_stats / get_chi_history / get_contemplation_status.
        """
        try:
            from titan_hcl.proxies.life_force_proxy import LifeForceProxy
            self._proxies["life_force"] = LifeForceProxy(
                bus=self.bus, guardian=self.guardian,
            )
            logger.info(
                "[TitanHCL] LifeForceProxy installed (rFP §4.G + "
                "D-SPEC-58 — LifeForceEngine now hosted in "
                "life_force_worker subprocess)")
        except Exception as e:
            logger.warning(
                "[TitanHCL] LifeForceProxy wiring failed: %s", e,
                exc_info=True)

    # _wire_sovereignty RETIRED v1.9.1 §4.L (D-SPEC-60, 2026-05-15) —
    # GreatCycleTracker now lives in titan_hcl/modules/sovereignty_worker.py
    # as Guardian-supervised L2 subprocess.

    def _wire_studio(self) -> None:
        """Install StudioProxy in self._proxies['studio'].

        Per rFP_titan_hcl_l2_separation_strategy §4.K + D-SPEC-57
        (SPEC v1.8.3, 2026-05-15). The inline StudioCoordinator
        instantiation that lived here previously has moved into
        `studio_worker` (separate subprocess, registered via the
        ModuleSpec block in `_register_modules`). titan_HCL holds only
        the proxy that exposes the same public surface — fire-and-forget
        request_* + _with_completion variants (D-SPEC-46 Future-registry)
        + get_gallery_async (work-RPC ≤2s) + get_stats (SHM-direct).

        The provider-specific OllamaCloudClient injection at
        `__init__.py:238` is REMOVED — studio_worker constructs zero
        provider clients (Maker direction 2026-05-15 Q2). Haiku
        generation routes via the canonical llm_proxy.distill work-RPC
        inside the worker, abstracting over any future inference provider.
        """
        try:
            from titan_hcl.proxies.studio_proxy import StudioProxy
            self._proxies["studio"] = StudioProxy(
                bus=self.bus, guardian=self.guardian,
            )
            logger.info(
                "[TitanHCL] StudioProxy installed (rFP §4.K + "
                "D-SPEC-57 — StudioCoordinator now hosted in "
                "studio_worker subprocess)")
        except Exception as e:
            logger.warning(
                "[TitanHCL] StudioProxy wiring failed: %s", e,
                exc_info=True)

    def _wire_social(self) -> None:
        """Wire SocialManager in Core (degraded mode — no API keys, but structure in place).

        Lifted verbatim from v5_core.py:838-869 per PLAN §4.2 Commit 4.
        """
        try:
            from titan_hcl.expressive.social import SocialManager
            sage_cfg = self._full_config.get("stealth_sage", {})
            # rFP_titan_hcl_l2_separation_strategy §4.P + D-SPEC-50 (v1.7.1,
            # 2026-05-14). The parent process no longer instantiates an
            # in-process SocialGraph — that hosting moved to
            # social_graph_worker (its own subprocess; G21 single-writer
            # of data/social_graph.db). SocialManager receives the
            # SocialGraphProxy instead: same async API surface (the
            # `*_async` methods SocialManager calls all exist on the
            # proxy), routed via bus.request_async work-RPC.
            social_graph = self._proxies.get("social_graph")
            social = SocialManager(
                metabolism_client=self._proxies.get("metabolism"),
                mood_engine=self._proxies.get("mood_engine"),
                memory=self._proxies.get("memory"),
                stealth_sage_config=sage_cfg,
                social_graph=social_graph,
            )
            # Wire dry-run mode from endurance config
            endurance_cfg = self._full_config.get("endurance", {})
            if endurance_cfg.get("social_dry_run", True):
                social._dry_run = True
                social._dry_run_log = endurance_cfg.get(
                    "social_dry_run_log", "./data/logs/social_dry_run.log"
                )
            self._proxies["social"] = social
            logger.info("[TitanHCL] SocialManager wired (dry_run=%s)", social._dry_run)
        except Exception as e:
            logger.warning("[TitanHCL] SocialManager wiring failed: %s", e)

    # ------------------------------------------------------------------
    # Observatory API (reuses existing V2 create_app)
    # ------------------------------------------------------------------

    def _create_observatory_app(self, api_cfg: dict):
        """Create the Observatory FastAPI app synchronously.

        Lifted verbatim from v5_core.py:935-944 per PLAN §4.2 Commit 4.
        """
        try:
            from titan_hcl.api import create_app
            app = create_app(self, self.event_bus, api_cfg)
            self._observatory_app = app
            return app
        except Exception as e:
            logger.warning("[TitanHCL] Observatory app creation failed: %s", e)
            return None

    async def _start_observatory(self, api_cfg: dict):
        """Launch the Observatory API server.

        Lifted verbatim from v5_core.py:946-963 per PLAN §4.2 Commit 4.
        """
        try:
            import uvicorn
            app = self._observatory_app
            if app is None:
                return
            host = api_cfg.get("host", "0.0.0.0")
            # Microkernel v2 Phase B.1 — TITAN_API_PORT env var (set by shadow
            # orchestrator's _phase_shadow_boot) overrides config.toml port,
            # so the shadow binds to its assigned shadow_port (7779/7777
            # ping-pong) instead of always 7777. Without this, the shadow
            # collides with the running old kernel and `address already in
            # use`. Codified 2026-04-28 PM during T1 swap E2E test.
            env_port = os.environ.get("TITAN_API_PORT")
            port = int(env_port) if env_port else int(api_cfg.get("port", 7777))
            uvi_config = uvicorn.Config(
                app=app, host=host, port=port, log_level="info", access_log=False,
            )
            self._uvicorn_server = uvicorn.Server(uvi_config)
            await self._uvicorn_server.serve()
        except SystemExit:
            logger.warning("[TitanHCL] Observatory could not bind port")
        except Exception as e:
            logger.warning("[TitanHCL] Observatory failed: %s", e)

    def reload_api(self) -> dict:
        """Hot-reload API routes by rebuilding the FastAPI app and swapping it.

        Returns dict with reload status. The uvicorn server keeps running —
        only the ASGI app reference changes. Zero downtime.

        Lifted verbatim from v5_core.py:965-988 per PLAN §4.2 Commit 4.
        """
        try:
            from titan_hcl.api import reload_api_app
            old_app = self._observatory_app
            new_app = reload_api_app(old_app)
            self._observatory_app = new_app

            # Swap the app in the running uvicorn server
            if hasattr(self, '_uvicorn_server') and self._uvicorn_server:
                self._uvicorn_server.config.app = new_app
                # Also update the loaded_app which uvicorn uses for serving
                if hasattr(self._uvicorn_server, 'config'):
                    self._uvicorn_server.config.loaded_app = new_app

            logger.info("[TitanHCL] API hot-reloaded — routes updated, server continuous")
            return {"status": "ok", "reloaded": True}
        except Exception as e:
            logger.error("[TitanHCL] API reload failed: %s", e)
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Agency Module (Step 7)
    # ------------------------------------------------------------------

    def _boot_reflex_collector(self) -> None:
        """Initialize the Sovereign Reflex Collector with executors.

        A.8.5 — flag-aware: when microkernel.a8_reflex_subprocess_enabled
        is true, parent's reflex_collector becomes a ReflexProxy that
        bus-routes the aggregation step (steps 1-4 of collect_and_fire:
        group → guardian-shield → combine → threshold + cooldown filter
        → top-N) to reflex_worker. Executors stay parent-resident (they
        reference plugin.soul / plugin.metabolism / plugin.memory_proxy
        / plugin.knowledge_proxy / plugin.social_proxy / plugin.agency
        and cannot trivially move to a subprocess) and run locally via
        the inherited _execute_selected step.

        When false (default), parent retains the regular ReflexCollector
        — byte-identical to pre-A.8.5 behavior.
        """
        try:
            from titan_hcl.logic.reflex_executors import register_reflex_executors
            from titan_hcl.params import get_params

            reflex_cfg = get_params("reflexes")

            _reflex_subproc_enabled = bool(
                self._full_config.get("microkernel", {}).get(
                    "a8_reflex_subprocess_enabled", False))
            if _reflex_subproc_enabled:
                from titan_hcl.proxies.reflex_proxy import ReflexProxy
                self.reflex_collector = ReflexProxy(self.bus, reflex_cfg)
                logger.info(
                    "[TitanHCL] ReflexCollector using subprocess proxy "
                    "(A.8.5 flag enabled)")
            else:
                from titan_hcl.logic.reflexes import ReflexCollector
                self.reflex_collector = ReflexCollector(reflex_cfg)

            # Register executors — they wrap existing subsystems. Works
            # on both ReflexCollector + ReflexProxy (proxy inherits
            # register_executor + _executors from base).
            count = register_reflex_executors(self.reflex_collector, self)
            logger.info("[TitanHCL] ReflexCollector booted: %d executors, threshold=%.2f",
                        count, self.reflex_collector.fire_threshold)
        except Exception as e:
            logger.warning("[TitanHCL] ReflexCollector boot failed: %s", e)
            self.reflex_collector = None

    def _boot_agency(self) -> None:
        """Initialize Agency — local module OR subprocess proxy (L3 §A.8.6).

        Original behavior (flag off, default): instantiates AgencyModule +
        SelfAssessment + HelperRegistry + 8 helpers in parent.

        Subprocess mode (microkernel.a8_agency_subprocess_enabled=true):
        instantiates AgencyProxy + AssessmentProxy that bus.request() into
        agency_worker — all LLM calls + helper.execute() awaits run in
        the worker subprocess, parent event loop never blocks on them.

        InterfaceAdvisor MOVED to interface_advisor_worker subprocess in
        v1.8.5 §4.H (D-SPEC-59, 2026-05-15) per `feedback_phase_c_break_
        monolith_ethos.md` — every L2 carve under Phase C earns its place
        via hot-reload + restart-isolation + own §9.B block. Parent reads
        rate state from `interface_advisor_state.bin` SHM slot via
        InterfaceAdvisorStateReader (sub-µs G18, 100ms cache); rate checks
        emit IMPULSE_RECEIVED bus event (fire-and-forget P3) to worker.

        ExpressionTranslator stays in parent — it only needs the
        helper-names list (worker advertises via AGENCY_READY +
        AGENCY_STATS broadcast → proxy._registry facade).
        """
        try:
            from titan_hcl.logic.interface_advisor_reader import (
                InterfaceAdvisorStateReader,
            )

            agency_cfg = self._full_config.get("agency", {})
            if not agency_cfg.get("enabled", True):
                logger.info("[TitanHCL] Agency disabled by config")
                return

            # InterfaceAdvisor MOVED to interface_advisor_worker subprocess
            # (v1.8.5 §4.H, D-SPEC-59). Parent reads rate state from SHM.
            self._interface_advisor_reader = InterfaceAdvisorStateReader()

            # L3 §A.8.6 — flag-routed agency residency.
            agency_subproc_enabled = bool(
                self._full_config.get("microkernel", {}).get(
                    "a8_agency_subprocess_enabled", False))

            if agency_subproc_enabled:
                from titan_hcl.proxies.agency_proxy import AgencyProxy
                from titan_hcl.proxies.assessment_proxy import AssessmentProxy
                self._agency = AgencyProxy(self.bus)
                self._agency_assessment = AssessmentProxy(self.bus)
                # Helper-names list comes from agency_worker via
                # AGENCY_READY broadcast — empty until first broadcast
                # arrives (typically <1s after worker boots). Bootstrap
                # ExpressionTranslator with empty list; it'll use
                # whatever's cached in the proxy._registry at translate
                # time anyway.
                _initial_helpers: list[str] = []
                logger.info("[TitanHCL] Agency using subprocess proxy "
                            "(A.8.6 flag enabled) — helpers list will populate "
                            "from AGENCY_READY broadcast")
            else:
                from titan_hcl.logic.agency.registry import HelperRegistry
                from titan_hcl.logic.agency.module import AgencyModule
                from titan_hcl.logic.agency.assessment import SelfAssessment

                # Create registry and register helpers (legacy local mode)
                registry = HelperRegistry()
                self._register_helpers(registry)

                # LLM function for Agency (uses Venice/OllamaCloud via inference config)
                llm_fn = self._create_agency_llm_fn()

                budget = int(agency_cfg.get("llm_budget_per_hour", 10))
                self._agency = AgencyModule(registry=registry, llm_fn=llm_fn, budget_per_hour=budget)
                self._agency_assessment = SelfAssessment(llm_fn=llm_fn)
                _initial_helpers = registry.list_helper_names() \
                    if hasattr(registry, 'list_helper_names') else []
                helper_names = registry.list_all_names()
                statuses = registry.get_all_statuses()
                available = [n for n, s in statuses.items() if s == "available"]
                logger.info("[TitanHCL] Agency local mode: %d helpers registered "
                            "(%d available): %s",
                            len(helper_names), len(available), available)

            # Expression Translation Layer — learned action selection.
            # Reads helpers list at construction; runtime translate() uses
            # self._agency._registry.list_helper_names() which works for
            # both local registry and proxy _RegistryFacade.
            try:
                from titan_hcl.logic.expression_translator import (
                    ExpressionTranslator, FeedbackRouter)
                self._expression_translator = ExpressionTranslator(
                    all_helpers=_initial_helpers)
                self._expression_translator.load(
                    "./data/neural_nervous_system/expression_state.json")
                self._feedback_router = FeedbackRouter(
                    hormonal_system=None,  # Wired later when neural_ns available
                    translator=self._expression_translator)
                logger.info("[TitanHCL] ExpressionTranslator booted "
                            "(sovereignty=%.1f%%)",
                            self._expression_translator.sovereignty_ratio * 100)
            except Exception as e:
                logger.warning("[TitanHCL] Expression layer init error: %s", e)
                self._expression_translator = None
                self._feedback_router = None

            # Subscribe agency to bus — receives IMPULSE/OUTER_DISPATCH from
            # spirit_worker, plus AGENCY_READY/AGENCY_STATS/ASSESSMENT_STATS
            # broadcasts from agency_worker (when flag on) for proxy cache
            # refresh in _agency_loop.
            # Option B (2026-04-29): explicit broadcast filter matching the
            # elif chain in _agency_loop (lines ~2105-2132). Manually
            # verified against scripts/migrate_bus_filters.py output.
            # IMPULSE / OUTER_DISPATCH / QUERY are the agency-routing types;
            # AGENCY_STATS / ASSESSMENT_STATS / AGENCY_READY are the
            # L3 §A.8.6 proxy-cache refresh types broadcast by the
            # agency_worker subprocess.
            self._agency_queue = self.bus.subscribe(
                "agency",
                types=[
                    IMPULSE, bus.OUTER_DISPATCH, bus.QUERY,
                    AGENCY_STATS, ASSESSMENT_STATS, AGENCY_READY,
                ],
            )
            # 2026-05-19 BOOT_TRACE diagnostic — confirms agency subscription
            # registered. Without this log, we cannot tell whether the
            # subscriber is eligible for IMPULSE cross-process delivery.
            logger.warning(
                "[TitanHCL] [BOOT_TRACE] _boot_agency complete — "
                "agency_queue=%s subscribed for IMPULSE/OUTER_DISPATCH/"
                "QUERY/AGENCY_STATS/ASSESSMENT_STATS/AGENCY_READY",
                type(self._agency_queue).__name__)

        except Exception as e:
            logger.warning("[TitanHCL] Agency boot failed: %s", e)
            self._agency = None

    def _register_helpers(self, registry) -> None:
        """Register all available helpers in the registry.

        Lifted verbatim from v5_core.py:1070-1175 per PLAN §4.2 Commit 4.
        Path adjustment: titan_params.toml location — v5_core is at
        titan_hcl/, plugin.py is at titan_hcl/core/, so ".." prefix
        added to resolve titan_hcl/titan_params.toml correctly.
        """
        try:
            from titan_hcl.logic.agency.helpers.infra_inspect import InfraInspectHelper
            # §7.P5 — prefer the live kernel journal; fall back to the log file.
            try:
                from titan_hcl.core.state_registry import resolve_titan_id
                _ii_tid = resolve_titan_id()
            except Exception:  # noqa: BLE001
                _ii_tid = ""
            registry.register(InfraInspectHelper(
                log_path="/tmp/titan_v3.log",
                service=f"titan-{_ii_tid}.service" if _ii_tid else None))
        except Exception as e:
            logger.warning("[TitanHCL] InfraInspect helper failed: %s", e)

        try:
            from titan_hcl.logic.agency.helpers.web_search import WebSearchHelper
            sage_cfg = self._full_config.get("stealth_sage", {})
            searxng_host = sage_cfg.get("searxng_host", "http://localhost:8080")
            firecrawl_key = sage_cfg.get("firecrawl_api_key", "")
            # BUG-KP-WEBSEARCH-HEALTH-DEFAULTS fix (2026-04-21) — forward
            # the same [knowledge_pipeline.budgets] MB→bytes dict that
            # knowledge_worker uses. Without this, WebSearchHelper's
            # HealthTracker had empty defaults and could clobber shared
            # data/knowledge_pipeline_health.json with budget=0 entries.
            _kp_cfg = self._full_config.get("knowledge_pipeline", {}) or {}
            _kp_budgets_mb = _kp_cfg.get("budgets", {}) or {}
            _kp_budgets_bytes = {
                k: int(v) * 1024 * 1024
                for k, v in _kp_budgets_mb.items()
                if isinstance(v, (int, float))
            }
            registry.register(WebSearchHelper(
                searxng_url=searxng_host,
                firecrawl_api_key=firecrawl_key,
                budgets=_kp_budgets_bytes,
            ))
        except Exception as e:
            logger.warning("[TitanHCL] WebSearch helper failed: %s", e)

        # SocialPostHelper REMOVED — all posting goes through SocialPressureMeter
        # (social_narrator + quality gate + rate limits + 11 post types).
        # Agency selecting social_post bypassed our designed narrator entirely.

        try:
            from titan_hcl.logic.agency.helpers.art_generate import ArtGenerateHelper
            exp_cfg = self._full_config.get("expressive", {})
            output_dir = exp_cfg.get("output_path", "./data/studio_exports")
            registry.register(ArtGenerateHelper(output_dir=output_dir))
        except Exception as e:
            logger.warning("[TitanHCL] ArtGenerate helper failed: %s", e)

        try:
            from titan_hcl.logic.agency.helpers.audio_generate import AudioGenerateHelper
            exp_cfg = self._full_config.get("expressive", {})
            audio_cfg = self._full_config.get("audio", {})
            output_dir = exp_cfg.get("output_path", "./data/studio_exports")
            max_duration = int(audio_cfg.get("max_duration_seconds", 30))
            sample_rate = int(audio_cfg.get("sample_rate", 44100))
            registry.register(AudioGenerateHelper(
                output_dir=output_dir,
                max_duration=max_duration,
                sample_rate=sample_rate,
            ))
        except Exception as e:
            logger.warning("[TitanHCL] AudioGenerate helper failed: %s", e)

        try:
            from titan_hcl.logic.agency.helpers.coding_sandbox import CodingSandboxHelper
            registry.register(CodingSandboxHelper())
        except Exception as e:
            logger.warning("[TitanHCL] CodingSandbox helper failed: %s", e)

        try:
            from titan_hcl.logic.agency.helpers.code_knowledge import CodeKnowledgeHelper
            registry.register(CodeKnowledgeHelper())
        except Exception as e:
            logger.warning("[TitanHCL] CodeKnowledge helper failed: %s", e)

        try:
            from titan_hcl.logic.agency.helpers.memo_inscribe import MemoInscribeHelper
            # MemoInscribeHelper reads config.toml directly for RPC + keypair.
            # Mainnet Lifecycle Wiring rFP: inject metabolism for memo gate
            # + governance reserve guard.
            registry.register(MemoInscribeHelper(
                metabolism=self._proxies.get("metabolism"),
                shm_reader_bank=self._shm_reader_bank))
        except Exception as e:
            logger.warning("[TitanHCL] MemoInscribe helper failed: %s", e)

        # Kin Discovery — consciousness-to-consciousness exchange
        try:
            from titan_hcl.logic.agency.helpers.kin_sense import KinSenseHelper
            import tomllib as _tomllib_kin
            _kin_params = {}
            # Plugin lives at titan_hcl/core/plugin.py — go up one to
            # titan_hcl/titan_params.toml.
            _kin_params_path = os.path.join(
                os.path.dirname(__file__), "..", "titan_params.toml")
            if os.path.exists(_kin_params_path):
                with open(_kin_params_path, "rb") as _kf:
                    _kin_params = _tomllib_kin.load(_kf)
            _kin_cfg = _kin_params.get("kin", {})
            if _kin_cfg.get("enabled", False):
                # TITAN_KIN_ADDRESSES env var overrides config (for T2 pointing to T1 via nginx)
                _kin_addrs = _kin_cfg.get("addresses", [])
                _env_addrs = os.environ.get("TITAN_KIN_ADDRESSES", "")
                if _env_addrs:
                    _kin_addrs = [a.strip() for a in _env_addrs.split(",") if a.strip()]
                registry.register(KinSenseHelper(
                    kin_addresses=_kin_addrs,
                    exchange_strength=_kin_cfg.get("exchange_strength", 0.03),
                ))
                logger.info("[TitanHCL] KinSense helper registered: addresses=%s",
                            _kin_addrs)
        except Exception as e:
            logger.warning("[TitanHCL] KinSense helper failed: %s", e)

    def _create_agency_llm_fn(self):
        """Create a lightweight async LLM function for Agency module.

        Phase 3 Chunk χ-bis (D-SPEC-88, 2026-05-18) — direct OllamaCloudClient
        + Venice fallback REPLACED by /v4/llm-distill round-trip. Mirrors
        the agency_worker.py subprocess path so both parent + worker share
        the same centralized LLM gateway. All LLM traffic appears in
        llm_state.bin.
        """
        inference_cfg = self._full_config.get("inference", {})
        api_cfg = self._full_config.get("api", {}) or {}
        _api_port = int(api_cfg.get("port", 7777))
        _api_base = f"http://127.0.0.1:{_api_port}"
        _internal_key = api_cfg.get("internal_key", "") or ""

        async def agency_llm(prompt: str, task: str = "agency_select") -> str:
            """LLM call for helper selection / assessment / code generation."""
            try:
                from titan_hcl.inference import get_model_for_task
                from titan_hcl.logic.llm_distill_client import (
                    distill_via_http_async)
                model = get_model_for_task(task)
                max_tok = 800 if task == "agency_code_gen" else 200
                result = await distill_via_http_async(
                    text=prompt,
                    instruction="",
                    api_base=_api_base,
                    internal_key=_internal_key,
                    model=model,
                    max_tokens=max_tok,
                    consumer=f"agency.{task}",
                    timeout_s=30.0,
                )
                if result:
                    return result
            except Exception as e:
                logger.warning("[Agency LLM] /v4/llm-distill failed: %s", e)

            raise RuntimeError("No LLM available for agency")

        return agency_llm

    # ------------------------------------------------------------------
    # plugin.run_chat() RETIRED in Phase C v1.17.0 (D-SPEC-72).
    # The chat pipeline now lives in agno_worker subprocess; /chat +
    # /v4/pitch-chat handlers route through agno_proxy.chat() per
    # SPEC §9.B agno_worker block + §9.F.2 llm_pipeline contract.
    # See titan-docs/rFP_agno_worker_and_llm_libraries_extraction.md.
    # ------------------------------------------------------------------



    async def _guardian_handler_loop(self) -> None:
        """Bus subscriber for QUERY dst="guardian" admin requests.

        Closes both BUG-HOT-RELOAD-CODE-LOADING and BUG-GUARDIAN-CONTROL-
        COMMANDS-ORPHAN. Implements the missing receiver for the long-
        documented Phase C admin pattern: api_subprocess publishes QUERY
        with `action ∈ {"restart_module", "start_module", "stop_module"}`
        + payload, this loop dispatches to Guardian methods on the parent
        process, RESPONSE rid-routed back via bus.

        Concurrency: each admin request may take 5-30s (worker restart +
        boot), so requests are dispatched to separate asyncio tasks so a
        slow restart doesn't serialize subsequent admin calls.
        """
        try:
            # D-SPEC-151: admin QUERY (restart/start/stop/reload_module) arrives
            # on dst="guardian" via the plugin's IN_PROCESS alias → here →
            # _handle_guardian_request → GuardianHCLClient → MODULE_*_REQUEST to
            # "guardian_hcl_lifecycle" (the real Orchestrator executor). ORIGINAL
            # hot-reload/restart-module design. The heartbeat flood was the
            # Orchestrator's separate undrained "guardian" queue (fixed by
            # subscribe_guardian=False), NOT this loop — it drains QUERY at 10 Hz
            # and discards heartbeats (original behavior, no accumulation).
            queue = self.bus.subscribe("guardian", types=[bus.QUERY])
        except Exception as e:
            logger.warning("[TitanHCL] guardian handler subscribe failed: %s", e)
            return
        logger.info("[TitanHCL] guardian handler loop started — listening for admin QUERY")
        while True:
            try:
                msgs = self.bus.drain(queue, max_msgs=20)
                for msg in msgs:
                    if msg.get("type") != bus.QUERY:
                        continue
                    payload = msg.get("payload") or {}
                    action = payload.get("action")
                    if action not in (
                        "restart_module", "start_module", "stop_module",
                        # SPEC §8.3 Phase B — per-module hot-reload (D-SPEC-50).
                        # `Guardian.reload_module()` is async; dispatched via
                        # asyncio.create_task in `_handle_guardian_request`.
                        "reload_module",
                    ):
                        continue
                    asyncio.get_event_loop().create_task(
                        self._handle_guardian_request(msg))
                await asyncio.sleep(0.1)  # 10 Hz drain — admin ops are rare
            except Exception as e:
                logger.error("[TitanHCL] guardian handler loop error: %s",
                             e, exc_info=True)
                await asyncio.sleep(2.0)

    async def _handle_guardian_request(self, msg: dict) -> None:
        """Process one Guardian admin request.

        Supported actions:
          - `restart_module(name, reason, start_method=None)` → calls
             `Guardian.restart_module()` (BUG-HOT-RELOAD-CODE-LOADING).
          - `start_module(name)` → `Guardian.start()` (BUG-GUARDIAN-CONTROL).
          - `stop_module(name, reason)` → `Guardian.stop()` (BUG-GUARDIAN-CONTROL).
        """
        rid = msg.get("rid")
        src = msg.get("src", "api")
        payload = msg.get("payload") or {}
        action = payload.get("action")
        inner = payload.get("payload") or {}
        result: dict = {"ok": False, "error": "unknown_action"}
        try:
            if action == "restart_module":
                name = inner.get("name")
                reason = inner.get("reason", "bus request")
                start_method = inner.get("start_method")
                # Off-load to executor — Guardian.restart_module() blocks
                # in stop() + start() (worker SIGTERM + new spawn). Keep
                # the asyncio loop free for concurrent admin requests.
                _r = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.guardian.restart_module(
                        name=name, reason=reason,
                        start_method=start_method))
                # guardian_hcl_client.restart_module() returns a bool; the API
                # endpoint expects a dict payload (result_dict.get("ok") etc.).
                # Normalize so the RESPONSE payload is always a dict (else the
                # caller hits "'bool' object has no attribute 'get'").
                result = _r if isinstance(_r, dict) else {
                    "ok": bool(_r),
                    "module": name,
                    "process_alive": bool(_r),
                    "start_method_used": start_method or "default",
                }
            elif action == "start_module":
                name = inner.get("name")
                ok = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.guardian.start(name))
                result = {"ok": bool(ok), "module": name, "error": None}
            elif action == "enable_module":
                # Re-enable a DISABLED module (after a max_restarts escalation):
                # Guardian.enable() clears the DISABLED state, resets the restart
                # counters, and starts it. start_module/restart_module are BLOCKED
                # by the DISABLED guard ("is disabled, not starting"), so a
                # flap-disabled module was previously UN-recoverable via the API
                # for ALL modules — only a full Titan restart cleared it. Closes
                # the structural recovery gap (BUG-GUARDIAN-CONTROL-COMMANDS-ORPHAN
                # family; 2026-06-09, surfaced by the backup flap incident).
                name = inner.get("name")
                ok = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.guardian.enable(name))
                result = {"ok": bool(ok), "module": name, "error": None}
            elif action == "stop_module":
                name = inner.get("name")
                reason = inner.get("reason", "bus request")
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.guardian.stop(name, reason=reason))
                result = {"ok": True, "module": name, "error": None}
            elif action == "reload_module":
                # SPEC §8.3 Phase B — per-module hot-reload (D-SPEC-50).
                # `Guardian.reload_module()` is async — awaitable directly.
                # The dispatch path (chat_bridge → QUERY → here) is what
                # `arch_map reload-module` and POST /v4/admin/reload-module
                # use to initiate from outside the parent process.
                name = inner.get("name")
                new_module_path = inner.get("new_module_path")
                timeout_s = float(inner.get("timeout_s", 30.0))
                # D-SPEC-151: align to GuardianHCLClient.reload_module(name, timeout,
                # **kwargs) — post-peer-spawn `self.guardian` is the bus-client
                # proxy (was the in-process Guardian's reload_module(module_name,
                # timeout_s)). Wrong kw names = TypeError "missing 'name'" (the
                # cutover never updated this caller; exposed once routing reached it).
                result = await self.guardian.reload_module(
                    name=name,
                    timeout=timeout_s,
                    new_module_path=new_module_path,
                )
                # reload_module returns {swap_id, module_name, status,
                # reason, total_elapsed_ms, ts} per SPEC §8.3 — add `ok`
                # flag for caller convenience without altering shape.
                result["ok"] = (result.get("status") == "ready")
        except Exception as e:
            logger.error("[TitanHCL] guardian handler %s raised: %s",
                         action, e, exc_info=True)
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        try:
            self.bus.publish({
                "type": bus.RESPONSE,
                "src": "guardian",
                "dst": src,
                "rid": rid,
                "payload": result,
                "ts": time.time(),
            })
            # SPEC §8.0.ter — flush before returning so the RESPONSE
            # reaches the requester's reply_queue before the handler
            # completes. Same rationale as _handle_chat_request flush:
            # the requester is awaiting via bus.request_async + rid.
            self.bus.flush(timeout=2.0)
        except Exception as e:
            logger.error("[TitanHCL] guardian RESPONSE publish failed: %s", e)

    # ------------------------------------------------------------------
    # Agency / Sovereignty / Impulse loops + handlers
    # ------------------------------------------------------------------
    # All lifted verbatim from v5_core.py:1218-1585 per PLAN §4.2 Commit 5.
    # Absolute imports replace relative imports.

    async def _agency_loop(self) -> None:
        """Listen for IMPULSE events on the bus and process them through Agency.

        Flow: IMPULSE → InterfaceAdvisor rate check → Agency handles → Assessment → ACTION_RESULT
        """
        # 2026-05-19 BOOT_TRACE — WARNING level so it survives journal flood
        logger.warning("[TitanHCL] [BOOT_TRACE] _agency_loop entered — "
                       "agency_queue type=%s",
                       type(getattr(self, "_agency_queue", None)).__name__)
        _bt_polls = 0
        _bt_msgs = 0
        _bt_last_log = time.time()
        while True:
            try:
                # Poll bus for IMPULSE messages addressed to agency or broadcast
                msg = None
                try:
                    msg = self._agency_queue.get_nowait()
                except Exception:
                    pass

                # 2026-05-19 BOOT_TRACE heartbeat — every 60s emit poll count
                # so we can see the loop is alive AND whether it's receiving
                # any messages. Cheap (1 log/min).
                _bt_polls += 1
                _now_bt = time.time()
                if _now_bt - _bt_last_log >= 60.0:
                    logger.warning(
                        "[TitanHCL] [BOOT_TRACE] _agency_loop alive — "
                        "polls=%d msgs_received=%d in last 60s",
                        _bt_polls, _bt_msgs)
                    _bt_polls = 0
                    _bt_msgs = 0
                    _bt_last_log = _now_bt

                if not msg:
                    await asyncio.sleep(2.0)  # Poll every 2s (impulses are rare)
                    continue

                _bt_msgs += 1
                msg_type = msg.get("type", "")
                # 2026-05-19 BOOT_TRACE — every msg received logged
                logger.warning(
                    "[TitanHCL] [BOOT_TRACE] msg received type=%s src=%s "
                    "dst=%s", msg_type, msg.get("src"), msg.get("dst"))

                if msg_type == IMPULSE:
                    await self._handle_impulse(msg)
                elif msg_type == bus.OUTER_DISPATCH:
                    await self._handle_outer_dispatch(msg)
                elif msg_type == bus.QUERY:
                    self._handle_agency_query(msg)
                elif msg_type == AGENCY_STATS:
                    # L3 §A.8.6 — refresh proxy stats cache
                    if hasattr(self._agency, "update_cached_stats"):
                        self._agency.update_cached_stats(msg.get("payload", {}) or {})
                elif msg_type == ASSESSMENT_STATS:
                    if hasattr(self._agency_assessment, "update_cached_stats"):
                        self._agency_assessment.update_cached_stats(
                            msg.get("payload", {}) or {})
                elif msg_type == AGENCY_READY:
                    # First broadcast after worker boot — seed proxy
                    # _registry helper-names cache so ExpressionTranslator's
                    # next translate() sees real helpers, not [].
                    if hasattr(self._agency, "update_cached_stats"):
                        boot_payload = msg.get("payload", {}) or {}
                        helpers = list(boot_payload.get("helpers", []) or [])
                        if helpers:
                            current = (getattr(self._agency, "_stats_cache", {})
                                       or {}).copy()
                            current["registered_helpers"] = helpers
                            self._agency.update_cached_stats(current)
                            logger.info("[TitanHCL] AGENCY_READY: worker advertises "
                                        "%d helpers — proxy cache seeded", len(helpers))

            except Exception as e:
                logger.error("[TitanHCL] Agency loop error: %s", e)
                await asyncio.sleep(5.0)

    # _rl_stats_loop RETIRED with the offline-RL subsystem
    # (RFP_synthesis_decision_authority P1) — it drained RLProxy's
    # `rl_proxy_stats`/SAGE_STATS broadcast queue; RLProxy is gone.

    # _sovereignty_loop RETIRED v1.8.3 §4.L (D-SPEC-57, 2026-05-15) —
    # SOVEREIGNTY_EPOCH consumption + tracker.record_epoch() + 100-message
    # criteria-snapshot log all moved to sovereignty_worker subprocess.

    # _sovereignty_queue property RETIRED v1.8.3 §4.L (D-SPEC-57) — kernel.py
    # bus.subscribe("sovereignty", types=[SOVEREIGNTY_EPOCH]) also retired
    # (worker subscribes directly via its own bus client).

    # _meditation_queue property REMOVED — meditation_worker subprocess
    # subscribes to MEDITATION_REQUEST via its own bus client (D-SPEC-57).

    async def _handle_impulse(self, msg: dict) -> None:
        """Process an IMPULSE event through the agency pipeline.

        Lifted verbatim from v5_core.py:1310-1481.
        """
        payload = msg.get("payload", {})
        posture = payload.get("posture", "unknown")
        impulse_id = payload.get("impulse_id", 0)

        logger.info("[TitanHCL] IMPULSE received: #%d posture=%s urgency=%.2f",
                    impulse_id, posture, payload.get("urgency", 0))

        # Rate check via SHM-rate-oracle (v1.8.5 §4.H, D-SPEC-59).
        # InterfaceAdvisor lives in interface_advisor_worker subprocess;
        # parent reads SHM snapshot sub-µs + (on within-limits) emits
        # IMPULSE_RECEIVED to worker so it records the timestamp + republishes.
        if self._interface_advisor_reader:
            feedback = self._interface_advisor_reader.check(IMPULSE, source="spirit")
            if feedback:
                # Phase D (D-SPEC-116): the RATE_LIMIT→spirit notify emit was
                # removed with spirit_worker (its handler only ever logged). The
                # rate-limit enforcement itself (the early return below) and the
                # local log line above are the real behavior — both retained.
                logger.info(
                    "[TitanHCL] IMPULSE rate-limited: current_rate=%d limit=%d",
                    feedback.get("current_rate", 0), feedback.get("limit", 0))
                return
            # Within-limits — fire-and-forget IMPULSE_RECEIVED to worker so
            # it records the timestamp in its sliding-window deque and
            # republishes interface_advisor_state.bin SHM (rate-throttled).
            try:
                self.bus.publish(make_msg(
                    bus.IMPULSE_RECEIVED, "core", "interface_advisor",
                    {"msg_type": IMPULSE, "source": "spirit",
                     "client_ts": time.time()},
                ))
            except Exception as e:
                logger.debug(
                    "[TitanHCL] IMPULSE_RECEIVED emit failed: %s", e)

        # Convert IMPULSE to INTENT (enriched with context)
        intent = {
            **payload,
            "trinity_snapshot": payload.get("trinity_snapshot", {}),
        }

        # Expression Translation Layer — try learned mapping first
        learned_selection = None
        if self._expression_translator:
            try:
                available = self._agency._registry.list_helper_names() \
                    if hasattr(self._agency._registry, 'list_helper_names') else []
                learned_selection = self._expression_translator.translate(
                    program=payload.get("triggering_program", ""),
                    intensity=payload.get("intensity", 0.0),
                    posture=posture,
                    available_helpers=available,
                    trinity_snapshot=payload.get("trinity_snapshot"),
                )
                if learned_selection:
                    # Inject learned selection — Agency will use it directly
                    intent["_learned_selection"] = learned_selection
                    self._expression_translator.record_action_type(was_learned=True)
                    logger.info("[TitanHCL] Expression: learned %s→%s (conf=%.2f)",
                                payload.get("triggering_program", "?"),
                                learned_selection["helper"],
                                learned_selection.get("confidence", 0))
                else:
                    self._expression_translator.record_action_type(was_learned=False)
            except Exception as e:
                logger.warning("[TitanHCL] Expression translator error: %s", e)

        # Agency Module handles the intent
        result = await self._agency.handle_intent(intent)
        if not result:
            logger.info("[TitanHCL] Agency skipped impulse #%d (no action taken)", impulse_id)
            return

        # Self-assessment
        if self._agency_assessment and result.get("success") is not None:
            try:
                assessment = await self._agency_assessment.assess(result)
                result["assessment"] = {
                    "score": assessment["score"],
                    "reflection": assessment["reflection"],
                    "enrichment": assessment["enrichment"],
                    "mood_delta": assessment["mood_delta"],
                    "threshold_direction": assessment["threshold_direction"],
                }
                logger.info("[TitanHCL] Assessment: score=%.2f direction=%s — %s",
                           assessment["score"], assessment["threshold_direction"],
                           assessment["reflection"][:80])
            except Exception as e:
                logger.warning("[TitanHCL] Assessment failed: %s", e)

        # Publish ACTION_RESULT back to bus (Spirit will pick it up).
        # Guard: 3 SQLite NOT NULL errors per hour surfaced 2026-04-15
        # (ACTION-RESULT-NULL-FIELDS in DEFERRED_ITEMS). Root cause: Agency
        # gated-away impulses (rate limit, unavailable helper) still
        # published ACTION_RESULT with empty helper/task_type/action_taken.
        # 3 downstream recorders (TitanCore inner_memory, SpiritWorker
        # ex_mem, ExperienceOrch) all rejected with NOT NULL constraint
        # failures. Skipping empty dispatches at the source — the action
        # didn't actually execute, so there's nothing to record.
        _helper = str(result.get("helper") or "").strip()
        if not _helper:
            logger.debug(
                "[TitanHCL] Skipping ACTION_RESULT with empty helper — "
                "gate/rate-limit path (success=%s)", result.get("success"))
        else:
            self.bus.publish(make_msg(ACTION_RESULT, "core", "all", result))
            logger.info("[TitanHCL] ACTION_RESULT published: helper=%s success=%s",
                        _helper, result.get("success"))

        # Feed action result to cognitive_worker for OBSERVATION (closed loop).
        # Phase D (D-SPEC-116): repointed from the retired spirit_worker; only
        # the X-engagement→MSL leg is live there now (others superseded/re-homed).
        try:
            self.bus.publish(make_msg(
                OUTER_OBSERVATION, "core", "cognitive_worker", {
                    "action_type": result.get("helper", ""),
                    "result": result,
                    "source": "impulse",
                }))
        except Exception as e:
            logger.warning("[TitanHCL] OUTER_OBSERVATION publish error: %s", e)

        # Record in Inner Memory (Phase M: action chain + event markers)
        try:
            if not hasattr(self, '_inner_memory'):
                from titan_hcl.logic.inner_memory import InnerMemoryStore
                self._inner_memory = InnerMemoryStore("./data/inner_memory.db")
            mem = self._inner_memory
            if mem:
                # Coalesce None → "" so `action_chains.helper TEXT NOT NULL`
                # never receives NULL. `.get(key, default)` only honors the
                # default on missing keys, not explicit None values.
                helper_name = result.get("helper") or ""
                # Bug B closure 2026-05-13: assessment score lives at
                # result["assessment"]["score"] (set above ~line 2864),
                # NOT result["score"]. Reading the wrong key left every
                # action_chains.score = 0.0 (14573 rows), which broke
                # practiced_response pool B (filters on
                # action_chains.score>=0.7). Fall back to
                # result["score"] for callers that bypass assessment.
                _assessment_score = (
                    (result.get("assessment") or {}).get("score")
                    if (result.get("assessment") or {}).get("score") is not None
                    else result.get("score", 0.0)
                )
                mem.record_action_chain(
                    impulse_id=result.get("impulse_id", 0),
                    triggering_program=result.get("triggering_program", ""),
                    posture=result.get("posture", ""),
                    helper=helper_name,
                    success=result.get("success", False),
                    score=_assessment_score,
                    reasoning=result.get("reflection", ""),
                    trinity_before=result.get("trinity_snapshot"),
                )
                # Record event markers for temporal tracking
                if helper_name == "web_search":
                    mem.record_event("explore", program="CURIOSITY")
                elif helper_name == "social_post":
                    mem.record_event("social", program="EMPATHY")
                elif helper_name in ("art_generate", "audio_generate"):
                    mem.record_event("create", program="CREATIVITY")
                    _work_type = "art" if helper_name == "art_generate" else "audio"
                    _file_path = result.get("file_path", "")
                    mem.record_creative_work(
                        work_type=_work_type,
                        file_path=_file_path,
                        triggering_program=result.get("triggering_program", ""),
                        posture=result.get("posture", ""),
                        # Same fix as action_chains above — assessment
                        # score lives at result["assessment"]["score"].
                        assessment_score=_assessment_score,
                    )
                    # Archive to ObservatoryDB for gallery/feed (in thread — non-blocking)
                    obs_db = getattr(self, "_observatory_db", None)
                    if obs_db and _file_path:
                        _style = result.get("art_style", _work_type)
                        asyncio.get_event_loop().run_in_executor(
                            None, obs_db.record_expressive,
                            _work_type,
                            f"{_style.replace('_', ' ').title()} ({result.get('triggering_program', 'autonomous')})",
                            result.get("result", ""),
                            _file_path,
                            "",
                            {
                                "triggering_program": result.get("triggering_program", ""),
                                "posture": result.get("posture", ""),
                                "score": result.get("score", 0.0),
                            },
                        )
                        # Inner perception fan-out happens via observatory_db
                        # hook (utils/observatory_db.record_expressive). One
                        # hook on the canonical write site covers all callers.
                elif helper_name == "infra_inspect":
                    mem.record_event("inspect", program="VIGILANCE")
                elif helper_name == "kin_sense":
                    mem.record_event("kin", program="EMPATHY")
        except Exception as e:
            logger.warning("[TitanHCL] Inner memory recording error: %s", e)

        # Expression Layer: route feedback + save state
        try:
            if self._feedback_router:
                self._feedback_router.route(result)
            if self._expression_translator:
                self._expression_translator.save(
                    "./data/neural_nervous_system/expression_state.json")
        except Exception as e:
            logger.warning("[TitanHCL] Expression feedback error: %s", e)

    async def _handle_outer_dispatch(self, msg: dict) -> None:
        """Handle OUTER_DISPATCH from two sources:
        1. Neural NS program fires (system=CREATIVITY/IMPULSE/etc.)
        2. Self-exploration expression fires (system=ART/MUSIC/SOCIAL/SPEAK)

        Both use autonomy-first path: no LLM calls, no budget consumed.
        Source distinguished by payload["source"]: "neural_ns" (default) or "self_exploration".

        Lifted verbatim from v5_core.py:1483-1563.
        """
        payload = msg.get("payload", {})
        signals = payload.get("signals", [])
        if not signals:
            return

        _dispatch_source = payload.get("source", "neural_ns")
        logger.info("[TitanHCL] OUTER_DISPATCH: %d signals from %s",
                    len(signals), _dispatch_source)

        # Get Trinity snapshot for context
        trinity_snapshot = {}
        try:
            body_proxy = self._proxies.get("body")
            if body_proxy:
                snap = body_proxy.get_tensor_snapshot()
                trinity_snapshot = snap if isinstance(snap, dict) else {}
        except Exception:
            pass

        # Dispatch via Agency (autonomy-first — no LLM calls, no budget consumed)
        results = await self._agency.dispatch_from_nervous_signals(
            outer_signals=signals,
            trinity_snapshot=trinity_snapshot,
        )

        # Publish results and assess outcomes
        # Self-exploration results are observed by OuterInterface (sensory decoder +
        # vocabulary reinforcement), so skip heavy LLM assessment to avoid double-enrichment.
        # Neural NS results get full agency assessment (enrichment routing, mood delta).
        for result in results:
            result["dispatch_source"] = _dispatch_source
            if (self._agency_assessment and result.get("success") is not None
                    and _dispatch_source != "self_exploration"):
                try:
                    assessment = await self._agency_assessment.assess(result)
                    result["assessment"] = {
                        "score": assessment["score"],
                        "reflection": assessment["reflection"],
                        "enrichment": assessment["enrichment"],
                        "mood_delta": assessment["mood_delta"],
                        "threshold_direction": assessment["threshold_direction"],
                    }
                except Exception:
                    pass

            # Same empty-payload guard as the normal-path publisher above
            _auto_helper = str(result.get("helper") or "").strip()
            if not _auto_helper:
                logger.debug(
                    "[TitanHCL] Skipping AUTONOMY ACTION_RESULT with empty "
                    "helper — posture=%s success=%s",
                    result.get("posture"), result.get("success"))
            else:
                self.bus.publish(make_msg(ACTION_RESULT, "core", "all", result))
                logger.info("[TitanHCL] AUTONOMY ACTION: %s → %s (success=%s)",
                            result.get("posture"), _auto_helper,
                            result.get("success"))

            # Feed action result to cognitive_worker for OBSERVATION (closed
            # loop). Phase D (D-SPEC-116): repointed from the retired
            # spirit_worker (X-engagement→MSL leg; others superseded/re-homed).
            try:
                self.bus.publish(make_msg(
                    OUTER_OBSERVATION, "core", "cognitive_worker", {
                        "action_type": result.get("helper", ""),
                        "result": result,
                        "source": payload.get("source", "nervous_system"),
                    }))
            except Exception as e:
                logger.warning("[TitanHCL] OUTER_OBSERVATION publish error: %s", e)

    def _handle_agency_query(self, msg: dict) -> None:
        """Handle agency status queries.

        Lifted verbatim from v5_core.py:1565-1580.
        """
        payload = msg.get("payload", {})
        action = payload.get("action", "")
        rid = msg.get("rid")
        src = msg.get("src", "")

        if action == "get_agency_stats":
            stats = {}
            if self._agency:
                stats["agency"] = self._agency.get_stats()
            if self._agency_assessment:
                stats["assessment"] = self._agency_assessment.get_stats()
            if self._interface_advisor_reader:
                # v1.8.5 §4.H (D-SPEC-59): stats come from SHM via the
                # compat-shim get_stats() on InterfaceAdvisorStateReader.
                stats["advisor"] = self._interface_advisor_reader.get_stats()
            self.bus.publish(make_msg(bus.RESPONSE, "core", src, stats, rid))

    # ------------------------------------------------------------------
    # Background Loops — Social, Meditation, Outer Trinity
    # ------------------------------------------------------------------
    # _v4_event_bridge_loop + _trinity_snapshot_loop EXTRACTED 2026-05-21
    # (RFP_phase_c_titan_hcl_cleanup Phase A+B) → observatory_worker
    # (titan_hcl/modules/observatory_worker.py). The parent no longer
    # bridges bus events to OBSERVATORY_EVENT nor writes ObservatoryDB history;
    # both are owned by the L3 observatory_worker. No resurrection.
    # ------------------------------------------------------------------

    # _social_engagement_loop REMOVED 2026-05-21 (D-SPEC-106). It was dead
    # code — its create_task was commented out at boot (never scheduled), and
    # mention polling + engagement is owned by social_worker (per-Titan
    # canonical-poller mode: MENTION_RECEIVED / SOCIAL_RECEIVED dispatch +
    # SHM publish, see titan_hcl/modules/social_worker.py). The parent's
    # ad-hoc SocialManager.monitor_and_engage poll loop (lifted from the
    # retired v5_core.py) is superseded; no resurrection.
    # ------------------------------------------------------------------
    # _meditation_loop REMOVED — extracted to meditation_worker subprocess per
    # rFP_titan_hcl_l2_separation_strategy §4.D + D-SPEC-57 (SPEC v1.8.3,
    # 2026-05-15). The dual-trigger orchestrator (bus MEDITATION_REQUEST +
    # fixed timer fallback), memory readiness probe, run_meditation_async
    # 300s G19 work-RPC, observatory record, MEDITATION_COMPLETE 3-target
    # fan-out, EPOCH_TICK emit — all now live in
    # titan_hcl/modules/meditation_worker.py. The _meditation_queue
    # pre-subscription in kernel.__init__ is RETIRED too (next chunk).
    # ~225 LOC retired from this file per D-SPEC-57 Q5 aggressive-cleanup.
    # Studio art generation now lives in meditation_worker as well — wired
    # via studio_proxy.generate_meditation_art_with_completion (v1.9.4 §4.K
    # D-SPEC-63 closure of post-§4.D regression).
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Agent (Agno) — delegates to LLM module
    # ------------------------------------------------------------------

    def create_agent(self):
        """Create the Agno sovereign agent.

        In V3 this delegates to the LLM module via proxy.
        Lifted verbatim from v5_core.py:2348-2356.
        """
        # D-SPEC-72: factory moved to modules/agno_agent_factory.py. Under
        # the Phase C path this method is rarely called (api_worker routes
        # /chat through agno_proxy → agno_worker.arun); retained for
        # back-compat with the legacy MCP / direct-invocation paths.
        from titan_hcl.modules.agno_agent_factory import create_agent as _create_agent
        return _create_agent(self)

    # ------------------------------------------------------------------
    # Status / Health
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Boot orchestration — PLAN §3 D10 5-phase order
    # ------------------------------------------------------------------

    async def boot(self) -> None:
        """Full plugin boot: kernel L0 + module wiring + L2/L3 loops.

        Phase order per PLAN §3 D10 preserves the boot-ordering invariants
        of legacy TitanCore.boot() (v5_core.py:219-311) while cleanly
        separating kernel (L0) from plugin (L1-L3) responsibilities:

          Phase 1 — _register_modules() — plugin registers the L1/L2/L3
                    module catalog with guardian (kernel-owned supervisor).
                    MUST run BEFORE kernel.start_modules() so every
                    ModuleSpec is known before any child process launches.
          Phase 2 — kernel.boot() — L0 async boot: bus._poll_fn hookup,
                    guardian_loop task, heartbeat_loop task, trinity shm
                    writer thread, spirit-fast writer hook (S3b).
          Phase 3 — kernel.start_modules() — guardian.start_all() actually
                    launches all autostart=True modules.
          Phase 4 — _create_proxies() — builds 8 proxies + 3 V2 aliases
                    + wires metabolism/sovereignty/studio/social.
          Phase 5 — Observatory + Agency + all plugin loops — event_bus,
                    observatory_db, FastAPI app, agency boot, reflex
                    collector, trinity snapshot loop, agency_loop (if
                    agency enabled), sovereignty_loop (if tracker wired),
                    meditation_loop, outer_trinity, v4_event_bridge.

        Mirrors v5_core.py:219-311 semantics. Byte-equivalent with
        kernel_plugin_split_enabled=true vs false.
        """
        boot_start = time.time()

        # ── Microkernel v2 §A.4 (S5) — pass plugin reference to kernel ─
        # The kernel_rpc server (started by kernel.boot() if flag is on)
        # needs a reference to this plugin instance to resolve method
        # paths against. Set BEFORE kernel.boot() so the RPC server can
        # bind cleanly when api_subprocess connects (which happens during
        # kernel.start_modules() below).
        self.kernel._plugin_ref = self

        # ── Phase 1: Module catalog ownership (Phase 6 / D-SPEC-135) ─
        # Pre-Phase-6 the plugin called self._register_modules() +
        # self._register_api_subprocess_module() to populate the in-process
        # Guardian's catalog. Under Phase 6 the catalog lives in
        # titan_hcl/module_catalog.py and is invoked exclusively by
        # scripts/guardian_hcl.py inside the guardian_hcl process. The
        # plugin no longer registers modules — guardian_hcl is already
        # supervising them by the time titan_hcl boots (INV-PROC-3).

        # ── Phase 2: Kernel L0 async boot ──────────────────────────
        await self.kernel.boot()

        # ── Phase 3: start_modules (no-op under Phase 6) ───────────
        # GuardianHCLClient.start_all() is a no-op — autostart modules
        # are already running in guardian_hcl process.
        self.kernel.start_modules()

        # ── Phase 4: Create proxies + wire L2/L3 subsystems ────────
        self._create_proxies()

        # Phase 2.5.E SocialXGateway reader binding (2026-05-23 bug-fix).
        # Construct the gateway READER for the /v4/community-engagement-stats
        # endpoint exposed at dashboard.py:4835. T1 owns social_x.db locally;
        # T2/T3 still construct the reader (same data dir layout) — the
        # gateway tolerates missing tables and returns zero/empty stats, so
        # the endpoint replies cleanly on every Titan. Construct here at boot
        # so the dashboard never sees `_social_x_gateway_reader is None`
        # under the kernel_rpc EXPOSED_METHODS allowlist (core/kernel.py:418).
        try:
            from titan_hcl.logic.social_x_gateway import SocialXGateway
            self._social_x_gateway_reader = SocialXGateway()
            logger.info("[TitanHCL] SocialXGateway reader bound — "
                        "/v4/community-engagement-stats endpoint live")
        except Exception as _sxg_err:
            logger.warning(
                "[TitanHCL] SocialXGateway reader init failed (endpoint "
                "will 503): %s", _sxg_err)
            # _social_x_gateway_reader stays None from __init__; endpoint
            # returns clean 503 instead of AttributeError.

        # ── Phase 5: Observatory + Agency + plugin-owned loops ─────
        # EventBus + ObservatoryDB (must exist before observatory app).
        from titan_hcl.api.events import EventBus
        self.event_bus = EventBus()

        # Observatory DB for persistent historical metrics.
        # rFP_universal_sqlite_writer Phase 2 — per-process singleton.
        from titan_hcl.utils.observatory_db import get_observatory_db
        self._observatory_db = get_observatory_db()
        self.event_bus.attach_db(self._observatory_db)

        # rFP_trinity_130d_awakening Phase 2 — start InnerPerceptionState.
        # AmbientChangeMonitor samples (cpu_thermal, circadian) at 1Hz on
        # a daemon thread; AudioPerception / VisualPerception are populated
        # by ``_notify_expressive_create`` at every record_expressive site.
        try:
            from titan_hcl.logic.inner_perception import InnerPerceptionState
            from titan_hcl.utils import system_sensor as _sys_sensor

            def _ambient_sampler():
                # Both producers already exist on system_sensor (utils/system_sensor.py).
                # cpu_thermal ∈ [0,1], circadian_phase ∈ [0,1]; sum ∈ [0,2].
                return (
                    _sys_sensor.get_cpu_thermal(),
                    _sys_sensor.get_circadian_phase(),
                )

            self._inner_perception = InnerPerceptionState(_ambient_sampler)
            self._inner_perception.start()
            # Register the obs_db record_expressive hook ONCE — every art /
            # audio / music / text emission flows through record_expressive,
            # so this single registration covers all callers (helpers,
            # meditation, future writers). Best-effort; hook errors do NOT
            # propagate into archival.
            try:
                self._observatory_db._on_expressive_create_hook = (
                    self._inner_perception.notify_create)
            except Exception:
                pass
            logger.info("[TitanHCL] InnerPerceptionState started "
                        "(ambient=1Hz; audio/visual via obs_db hook)")
        except Exception as _ip_err:
            logger.warning("[TitanHCL] InnerPerceptionState start failed: %s",
                           _ip_err, exc_info=True)
            self._inner_perception = None

        # rFP_trinity_130d_awakening Phase 2 — outer-spirit history aggregator.
        # ExperientialMemory access via cognitive_worker's coordinator (lazy
        # lookup so we don't fail boot if cognitive_worker isn't up yet).
        # Phase B.5: spirit_proxy retired — the coordinator lives in
        # cognitive_worker post-D8-3; we look it up directly on plugin
        # state. ExperientialMemory was previously surfaced via
        # spirit_proxy._coordinator attribute (legacy V3-inline path);
        # under Phase C the coordinator instance lives in cognitive_worker
        # and is only exposed via its SHM publishers — outer-spirit
        # history aggregator now reads from in-proc coordinator handle if
        # available else None (None disables aggregation gracefully).
        try:
            from titan_hcl.logic.outer_spirit_history import OuterSpiritHistory

            def _e_mem_lookup():
                # Phase C: coordinator instance owned by cognitive_worker
                # subprocess; not accessible from parent. Surface inline
                # only if a parent-side coordinator handle is wired
                # (legacy Phase A+B path).
                coord = getattr(self, "_coordinator", None)
                if coord is None:
                    return None
                return getattr(coord, "_experiential_memory", None) or getattr(coord, "e_mem", None)

            self._outer_spirit_history = OuterSpiritHistory(_e_mem_lookup)
            logger.info("[TitanHCL] OuterSpiritHistory started "
                        "(env_adapt + graceful_rest + circadian + dream_recall)")
        except Exception as _osh_err:
            logger.warning("[TitanHCL] OuterSpiritHistory start failed: %s",
                           _osh_err, exc_info=True)
            self._outer_spirit_history = None

        # Microkernel v2 §A.4 (S5) — flag-aware API path:
        #   flag on  → api_subprocess (Guardian-spawned in Phase 3 above)
        #              owns uvicorn; SKIP legacy in-process path here.
        #   flag off → legacy in-process path runs (byte-identical pre-S5).
        api_subprocess_active = self._full_config.get("microkernel", {}).get(
            "api_process_separation_enabled", False)
        api_cfg = self._full_config.get("api", {})
        if api_cfg.get("enabled", True) and not api_subprocess_active:
            self._create_observatory_app(api_cfg)
            asyncio.get_event_loop().create_task(self._start_observatory(api_cfg))
        elif api_subprocess_active:
            logger.info(
                "[TitanHCL] API subprocess mode active — "
                "legacy _start_observatory skipped (Microkernel v2 §A.4)")

        # Agency Module (autonomous action pipeline)
        # 2026-05-19 BOOT_TRACE — confirms boot reaches Phase 5 agency block
        logger.warning("[TitanHCL] [BOOT_TRACE] calling _boot_agency() now")
        self._boot_agency()
        logger.warning(
            "[TitanHCL] [BOOT_TRACE] _boot_agency() returned, "
            "self._agency=%s self._agency_queue=%s",
            type(getattr(self, "_agency", None)).__name__,
            type(getattr(self, "_agency_queue", None)).__name__)

        # Reflex Collector (Sovereign Tool System)
        self._boot_reflex_collector()

        # Trinity snapshot history → observatory_worker (RFP Phase B, 2026-05-21).

        # Agency bus listener (IMPULSE → INTENT → helper execution → ACTION_RESULT)
        if self._agency:
            # 2026-05-19 BOOT_TRACE — confirms create_task is reached
            logger.warning(
                "[TitanHCL] [BOOT_TRACE] scheduling _agency_loop task — "
                "self._agency=%s", type(self._agency).__name__)
            asyncio.get_event_loop().create_task(self._agency_loop())
        else:
            logger.warning(
                "[TitanHCL] [BOOT_TRACE] _agency_loop NOT scheduled — "
                "self._agency is None (Agency boot failed or disabled)")


        # Chat bus bridge RETIRED in Phase C v1.17.0 (D-SPEC-72) — replaced
        # by agno_worker subprocess + agno_proxy.chat() per SPEC §9.B. The
        # api_subprocess now publishes CHAT_REQUEST with dst="agno_worker"
        # (not dst="chat_handler"), so this parent-side bridge was dead code.

        # Guardian bus bridge — parent-side handler for QUERY dst="guardian"
        # admin operations from api_subprocess. Same Phase C "events/commands
        # over bus" pattern as _chat_handler_loop. Closes BOTH
        # BUG-HOT-RELOAD-CODE-LOADING AND BUG-GUARDIAN-CONTROL-COMMANDS-ORPHAN
        # in one architectural fix: api_subprocess can now invoke
        # guardian.restart_module() (with optional `start_method="spawn"`
        # override for true code reload) over the bus, AND the historically
        # orphan GUARDIAN_{START,STOP,RESTART}_REQUEST publishers from
        # command_sender.py finally have a receiver. SPEC §[KERNEL_RPC] +
        # Phase C G19 — work-RPC pattern with rid-routed RESPONSE.
        asyncio.get_event_loop().create_task(self._guardian_handler_loop())

        # Sovereignty listener RETIRED v1.8.3 §4.L (D-SPEC-57, 2026-05-15) —
        # SOVEREIGNTY_EPOCH is now consumed by sovereignty_worker subprocess
        # directly (registered via guardian.register(ModuleSpec(name="sovereignty",
        # ...)) above). spirit_worker.py:3845 already emits with dst="sovereignty",
        # so no producer change needed.

        # Meditation cycle (memory consolidation, mempool scoring, Cognee cognify)
        # REMOVED — extracted to meditation_worker subprocess per D-SPEC-57.

        # Phase C dissolution (2026-05-22): the OUTER_SOURCES_SNAPSHOT broadcast
        # (_publish_outer_sources_loop) is RETIRED — it carried STATE over the
        # bus (G18 violation) and its only live consumers (the Python outer_*
        # workers) are unspawned under l0_rust_enabled=true. Outer source data
        # now flows SHM-direct: each sidecar assembles via the §9.F
        # outer_source_assembly helper + re-homed breath trackers (below).

        # Phase C C-S6: outer sensor refresh sidecars (SPEC §9.D).
        # These are in-process asyncio tasks that snapshot the canonical
        # `sources` dict (per `_gather_outer_trinity_sources`) into msgpack-
        # encoded sensor_cache_outer_*.bin shm slots that titan-outer-
        # {body,mind,spirit}-rs Rust daemons read. Sidecars run
        # UNCONDITIONALLY (not flag-gated per PLAN §1.1 item 9 + SPEC §9.D
        # + §11.B line 1236) — writing to a slot Rust daemons don't yet
        # read is zero-cost and gives a soak window before C-S7 first
        # flag-flip. Each sidecar has its own in-process restart loop
        # (in-process exception handler per SPEC §11.B line 1236).
        try:
            import threading
            import traceback as _tb_mod
            from titan_hcl.logic import outer_body_sensor_refresh as _obsr
            from titan_hcl.logic import outer_mind_sensor_refresh as _omsr
            from titan_hcl.logic import outer_spirit_sensor_refresh as _ossr
            from titan_hcl.logic.outer_source_assembly import (
                OuterSourceContext, OuterHeavyStatsRefresher)
            from titan_hcl.logic.outer_sidecar_providers import (
                make_outer_body_provider, make_outer_mind_provider,
                make_outer_spirit_provider)
            from titan_hcl.core.state_registry import resolve_titan_id as _rtid
            # Phase C dissolution (2026-05-22): each sidecar assembles its source
            # data SHM-direct via the §9.F helper + re-homed breath trackers,
            # replacing the parent _gather_outer_sources + the OUTER_SOURCES_SNAPSHOT
            # bus broadcast (G18). One in-process heavy-stats refresher (G20) feeds
            # the DB-count keys + OSH's heavy dream_recall off the sidecar hot path.
            _osa_tid = _rtid()
            _osa_data_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "data")
            self._outer_heavy_refresher = OuterHeavyStatsRefresher(
                titan_id=_osa_tid, data_dir=_osa_data_dir,
                is_x_gateway=(str(_osa_tid).upper() == "T1"),
                outer_spirit_history=self._outer_spirit_history)
            self._outer_heavy_refresher.start()
            # inner_perception_state.bin publisher (C.7) — InnerPerception is
            # parent-resident hardware; publish it SHM-direct so mind_worker
            # reads it without the retired OUTER_SOURCES_SNAPSHOT broadcast (G18,
            # G21 single-writer = this parent thread).
            try:
                from titan_hcl.logic.inner_perception_state_publisher import (
                    InnerPerceptionStatePublisher)
                self._inner_perception_state_pub = InnerPerceptionStatePublisher(
                    titan_id=_osa_tid)

                def _inner_perception_publish_loop() -> None:
                    time.sleep(15)
                    while True:
                        try:
                            ip = getattr(self, "_inner_perception", None)
                            self._inner_perception_state_pub.publish(ip)
                        except Exception as _ipe:
                            logger.debug(
                                "[InnerPerceptionStatePub] loop: %s", _ipe)
                        time.sleep(1.0)
                threading.Thread(
                    target=_inner_perception_publish_loop,
                    name="inner_perception_state_publisher",
                    daemon=True).start()
            except Exception as _ip_pub_err:
                logger.error(
                    "[TitanHCL] inner_perception_state publisher boot failed "
                    "— inner_mind feeling[5/7/9] fall back to defaults: %s",
                    _ip_pub_err, exc_info=True)
            # expression_state.bin publisher: l0_rust is permanently true
            # (Phase C canonical) → expression_worker owns the slot (G21), so
            # the legacy parent publisher loop was retired (config-shm Phase D).
            # The parent keeps ONLY a low-rate bridge of translator stats to
            # expression_worker (translator stays in the main plugin — action
            # selection is on the impulse hot path; RPC would add latency).
            self._expression_state_pub = None

            def _expression_translator_stats_emit_loop() -> None:
                time.sleep(20)  # let translator finish boot warm-up
                while True:
                    try:
                        tr = getattr(self, "_expression_translator", None)
                        if tr is not None:
                            try:
                                tstats = tr.get_stats() or {}
                            except Exception as _gs_err:
                                logger.debug(
                                    "[ExpressionTranslatorStatsBridge] "
                                    "get_stats raised: %s", _gs_err)
                                tstats = {}
                            try:
                                par30 = float(
                                    tr.posture_authenticity_ratio_30())
                            except (AttributeError, TypeError, ValueError):
                                par30 = 0.0
                            payload = dict(tstats)
                            payload["posture_authenticity_ratio_30"] = par30
                            payload["ts"] = time.time()
                            try:
                                self.bus.publish(make_msg(
                                    bus.EXPRESSION_TRANSLATOR_STATS_UPDATED,
                                    src="parent",
                                    dst="expression_worker",
                                    payload=payload,
                                ))
                            except Exception as _pub_err:
                                logger.debug(
                                    "[ExpressionTranslatorStatsBridge] "
                                    "publish failed: %s", _pub_err)
                    except Exception as _outer_err:
                        logger.debug(
                            "[ExpressionTranslatorStatsBridge] loop: %s",
                            _outer_err)
                    time.sleep(5.0)
            threading.Thread(
                target=_expression_translator_stats_emit_loop,
                name="expression-translator-stats-bridge",
                daemon=True).start()
            logger.info(
                "[TitanHCL] ExpressionTranslator stats bridge started "
                "(5s; l0_rust=true L3 closure)")
            _osa_ctx = OuterSourceContext(
                shm_bank=self._shm_reader_bank, titan_id=_osa_tid,
                data_dir=_osa_data_dir, start_time=self._start_time,
                bus_stats_provider=lambda: self.bus.stats,
                observatory_db=self._observatory_db,
                heavy_stats=self._outer_heavy_refresher.cache,
                outer_spirit_history=self._outer_spirit_history)
            self._outer_body_sensor_sidecar = _obsr.OuterBodySensorRefresh(
                sources_provider=make_outer_body_provider(
                    _osa_ctx, _obsr.SOURCE_KEYS))
            self._outer_mind_sensor_sidecar = _omsr.OuterMindSensorRefresh(
                sources_provider=make_outer_mind_provider(
                    _osa_ctx, _omsr.SOURCE_KEYS))
            self._outer_spirit_sensor_sidecar = _ossr.OuterSpiritSensorRefresh(
                sources_provider=make_outer_spirit_provider(
                    _osa_ctx, _ossr.SOURCE_KEYS))

            # rFP_phase_c_close_all_runtime_gaps chunk 9H: each sidecar
            # runs in its own daemon thread with its own asyncio loop.
            # The previous architecture (3 × `asyncio.create_task` on the
            # shared main loop) deterministically reproduced a scheduling
            # bug under titan-kernel-rs where ONLY outer_body's run()
            # entered (verified live on T3 2026-05-06: body's "starting"
            # log fired 57s after task creation; mind+spirit "starting"
            # logs never appeared, no traceback). Per-thread isolation
            # sidesteps any main-loop scheduling drama. Threads are
            # `daemon=True` so they exit cleanly with the parent.
            # `_gather_outer_sources` is synchronous + reads in-process
            # state already designed for cross-thread access (proxy
            # registry uses parking_lot/threading.Lock locks).
            def _run_sidecar_thread(_sidecar, _name):
                try:
                    asyncio.run(_sidecar.run())
                except Exception:
                    logger.critical(
                        "[TitanHCL] sidecar thread %s crashed:\n%s",
                        _name, _tb_mod.format_exc(),
                    )
            for _sidecar, _name in (
                (self._outer_body_sensor_sidecar, "body"),
                (self._outer_mind_sensor_sidecar, "mind"),
                (self._outer_spirit_sensor_sidecar, "spirit"),
            ):
                _t = threading.Thread(
                    target=_run_sidecar_thread,
                    args=(_sidecar, _name),
                    name=f"outer_{_name}_sensor_refresh",
                    daemon=True,
                )
                _t.start()
            logger.info(
                "[TitanHCL] outer sensor refresh sidecars started in "
                "dedicated threads (body=%.1fs / mind=%.1fs / spirit=%.1fs cadence)",
                self._outer_body_sensor_sidecar._refresh_period_s,
                self._outer_mind_sensor_sidecar._refresh_period_s,
                self._outer_spirit_sensor_sidecar._refresh_period_s,
            )
        except Exception as e:
            logger.warning(
                "[TitanHCL] outer sensor refresh sidecar boot failed "
                "(non-fatal — Rust outer daemons read last-known): %s", e)

        # V4 event bridge → observatory_worker (RFP Phase A, 2026-05-21).

        # Social engagement: owned by social_worker (per-Titan canonical-poller
        # mention/DM polling + dispatch). The parent's _social_engagement_loop
        # was removed 2026-05-21 (D-SPEC-106) — it was never scheduled.

        # ── §G5.2 item 2 + §G12 (D-SPEC-112) — FocusPID cascade publisher ──
        # Trinity Middle-Path Homeostasis P0 §1.4: writes ``focus_input.bin``
        # at ~7.83 Hz from the 6 canonical trinity tensors. Daemons read +
        # compose into their per-tick ``enrichment_force``. Single-writer
        # per G21/INV-4.
        try:
            import threading as _focus_pid_threading
            from titan_hcl.logic.focus_pid_publisher import FocusPIDPublisher

            _bank = self._shm_reader_bank

            def _focus_trinity_reader() -> dict[str, list[float] | None]:
                def _vals(read_fn):
                    try:
                        d = read_fn()
                    except Exception:
                        return None
                    if not d:
                        return None
                    v = d.get("values")
                    return list(v) if v is not None else None

                return {
                    "inner_body": _vals(_bank.read_inner_body_5d),
                    "inner_mind": _vals(_bank.read_inner_mind_15d),
                    "inner_spirit": _vals(_bank.read_inner_spirit_45d),
                    "outer_body": _vals(_bank.read_outer_body_5d),
                    "outer_mind": _vals(_bank.read_outer_mind_15d),
                    "outer_spirit": _vals(_bank.read_outer_spirit_45d),
                }

            self._focus_pid_publisher = FocusPIDPublisher(
                trinity_reader=_focus_trinity_reader,
            )
            self._focus_pid_stop_event = _focus_pid_threading.Event()
            self._focus_pid_publisher.start_thread(self._focus_pid_stop_event)
            logger.info(
                "[TitanHCL] FocusPIDPublisher started (focus_input.bin ~7.83 Hz)"
            )
        except Exception as _e:
            logger.warning(
                "[TitanHCL] FocusPIDPublisher boot failed (non-fatal — daemons "
                "read neutral nudge): %s",
                _e,
            )

        self._background_tasks_started = True
        boot_s = time.time() - boot_start
        logger.info(
            "[TitanHCL] Async boot complete in %.2fs | Modules registered: %s",
            boot_s, list(self.guardian._modules.keys()),
        )

    def get_v3_status(self) -> dict:
        """Return V3-specific status for Observatory API.

        Lifted verbatim from v5_core.py:2362-2378.
        """
        status = {
            "version": "3.0",
            "mode": "microkernel",
            "boot_time": round(time.time() - self._start_time, 1),
            "limbo": self._limbo_mode,
            "bus_stats": self.bus.stats,
            "bus_modules": list(self.bus.modules),
            "guardian_status": self.guardian.get_status(),
        }
        # Include Agency stats if available
        if self._agency:
            status["agency"] = self._agency.get_stats()
        if self._agency_assessment:
            status["assessment"] = self._agency_assessment.get_stats()
        return status
