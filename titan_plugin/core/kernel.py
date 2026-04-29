"""
titan_plugin/core/kernel.py — TitanKernel (L0 microkernel).

Owns the foundational infrastructure that never restarts:
  - TitanBus (DivineBus — unified IPC)
  - Guardian (module supervisor)
  - StateRegister (legacy in-process state buffer, read path)
  - RegistryBank (/dev/shm state registry framework — Microkernel v2 §A.2)
  - SovereignSoul + HybridNetworkClient (identity)
  - DiskHealthMonitor + BusHealthMonitor
  - Trinity/Neuromod/Epoch shm writers (Microkernel v2 §A.2)
  - Spirit-fast 70.47 Hz shm writer hook (Microkernel v2 §A.7 / §L1)

Does NOT contain (per rFP "What L0 Does NOT Contain"):
  - LLM calls (L3 — plugin)
  - Reasoning logic (L2 — plugin)
  - Database connections (handled by L1/L2/L3 workers in their own processes)
  - HTTP API (L3 — plugin; will fully separate in S5)
  - Heavy Python libraries (torch, faiss — those live in L1/L2 workers)

This class is paired with `titan_plugin.core.plugin.TitanPlugin` which holds
the L2/L3 coordinator state (proxies, agency, observatory, agno, dream
inbox) and orchestrates the full boot sequence via `plugin.boot()`.

When `microkernel.kernel_plugin_split_enabled=false` (default), the legacy
`titan_plugin.legacy_core.TitanCore` monolith is used instead. The split
path is byte-behavior-equivalent to the legacy path.

See:
  - titan-docs/rFP_microkernel_v2_shadow_core.md §L0 + §A.1
  - titan-docs/PLAN_microkernel_phase_a_s3.md §2.1 + §3 (D1-D10)
  - titan-docs/sessions/SESSION_20260424_microkernel_phase_a_s1_s2_shipped.md
"""
import asyncio
import inspect
import hashlib
import logging
import os
import threading
import time
from typing import Optional

from titan_plugin.bus import (
    DISK_CRITICAL,
    DISK_EMERGENCY,
    DISK_RECOVERED,
    DISK_WARNING,
    MODULE_HEARTBEAT,
    SOLANA_BALANCE_UPDATED,
    DivineBus,
    make_msg,
)
from titan_plugin.guardian import Guardian

logger = logging.getLogger(__name__)


# ── Microkernel v2 §A.4 (S5) — kernel RPC exposed methods ───────────
#
# Canonical list of dotted method paths the API subprocess can call via
# the kernel_rpc Unix-socket RPC. Derived from `arch_map api-status`
# audit of titan_plugin/api/{dashboard,chat,maker,webhook}.py for
# patterns matching `<plugin_var>.X.Y(...)` and `<plugin_var>.X` runtime
# access (vs `from titan_plugin.X import ...` module imports, which are
# direct in the API process and don't need RPC).
#
# Adding a new endpoint that needs a new plugin attribute → add it
# here. The drift-detection test (tests/test_kernel_rpc_exposed_methods.py
# — commit #14) statically analyzes the API code and asserts every
# `plugin.X` access pattern is in this set.
#
# Bus-backed proxies (body, mind, spirit, memory, rl, llm, media,
# timechain, gatekeeper, mood_engine, social_graph) work cross-process
# via the bus's existing mp.Queue mechanism — they don't go through
# kernel_rpc. Listed here for completeness because endpoint code reaches
# them via `plugin.X` paths; the RPC server resolves them as references
# and the proxy's bus.request() does the actual cross-process call.
#
# Module-import paths (plugin.core, plugin.logic, plugin.utils,
# plugin.api, plugin.expressive) are NOT here — they're imported
# directly in the API process via `from titan_plugin.X import ...`.
KERNEL_RPC_EXPOSED_METHODS: frozenset[str] = frozenset({
    # Kernel-owned (L0) — Soul, Guardian, Bus, network
    "soul",
    "soul.evolve_soul",
    "soul.get_active_directives",
    "soul.current_gen",
    "soul._maker_pubkey",
    "soul._nft_address",
    "guardian",
    "guardian.get_status",
    "guardian.get_modules_by_layer",
    "guardian.layer_stats",
    "guardian.start",
    "guardian.enable",
    "bus",
    "bus.publish",
    "bus.request",
    "bus.stats",
    # Microkernel v2 Phase B.1 — Shadow Core Swap
    # Top-level and "kernel.X" prefixed paths both exposed (api_subprocess
    # accesses via titan_state.kernel.shadow_swap_orchestrate which
    # generates kernel_rpc path "kernel.shadow_swap_orchestrate").
    "shadow_swap_orchestrate",
    "shadow_swap_status",
    "hibernate_runtime",
    "restore_from_snapshot",
    "kernel_version",
    "dump_heap",
    "dump_tracemalloc",
    "dump_thread_inventory",
    "kernel",
    "kernel.shadow_swap_orchestrate",
    "kernel.shadow_swap_status",
    "kernel.hibernate_runtime",
    "kernel.restore_from_snapshot",
    "kernel.kernel_version",
    "kernel.dump_heap",
    "kernel.dump_tracemalloc",
    "kernel.dump_thread_inventory",
    # Microkernel v2 Phase B.2.1 — broker stats for /v4/state.bus_broker
    # + orchestrator's adoption-wait check + arch_map bus-status.
    "bus_broker_stats",
    "kernel.bus_broker_stats",
    # Phase A retrofit (2026-04-27) — swap-aware proxy interlock.
    # Guardian.start() inside the kernel calls these directly (no RPC),
    # but api_subprocess endpoints / arch_map may also probe via RPC.
    "is_shadow_swap_active",
    "kernel.is_shadow_swap_active",
    "wait_for_swap_completion",
    "kernel.wait_for_swap_completion",
    "network",
    "network.get_balance",
    "network.get_raw_account_data",
    "network.premium_rpc",
    "network.pubkey",
    "network.rpc_urls",
    # Plugin-owned in-memory state
    "_full_config",
    "_full_config.get",
    "_limbo_mode",
    "_start_time",
    "_is_meditating",
    "_last_commit_signature",
    "_last_execution_mode",
    "_last_research_sources",
    "_current_user_id",
    "_pending_self_composed",
    "_pending_self_composed_confidence",
    "_dream_inbox",
    "_proxies",
    "_proxies.get",
    "_agency",
    "_agency.get_stats",
    "_agency_assessment",
    "_agency_assessment.get_stats",
    "_interface_advisor",
    "_interface_advisor.get_stats",
    "_gather_current_state",
    "_get_state_narrator",
    # Plugin methods
    "reload_api",
    "get",
    "get_v3_status",
    # Plugin module-attribute references
    "maker",
    "backup",
    "backup._last_personality_date",
    "backup._last_soul_date",
    "backup._meditation_count",
    "backup.get_latest_backup_record",
    "metabolism",
    "metabolism.get_directive_alignment",
    "metabolism.get_learning_velocity",
    "metabolism.get_metabolic_health",
    "metabolism.get_social_density",
    "studio",
    "config_loader",
    "params",
    "persistence",
    "recorder",
    "recorder.buffer",
    "social",
    # Bus-backed proxies (route via bus's own IPC, not kernel_rpc;
    # listed for getattr resolution from API endpoint code)
    "memory",
    "memory._cognee_ready",
    "memory._node_store",
    "memory._node_store.items",
    "memory.fetch_mempool",
    "memory.fetch_social_metrics",
    "memory.get_coordinator",
    "memory.get_knowledge_graph",
    "memory.get_memory_status",
    "memory.get_neuromod_state",
    "memory.get_ns_state",
    "memory.get_persistent_count",
    "memory.get_reasoning_state",
    "memory.get_top_memories",
    "memory.get_topology",
    "memory.inject_memory",
    "mood_engine",
    "mood_engine.get_mood_label",
    "mood_engine.get_mood_valence",
    "mood_engine.previous_mood",
    "mood_engine.force_zen",
    "gatekeeper",
    # WebSocket EventBus (relocates to API subprocess in S5; kept here
    # for legacy in-process path compatibility and emit-from-kernel mirror)
    "event_bus",
    "event_bus.emit",
    "event_bus.subscriber_count",
})


# PERSISTENCE_BY_DESIGN: TitanKernel._config is runtime bootstrap state
# (loaded from config.toml + ~/.titan/secrets.toml); it is not self-owned
# mutable state to persist.
class TitanKernel:
    """
    L0 microkernel — foundational infrastructure that never restarts.

    Usage (via TitanPlugin — see titan_plugin.core.plugin):
        kernel = TitanKernel(wallet_path)
        plugin = TitanPlugin(kernel)
        await plugin.boot()  # orchestrates kernel.boot() + module wiring

    Kernel boot sequence (commit 2, next — see PLAN §4.1 Commit 2):
        await kernel.boot()
          └─ bus._poll_fn hookup, _guardian_loop task, _heartbeat_loop task,
             _start_trinity_shm_writer thread, _start_spirit_shm_writer hook

    This commit (#1 — kernel skeleton) lands __init__ only. Boot + loops
    arrive in commit 2.
    """

    def __init__(self, wallet_path: str):
        self._boot_start = time.time()

        # ── Load config ──────────────────────────────────────────────
        self._config = self._load_full_config()

        # ── Divine Bus ───────────────────────────────────────────────
        self.bus = DivineBus(maxsize=10000)
        # Option B (2026-04-29): both subscribers are reply_only=True so
        # they're already excluded from dst="all" broadcasts; types=[]
        # is documentation that no broadcasts are expected here. Targeted
        # dst="core" and dst="meditation" msgs (RPC, MEDITATION_REQUEST,
        # etc.) bypass the filter and reach the queue normally.
        self._core_queue = self.bus.subscribe(
            "core", reply_only=True, types=[])
        # Pre-subscribe meditation queue before Guardian starts modules
        # (spirit_worker may send MEDITATION_REQUEST during boot).
        self._meditation_queue = self.bus.subscribe(
            "meditation", reply_only=True, types=[])
        # Mainnet Lifecycle Wiring rFP (2026-04-20): subscribe eagerly so
        # SOVEREIGNTY_EPOCH messages from spirit_worker never drop on the
        # "dst without subscriber" path.
        # Option B (2026-04-29): only SOVEREIGNTY_EPOCH is consumed by
        # _sovereignty_loop in plugin.py:2218 / legacy_core.py:1488.
        from titan_plugin.bus import SOVEREIGNTY_EPOCH as _SE
        self._sovereignty_queue = self.bus.subscribe(
            "sovereignty", types=[_SE])

        # ── StateRegister (real-time state buffer) ──────────────────
        from titan_plugin.logic.state_register import StateRegister
        self.state_register = StateRegister()
        enrichment_cfg = self._config.get("spirit_enrichment", {})
        snapshot_interval = float(enrichment_cfg.get("micro_tick_interval", 10.0))
        self.state_register.start(self.bus, snapshot_interval=snapshot_interval)

        # ── Microkernel v2 Phase A §A.2 — StateRegistry bank (shm) ──
        # Owns writers/readers for /dev/shm/titan_{titan_id}/*.bin.
        # Writers are populated by background threads reading from
        # state_register (this process) and spirit_worker (subprocess).
        # Feature-gated via [microkernel] flags in titan_params.toml;
        # all default false so the shm path is byte-identical to the
        # legacy path until Maker flips a flag.
        #
        # titan_id resolution follows the canonical precedence chain
        # (data/titan_identity.json → TITAN_ID env → "T1") via
        # resolve_titan_id() — same pattern as emot_shm_protocol. This
        # is critical on T2+T3 which share /dev/shm on one VPS: without
        # the canonical resolver, both would default to "T1" and stomp
        # each other's trinity_state.bin.
        from titan_plugin.core.state_registry import RegistryBank, resolve_titan_id
        self._titan_id = resolve_titan_id()
        self.registry_bank = RegistryBank(
            titan_id=self._titan_id, config=self._config,
        )

        # ── Guardian ─────────────────────────────────────────────────
        # [guardian] toml plumbed 2026-04-16 (dead-wiring audit). Section
        # reaches Guardian which reads heartbeat_timeout_default /
        # max_restarts_in_window / restart_window / sustained_uptime_reset
        # with module constants as fallbacks.
        self.guardian = Guardian(self.bus, config=self._config.get("guardian", {}))
        # Microkernel v2 Phase A retrofit (2026-04-27): wire kernel ref into
        # Guardian so its start() / restart() consult is_shadow_swap_active().
        # Prevents proxy-driven lazy-starts from resurrecting workers mid-swap
        # (which would re-acquire DB locks → fail shadow_boot).
        self.guardian._kernel_ref = self
        # Shadow-swap completion signaling — endpoints / proxies block on this
        # event during a swap and resume when orchestrator finishes (success
        # or rollback). Initial state: set (no swap in flight = "done").
        self._shadow_swap_lock = threading.Lock()
        self._shadow_swap_active: Optional[str] = None
        self._shadow_swap_progress: dict[str, dict] = {}
        self._shadow_swap_history: dict[str, dict] = {}
        self._shadow_swap_done_event = threading.Event()
        self._shadow_swap_done_event.set()

        # ── Disk Health Monitor ──────────────────────────────────────
        # Background thread publishing DISK_WARNING/CRITICAL/EMERGENCY on
        # edge-detected transitions. On EMERGENCY, triggers graceful
        # Guardian.stop_all() via shutdown_fn hook. Protects against the
        # 2026-04-14 disk-full cascade pattern.
        from titan_plugin.core.disk_health import DiskHealthMonitor

        _disk_state_to_msg = {
            "warning": DISK_WARNING,
            "critical": DISK_CRITICAL,
            "emergency": DISK_EMERGENCY,
            "healthy": DISK_RECOVERED,
        }

        def _disk_publish(state, free_bytes):
            self.bus.publish(make_msg(
                _disk_state_to_msg[state.value], "disk_health", "all",
                {"state": state.value, "free_bytes": int(free_bytes)},
            ))

        def _disk_shutdown(reason):
            # Graceful all-worker stop — Guardian's own cleanup path runs
            # on a worker thread (commit f19a354) so this cannot deadlock
            # the event loop.
            logger.error("[TitanKernel] Initiating graceful shutdown: %s", reason)
            try:
                self.guardian.stop_all(reason=reason)
            except Exception as e:
                logger.error("[TitanKernel] shutdown stop_all error: %s", e)

        self.disk_health = DiskHealthMonitor(
            path=os.getcwd(),
            publish_fn=_disk_publish,
            shutdown_fn=_disk_shutdown,
        )
        self.disk_health.start()

        # ── Bus Health Monitor ───────────────────────────────────────
        # Tracks META_CGN_SIGNAL emission rates, queue depths, orphan
        # signals. Exposed via /v4/bus-health for session startup check.
        # Wired as module-level singleton so emit_meta_cgn_signal helper
        # can record emissions from any producer context.
        from titan_plugin.core.bus_health import BusHealthMonitor, set_global_monitor

        def _bus_health_publish(msg_type: str, payload: dict):
            try:
                self.bus.publish(make_msg(msg_type, "bus_health", "all", payload))
            except Exception as e:
                logger.debug("[BusHealth] publish error: %s", e)

        self.bus_health = BusHealthMonitor(publish_fn=_bus_health_publish)
        set_global_monitor(self.bus_health)
        logger.info("[BusHealth] monitor wired as global singleton")

        # ── Wallet Resolution & Soul ─────────────────────────────────
        self._limbo_mode = False
        self._wallet_path_raw = wallet_path
        resolved_wallet = self._resolve_wallet(wallet_path)
        if resolved_wallet is None:
            self._limbo_mode = True
            logger.warning("[TitanKernel] No keypair — LIMBO MODE")

        # Boot Soul (lightweight — just Ed25519 keys, no network calls)
        if not self._limbo_mode:
            from titan_plugin.core.soul import SovereignSoul
            from titan_plugin.core.network import HybridNetworkClient
            network_cfg = self._config.get("network", {})
            self.network = HybridNetworkClient(config=network_cfg)
            self.soul = SovereignSoul(resolved_wallet, self.network, config=network_cfg)
        else:
            self.network = None
            self.soul = None

        # Shared stop event for shm writer threads.
        self._shm_writer_stop_evt: Optional[threading.Event] = None
        # Microkernel v2 §A.4 (S5) — kernel_rpc server holder (set in
        # _start_kernel_rpc when api_process_separation_enabled flag is on).
        # _plugin_ref is set by TitanPlugin.boot() before kernel.boot() runs
        # so the RPC server can resolve method paths against the plugin.
        self._rpc_server = None
        self._plugin_ref = None

        # Process uptime anchor — consumed by _heartbeat_loop. Distinct
        # from _boot_start (which times sync __init__ duration).
        self._start_time = time.time()

        boot_ms = (time.time() - self._boot_start) * 1000
        logger.info(
            "[TitanKernel] L0 sync init complete in %.0fms (titan_id=%s, limbo=%s)",
            boot_ms, self._titan_id, self._limbo_mode,
        )

    # ------------------------------------------------------------------
    # Boot (async) — L0-only, called by TitanPlugin.boot() per D10
    # ------------------------------------------------------------------

    async def boot(self) -> None:
        """L0 async boot: bus poll hookup, guardian health loop, heartbeat,
        shm writer threads.

        Does NOT call guardian.start_all() — Plugin registers modules
        first, then calls `kernel.start_modules()`. Does NOT create the
        observatory app, event bus, or any L2/L3 loops — those are
        Plugin's responsibility.

        Called from TitanPlugin.boot() per PLAN §3 D10 boot-order
        invariants.
        """
        # ── Wire bus poll function for synchronous proxy requests ───
        # Guardian drains worker send queues whenever the bus needs to
        # dispatch a pending proxy QUERY/RESPONSE. Non-blocking.
        self.bus._poll_fn = self.guardian.drain_send_queues

        loop = asyncio.get_event_loop()

        # Guardian health monitor tick (every 1s, offloaded to thread)
        loop.create_task(self._guardian_loop())

        # Kernel heartbeat publisher (every 10s)
        loop.create_task(self._heartbeat_loop())

        # Microkernel v2 Phase A §A.2 — Trinity shm writer (daemon thread).
        self._start_trinity_shm_writer()

        # Microkernel v2 Phase A §A.7 — spirit-fast writer hook.
        # Actual 70.47 Hz writes happen inside spirit_worker subprocess
        # (D7 — 70 Hz bus traffic would flood the bus). This call is
        # architectural symmetry + boot-log visibility.
        self._start_spirit_shm_writer()

        # Microkernel v2 Phase A §A.2 part 2 (S4) — immutable identity
        # shm registry. One-shot write of titan_id + maker_pubkey +
        # kernel_instance_nonce. Stable within kernel lifetime; nonce
        # changes on every kernel restart (enables Phase B shadow-core
        # worker reattach detection per PLAN §2.4).
        self._write_identity_shm()

        # Microkernel v2 Phase A §A.4 (S5) — kernel_rpc Unix-socket server.
        # Listens on /tmp/titan_kernel_{titan_id}.sock; the API subprocess
        # (when api_process_separation_enabled flag is on) connects via
        # HMAC handshake and issues msgpack-framed RPC calls. Server runs
        # in a daemon thread so it doesn't block the async event loop.
        # No-op when flag off — legacy in-process API path stays active.
        self._start_kernel_rpc()

        # Microkernel v2 Phase B.2 — Unix-socket pub/sub broker (workers
        # connect from separate processes and survive kernel swaps).
        # Authkey derived from identity keypair via HKDF-SHA256 (no
        # persistent secret on disk; resurrection-safe). DivineBus.publish()
        # additionally fans out to broker subscribers. No-op when
        # microkernel.bus_ipc_socket_enabled=false (default).
        self._start_bus_socket_broker()

        # Microkernel v2 §A.4 S5 amendment (2026-04-25) — bus-cached state
        # publisher for the api_subprocess. Background thread; emits a
        # bulk snapshot every 2 seconds. No-op when flag off.
        self._start_state_snapshot_publisher()

        # M1-H4 (2026-04-26) — periodic SOL balance fetch + publish.
        # DivineBus race is fixed (bus.py _lock + race tests pass), but a
        # SECOND issue surfaced: even with the bus race resolved, the
        # publisher's first publish during boot still destabilizes
        # api_subprocess uvicorn on T2/T3 (T1 survives but Recv-Q
        # accumulates). Root cause not yet pinned — could be
        # multiprocessing.Queue.put racing with subprocess recv, or the
        # solana SDK's requests connection pool init colliding with
        # uvicorn startup. Default OFF until next session implements
        # delayed-first-publish (wait for Guardian api READY signal).
        # rFP §3.5 — delayed-first-publish ships in the publisher itself
        # (kernel._start_balance_publisher waits balance_publisher_first_delay_s
        # before the first emit). The flag stays opt-in for one rollout cycle:
        # flip on T1 first, soak 24h, then enable on T2/T3. Until the OBS gate
        # passes, default = OFF.
        if self._config.get("microkernel", {}).get(
                "balance_publisher_enabled", False):
            self._start_balance_publisher()
        else:
            logger.info(
                "[BalancePublisher] disabled — set "
                "microkernel.balance_publisher_enabled=true to enable "
                "(delayed-first-publish fix is in place)")

        # BUG-VAULT-COMMITS-NOT-LANDING (2026-04-29) — bus-bridge for
        # vault commits. memory_worker subprocess runs MeditationEpoch with
        # network_client=None (deployer keypair stays in main process for
        # security); on-chain TX submission is delegated to this loop. See
        # ANCHOR_REQUEST docstring in titan_plugin/bus.py for the wire
        # contract. The loop is no-op in limbo mode (no keypair) — request
        # gets a clean error response so memory_worker falls back to
        # MEDITATION_LOCAL signature.
        loop.create_task(self._anchor_request_loop())

        logger.info("[TitanKernel] Async boot complete — L0 loops running")

    def start_modules(self) -> None:
        """Start Guardian-supervised autostart modules.

        Called by TitanPlugin.boot() AFTER _register_modules so that every
        ModuleSpec is known to Guardian before any child processes launch.
        """
        self.guardian.start_all()
        logger.info(
            "[TitanKernel] Modules started: %s",
            list(self.guardian._modules.keys()),
        )

    async def shutdown(self, reason: str = "shutdown") -> None:
        """Graceful L0 stop — signal shm writer threads, stop disk health,
        stop all supervised modules.

        Safe to call repeatedly; each subsystem's stop path is idempotent.
        """
        logger.info("[TitanKernel] Shutdown initiated: %s", reason)
        if self._shm_writer_stop_evt is not None:
            self._shm_writer_stop_evt.set()
        # Microkernel v2 §A.4 (S5) — stop kernel_rpc server (idempotent).
        self._stop_kernel_rpc()
        # Microkernel v2 Phase B.2 — stop bus_socket broker (idempotent).
        self._stop_bus_socket_broker()
        # Microkernel v2 §A.4 S5 amendment — stop state snapshot publisher.
        self._stop_state_snapshot_publisher()
        # M1-H4 — stop balance publisher.
        self._stop_balance_publisher()
        try:
            self.disk_health.stop()
        except Exception as e:
            logger.warning("[TitanKernel] disk_health.stop error: %s", e)
        try:
            self.guardian.stop_all(reason=reason)
        except Exception as e:
            logger.warning("[TitanKernel] guardian.stop_all error: %s", e)

    # ------------------------------------------------------------------
    # Private L0 loops (event-loop tasks)
    # ------------------------------------------------------------------

    async def _guardian_loop(self) -> None:
        """Periodically call Guardian monitor tick + drain worker send queues.

        CRITICAL: Guardian work is inherently blocking (subprocess joins,
        queue cleanup, SAVE_NOW waits up to 30s, SIGTERM waits up to 15s).
        If these ran directly on the asyncio event loop, any worker-cleanup
        pathology would freeze uvicorn, the bus dispatcher, and every other
        coroutine — exactly the cascade observed 2026-04-14 on T1 when a
        Guardian cleanup deadlocked for 31+ minutes. We therefore offload
        monitor_tick to a worker thread so the event loop remains responsive
        no matter what Guardian is doing. drain_send_queues is fast and
        stays on-loop for lowest latency.
        """
        while True:
            try:
                await asyncio.to_thread(self.guardian.monitor_tick)
                routed = self.guardian.drain_send_queues()
                if routed > 0:
                    logger.debug("[TitanKernel] Routed %d messages from workers", routed)
            except Exception as e:
                logger.error("[TitanKernel] Guardian tick error: %s", e)
            await asyncio.sleep(1.0)

    async def _heartbeat_loop(self) -> None:
        """Publish kernel heartbeat to the bus (every 10s)."""
        while True:
            try:
                import psutil
                proc = psutil.Process()
                rss_mb = proc.memory_info().rss / (1024 * 1024)
            except Exception:
                rss_mb = 0

            self.bus.publish(make_msg(
                MODULE_HEARTBEAT, "core", "guardian",
                {"rss_mb": round(rss_mb, 1), "uptime": round(time.time() - self._start_time, 1)},
            ))
            await asyncio.sleep(10.0)

    # ------------------------------------------------------------------
    # Vault anchor bus-bridge (BUG-VAULT-COMMITS-NOT-LANDING — 2026-04-29)
    # ------------------------------------------------------------------

    async def _anchor_request_loop(self) -> None:
        """Listen for ANCHOR_REQUEST from memory_worker and submit on-chain TX.

        Microkernel v2's memory_worker subprocess runs
        ``MeditationEpoch(network_client=None)`` so meditation cycles cannot
        sign + submit TXes themselves. This loop is the bridge: memory_worker
        emits ``ANCHOR_REQUEST`` with the meditation's state_root + payload,
        kernel builds the vault commit instructions and submits via
        ``self.network`` (which holds the deployer keypair), and replies via
        ``bus.RESPONSE`` matched on the request's ``rid``.

        The loop subscribes ``reply_only=True`` so it does not receive
        broadcast state messages — only targeted ANCHOR_REQUEST frames.

        Wire contract documented at ``titan_plugin/bus.py`` near
        ``ANCHOR_REQUEST``.
        """
        from queue import Empty
        from titan_plugin.bus import ANCHOR_REQUEST

        queue = self.bus.subscribe("kernel", reply_only=True)
        logger.info("[TitanKernel] Anchor request loop subscribed (kernel queue)")

        while True:
            try:
                msg = await asyncio.to_thread(queue.get, True, 1.0)
            except Empty:
                continue
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                break
            except Exception as e:  # pragma: no cover — defensive
                logger.warning("[TitanKernel] anchor queue read error: %s", e)
                await asyncio.sleep(1.0)
                continue

            if msg.get("type") != ANCHOR_REQUEST:
                # Stale RESPONSE landing in our queue — ignore.
                continue

            try:
                await self._handle_anchor_request(msg)
            except Exception as e:
                logger.warning(
                    "[TitanKernel] anchor handler error: %s", e, exc_info=True,
                )
                # Best-effort error response so memory_worker doesn't deadlock
                # the meditation cycle waiting on a reply.
                self._publish_anchor_response(
                    msg.get("src", "memory"), msg.get("rid"), None,
                    f"handler_exception: {type(e).__name__}",
                )

    async def _handle_anchor_request(self, msg: dict) -> None:
        """Build vault commit instructions, submit TX, reply with signature."""
        payload = msg.get("payload", {}) or {}
        src = msg.get("src", "memory")
        rid = msg.get("rid")
        state_root = payload.get("state_root", "") or ""
        payload_json = payload.get("payload", "") or ""
        promoted_count = int(payload.get("promoted_count", 0) or 0)

        # Limbo / no-keypair guard. Reply with explicit error so
        # memory_worker falls back to MEDITATION_LOCAL.
        if self._limbo_mode or self.network is None:
            logger.info(
                "[TitanKernel] Anchor request received in limbo mode — "
                "no TX submission (state_root=%s, promoted=%d)",
                state_root[:16], promoted_count,
            )
            self._publish_anchor_response(src, rid, None, "limbo_mode_no_network")
            return

        vault_program_id = self._config.get("network", {}).get(
            "vault_program_id", "")
        if not vault_program_id:
            logger.info(
                "[TitanKernel] Anchor request: no vault_program_id configured "
                "— skipping TX (state_root=%s)", state_root[:16],
            )
            self._publish_anchor_response(src, rid, None, "no_vault_program_id")
            return

        # Lazy-init MeditationEpoch helper for instruction-building only.
        # ``memory_graph=None`` is safe — _build_commit_instructions only
        # uses self.network.pubkey + self._vault_program_id + 3 helpers
        # (_get_timechain_merkle / _get_vault_latest_root /
        # _compute_sovereignty_bp) which read disk + RPC, never self.memory.
        helper = getattr(self, "_anchor_helper", None)
        if helper is None:
            from titan_plugin.logic.meditation import MeditationEpoch
            helper = MeditationEpoch(
                memory_graph=None,
                network_client=self.network,
                config=self._config.get("inference", {}) or {},
            )
            helper._vault_program_id = vault_program_id
            self._anchor_helper = helper

        # Build instructions off the event loop — sync DB reads + sync httpx.
        try:
            instructions = await asyncio.to_thread(
                helper._build_commit_instructions, state_root, payload_json,
            )
        except Exception as e:
            logger.warning(
                "[TitanKernel] Anchor _build_commit_instructions failed: %s", e,
            )
            self._publish_anchor_response(
                src, rid, None, f"build_failed: {type(e).__name__}",
            )
            return

        if not instructions:
            logger.info(
                "[TitanKernel] Anchor build returned no instructions — "
                "skipping TX (state_root=%s, promoted=%d)",
                state_root[:16], promoted_count,
            )
            self._publish_anchor_response(src, rid, None, "no_instructions")
            return

        # Submit TX. send_sovereign_transaction is async + handles priority
        # fee, retries, and budget enforcement (network._check_budget_exceeded).
        try:
            tx_signature = await self.network.send_sovereign_transaction(
                instructions, priority="HIGH",
            )
        except Exception as e:
            logger.warning(
                "[TitanKernel] Anchor send_sovereign_transaction failed: %s", e,
            )
            self._publish_anchor_response(
                src, rid, None, f"send_failed: {type(e).__name__}",
            )
            return

        if tx_signature:
            logger.info(
                "[TitanKernel] Vault anchor TX landed (sig=%s, promoted=%d, "
                "root=%s)",
                tx_signature[:16], promoted_count, state_root[:16],
            )
            self._publish_anchor_response(src, rid, tx_signature, None)
        else:
            # send_sovereign_transaction returned None — likely budget
            # exceeded or RPC outage. Both already logged inside network.py.
            logger.warning(
                "[TitanKernel] Vault anchor TX returned None signature "
                "(budget exceeded? RPC down?)",
            )
            self._publish_anchor_response(src, rid, None, "tx_returned_none")

    def _publish_anchor_response(
        self,
        dst: str,
        rid: Optional[str],
        tx_signature: Optional[str],
        error: Optional[str],
    ) -> None:
        """Publish ANCHOR response (bus.RESPONSE matched on rid)."""
        from titan_plugin.bus import RESPONSE
        self.bus.publish(make_msg(
            RESPONSE, "kernel", dst,
            {"tx_signature": tx_signature, "error": error},
            rid=rid,
        ))

    # ------------------------------------------------------------------
    # Shm writer threads (Microkernel v2 §A.2 + §A.7)
    # ------------------------------------------------------------------

    def _start_trinity_shm_writer(self) -> None:
        """Microkernel v2 Phase A §A.2 — Trinity shm writer (daemon thread).

        Reads state_register's 130D felt + 30D topology + 2D journey at
        ~Schumann/9 cadence (body publish rate), assembles the 162D
        TITAN_SELF vector, and writes it to
        /dev/shm/titan_{id}/trinity_state.bin via StateRegistryWriter
        (SeqLock + persistent mmap).

        Content-hash-gated: shm seq only bumps when the assembled vector
        actually changes. Under healthy operation, this tracks the natural
        Schumann-derived body/mind publish cadence without needing fixed
        timers.

        Fallback behavior: if TRINITY_STATE feature flag is false, the
        loop still runs (and burns ~nothing) but makes no shm writes —
        readers will fall back to legacy state_register path.

        Lifted verbatim from v5_core.py:1705-1797 per PLAN §4.1 Commit 2;
        only rename: self._registry_bank → self.registry_bank (public
        attribute in kernel).
        """
        import numpy as _np

        from titan_plugin.core.state_registry import TRINITY_STATE

        # Poll slightly faster than body.publish_interval (1.15s = Schumann/9)
        # so we catch each update promptly. Content-hash gates prevents
        # spurious writes when state hasn't changed.
        poll_interval_s = 0.5
        stop_evt = self._shm_writer_stop_evt
        if stop_evt is None:
            stop_evt = threading.Event()
            self._shm_writer_stop_evt = stop_evt

        def _writer_loop() -> None:
            last_hash: Optional[bytes] = None
            consecutive_errors = 0
            # Wait a beat so state_register has its first bus tick absorbed.
            stop_evt.wait(2.0)
            while not stop_evt.is_set():
                try:
                    if not self.registry_bank.is_enabled(TRINITY_STATE):
                        # Flag off — sleep and check again. Cheap no-op.
                        stop_evt.wait(poll_interval_s)
                        continue

                    # Assemble 162D = 130D felt + 30D topology + 2D journey.
                    felt_130 = self.state_register.get_full_130dt()
                    topo_30 = self.state_register.get_full_30d_topology()
                    snapshot = self.state_register.snapshot()
                    consciousness = snapshot.get("consciousness", {}) or {}
                    journey_2 = [
                        float(consciousness.get("curvature", 0.0)),
                        float(consciousness.get("density", 0.0)),
                    ]
                    # Ensure exact-length lists (get_full_* guarantees this).
                    values = (list(felt_130)[:130]
                              + list(topo_30)[:30]
                              + journey_2[:2])
                    if len(values) != 162:
                        # Defensive — should never happen given get_full_* contracts.
                        consecutive_errors += 1
                        if consecutive_errors == 1 or consecutive_errors % 10 == 0:
                            logger.warning(
                                "[TrinityShmWrite] assembled length %d != 162; skipping",
                                len(values))
                        stop_evt.wait(poll_interval_s)
                        continue

                    arr = _np.asarray(values, dtype=_np.float32)
                    payload_bytes = arr.tobytes(order="C")
                    h = hashlib.blake2b(payload_bytes, digest_size=16).digest()
                    if h != last_hash:
                        self.registry_bank.writer(TRINITY_STATE).write(arr)
                        last_hash = h
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors == 1 or consecutive_errors % 20 == 0:
                        logger.warning(
                            "[TrinityShmWrite] iteration failed (#%d): %s",
                            consecutive_errors, e, exc_info=True)
                stop_evt.wait(poll_interval_s)

        t = threading.Thread(
            target=_writer_loop,
            daemon=True,
            name="trinity-shm-writer",
        )
        t.start()
        logger.info(
            "[TitanKernel] Trinity shm writer thread started "
            "(poll=%.2fs, gate=microkernel.shm_trinity_enabled)",
            poll_interval_s,
        )

    def _start_spirit_shm_writer(self) -> None:
        """Microkernel v2 Phase A §A.7 — spirit-fast shm writer hook (S3b).

        Placeholder method for architectural symmetry with
        _start_trinity_shm_writer. The ACTUAL 70.47 Hz write happens
        inside spirit_worker subprocess (PLAN D7 — 70 Hz bus traffic
        would flood the bus; 45D tensor is already computed in-process
        at spirit_worker.py:2036 via collect_spirit_45d()).

        This method exists so:
          1. Kernel boot logs reflect all active shm paths (visibility).
          2. Future refactors or flag flips have a single kernel-side
             hook to attach to (e.g., to aggregate spirit-fast seq metrics
             in /v4/kernel-status).
          3. Symmetry: Trinity/Neuromod/Epoch writers all announced at
             kernel boot; spirit-fast follows the same shape.

        Emits a single INFO log on boot noting the flag state for the
        current kernel process.
        """
        flag = (
            self._config.get("microkernel", {}).get("shm_spirit_fast_enabled", False)
        )
        logger.info(
            "[TitanKernel] Spirit-fast shm writer: owned by spirit_worker "
            "(config microkernel.shm_spirit_fast_enabled=%s)",
            flag,
        )

    def _write_identity_shm(self) -> None:
        """Microkernel v2 Phase A §A.2 part 2 (S4) — immutable identity shm.

        Writes [titan_id:32 | maker_pubkey:32 | kernel_instance_nonce:32]
        to /dev/shm/titan_{id}/identity.bin exactly once at kernel boot.

        - titan_id + maker_pubkey are stable across kernel restarts.
        - kernel_instance_nonce is random per boot (secrets.token_bytes).
          Enables (a) worker reattach detection for Phase B shadow-core
          swap, (b) external monitoring distinguishing "same Titan, new
          kernel instance" from "same running kernel", (c) cross-process
          consistency checks for child processes.

        Feature-flag gated. No-op if shm_identity_enabled=false.

        Per PLAN §2.4: nonce never persists to disk — ephemeral-per-kernel
        by design. self._kernel_instance_nonce is set as a side effect for
        future API exposure (/v4/kernel-status can surface the nonce).
        """
        import secrets
        import numpy as _np

        from titan_plugin.core.state_registry import IDENTITY

        if not self.registry_bank.is_enabled(IDENTITY):
            logger.info(
                "[TitanKernel] Identity shm writer skipped "
                "(microkernel.shm_identity_enabled=False)")
            return

        # 32B titan_id (UTF-8, NUL-padded)
        tid_bytes = self.titan_id.encode("utf-8")[:32]
        tid_bytes = tid_bytes + b"\x00" * (32 - len(tid_bytes))

        # 32B maker_pubkey (Ed25519 raw; zero-filled if no maker_pubkey)
        mk_bytes = b"\x00" * 32
        if self.soul is not None:
            try:
                maker_pk = getattr(self.soul, "_maker_pubkey", None)
                if maker_pk is not None:
                    raw = bytes(maker_pk)
                    mk_bytes = raw[:32] + b"\x00" * max(0, 32 - len(raw))
            except Exception as e:
                logger.warning(
                    "[TitanKernel] maker_pubkey serialization failed: %s", e)

        # 32B kernel instance nonce — random per boot
        self._kernel_instance_nonce = secrets.token_bytes(32)

        payload = tid_bytes + mk_bytes + self._kernel_instance_nonce  # 96B
        arr = _np.frombuffer(payload, dtype=_np.uint8)
        try:
            self.registry_bank.writer(IDENTITY).write(arr)
            logger.info(
                "[TitanKernel] Identity shm written "
                "(titan_id=%s, maker_pubkey=%s..., kernel_nonce=%s...)",
                self.titan_id,
                mk_bytes[:4].hex(),
                self._kernel_instance_nonce[:4].hex(),
            )
        except Exception as e:
            logger.warning(
                "[TitanKernel] Identity shm write failed: %s", e, exc_info=True)

    def _start_kernel_rpc(self) -> None:
        """Microkernel v2 Phase A §A.4 (S5) — kernel_rpc Unix-socket server.

        Listens on /tmp/titan_kernel_{titan_id}.sock for the API subprocess.
        Per-boot 32-byte authkey at /tmp/titan_kernel_{titan_id}.authkey.
        HMAC-SHA256 challenge-response on connect; msgpack-framed
        request/response thereafter.

        Feature-flag gated. No-op when api_process_separation_enabled=false
        (legacy in-process API path stays active).

        Server runs in a daemon thread (KernelRPCServer.serve_forever) so
        the async event loop is never blocked by accept()/recv() calls.

        Sets self._rpc_server (for stop_kernel_rpc) and self._plugin_ref
        (set by TitanPlugin.boot() before this method runs).
        """
        if not self._config.get("microkernel", {}).get(
                "api_process_separation_enabled", False):
            logger.info(
                "[TitanKernel] kernel_rpc skipped "
                "(microkernel.api_process_separation_enabled=False)")
            return

        if not getattr(self, "_plugin_ref", None):
            logger.warning(
                "[TitanKernel] kernel_rpc cannot start — _plugin_ref not set "
                "(TitanPlugin.boot() must assign self.kernel._plugin_ref before "
                "calling kernel.boot())")
            return

        from titan_plugin.core.kernel_rpc import KernelRPCServer
        try:
            # Capture the kernel's running asyncio loop so the RPC server
            # can await coroutine results (e.g. network.get_balance) on it
            # rather than spinning up a fresh loop per call.
            try:
                kernel_loop = asyncio.get_running_loop()
            except RuntimeError:
                kernel_loop = None
            self._rpc_server = KernelRPCServer(
                plugin_ref=self._plugin_ref,
                titan_id=self.titan_id,
                exposed_methods=KERNEL_RPC_EXPOSED_METHODS,
                kernel_loop=kernel_loop,
            )
            t = threading.Thread(
                target=self._rpc_server.serve_forever,
                daemon=True,
                name="kernel-rpc-server",
            )
            t.start()
            logger.info(
                "[TitanKernel] kernel_rpc server started "
                "(socket=%s, %d exposed methods)",
                self._rpc_server.sock_path,
                len(KERNEL_RPC_EXPOSED_METHODS))
        except Exception as e:
            logger.warning(
                "[TitanKernel] kernel_rpc start failed: %s", e, exc_info=True)
            self._rpc_server = None

    def _stop_kernel_rpc(self) -> None:
        """Graceful kernel_rpc shutdown — called from kernel.shutdown."""
        rpc = getattr(self, "_rpc_server", None)
        if rpc is not None:
            try:
                rpc.stop()
            except Exception as e:
                logger.warning(
                    "[TitanKernel] kernel_rpc stop error: %s", e)

    # ── Microkernel v2 Phase B.2 — Bus IPC broker ────────────────────────

    def _start_bus_socket_broker(self) -> None:
        """Start the Unix-socket pub/sub broker (Phase B.2). No-op when
        microkernel.bus_ipc_socket_enabled=false (default).

        Authkey is HKDF-SHA256 derived from the identity keypair stored at
        the soul's wallet path. Resurrection-safe: same identity →
        same authkey, every boot, forever. Survives kernel swaps; workers
        reconnect to the same socket path with the same key.

        Sets self._bus_broker (for stop) and attaches it to self.bus so
        DivineBus.publish() fans out to cross-process subscribers.
        """
        # Phase C C-S2: when l0_rust_enabled=true, titan-kernel-rs owns the
        # bus broker. Python skips its own BusSocketServer and connects as a
        # bus client only. Per PLAN_microkernel_phase_c_s2_kernel.md §12.3 +
        # SPEC §3.0 (default-false → byte-identical to today).
        if self._config.get("microkernel", {}).get("l0_rust_enabled", False):
            logger.info(
                "[TitanKernel] microkernel.l0_rust_enabled=true → skipping "
                "Python BusSocketServer; connecting as bus client to "
                "kernel-rs-bound socket")
            return

        if not self._config.get("microkernel", {}).get(
                "bus_ipc_socket_enabled", False):
            logger.info(
                "[TitanKernel] bus_socket broker skipped "
                "(microkernel.bus_ipc_socket_enabled=False)")
            return

        # Load identity secret bytes for HKDF derivation
        try:
            identity_secret = self._load_identity_secret_for_bus()
        except Exception as e:
            logger.warning(
                "[TitanKernel] bus_socket broker cannot start — "
                "identity secret unavailable: %s", e, exc_info=True)
            return

        from titan_plugin.core.bus_authkey import derive_bus_authkey
        from titan_plugin.core.bus_socket import BusSocketServer
        from titan_plugin.core.worker_bus_bootstrap import (
            ENV_BUS_KEYPAIR_PATH,
            ENV_BUS_SOCKET_PATH,
            ENV_BUS_TITAN_ID,
        )
        try:
            authkey = derive_bus_authkey(identity_secret, self.titan_id)
            # Phase B.2.1 fix (2026-04-27): pass in-process delivery callback
            # so worker → kernel messages reach in-process subscribers
            # (shadow_swap orchestrator etc.). publish_in_process skips the
            # broker forward to avoid a worker → broker → kernel → broker loop.
            self._bus_broker = BusSocketServer(
                titan_id=self.titan_id,
                authkey=authkey,
                on_inbound_publish=self.bus.publish_in_process,
            )
            self._bus_broker.start()
            self.bus.attach_broker(self._bus_broker)
            # Set env vars for Guardian-spawned workers — they read these
            # in worker_bus_bootstrap.setup_worker_bus to decide socket vs
            # legacy mode. Inherited by both fork and spawn ctx.Process.
            os.environ[ENV_BUS_SOCKET_PATH] = str(self._bus_broker.sock_path)
            os.environ[ENV_BUS_TITAN_ID] = self.titan_id
            os.environ[ENV_BUS_KEYPAIR_PATH] = str(self.soul.wallet_path)
            logger.info(
                "[TitanKernel] bus_socket broker started (path=%s)",
                self._bus_broker.sock_path,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[TitanKernel] bus_socket broker start failed: %s",
                e, exc_info=True)
            self._bus_broker = None

    def is_shadow_swap_active(self) -> bool:
        """Phase A retrofit (2026-04-27): True iff a shadow swap is in flight.

        Read by Guardian.start() to defer proxy lazy-starts during a swap
        (prevents mid-swap worker resurrection that holds DB locks). Also
        consulted by /maker/upgrade-status, arch_map, and any caller that
        wants to render swap state.

        Thread-safe via _shadow_swap_lock.
        """
        with self._shadow_swap_lock:
            return self._shadow_swap_active is not None

    def wait_for_swap_completion(self, timeout: float = 60.0) -> bool:
        """Phase A retrofit (2026-04-27): block until any in-flight swap
        completes (success OR rollback OR error).

        Returns True immediately if no swap is active. Returns True when
        the orchestrator finishes within `timeout` seconds. Returns False
        on timeout.

        Used by Guardian.start() so proxy lazy-start threads block during
        a swap window — they wake up automatically when the swap settles
        and proceed against whichever kernel won. Autonomous: no exception
        thrown, no user retry needed.

        Thread-safety: returns immediately if no swap (no lock-wait).
        Otherwise waits on a threading.Event signaled by _run_swap finally.
        """
        if not self.is_shadow_swap_active():
            return True
        return self._shadow_swap_done_event.wait(timeout=timeout)

    def bus_broker_stats(self) -> Optional[dict]:
        """Phase B.2.1 — return broker.stats() or None if no broker running.

        Exposed via KERNEL_RPC_EXPOSED_METHODS so api_subprocess (separate
        process per S5) can read it through the kernel_rpc proxy. The
        dashboard calls this for /v4/state.bus_broker; orchestrator's
        adoption-wait probes guardian.get_status() for the adopted set
        (see C5) and uses bus_broker stats for subscriber-count gating
        in HealthCriteria.

        Returns:
            dict with sock_path + subscriber_count + subscribers list, or
            None when bus_ipc_socket_enabled=false / broker not running.
        """
        broker = getattr(self, "_bus_broker", None)
        if broker is None:
            return None
        try:
            return broker.stats()
        except Exception:  # noqa: BLE001
            return None

    def _stop_bus_socket_broker(self) -> None:
        """Graceful broker shutdown — called from kernel.shutdown."""
        broker = getattr(self, "_bus_broker", None)
        if broker is not None:
            try:
                self.bus.detach_broker()
            except Exception:  # noqa: BLE001
                pass
            try:
                broker.stop(timeout=2.0)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[TitanKernel] bus_socket broker stop error: %s", e)
            self._bus_broker = None
            # Clear env vars so any future fork/spawn after kernel shutdown
            # falls back to legacy mode rather than connecting to a closed sock
            from titan_plugin.core.worker_bus_bootstrap import (
                ENV_BUS_KEYPAIR_PATH,
                ENV_BUS_SOCKET_PATH,
                ENV_BUS_TITAN_ID,
            )
            for k in (ENV_BUS_SOCKET_PATH, ENV_BUS_TITAN_ID, ENV_BUS_KEYPAIR_PATH):
                os.environ.pop(k, None)

    def _load_identity_secret_for_bus(self) -> bytes:
        """Read the identity keypair file and return raw secret bytes for
        HKDF input. Solana keypair JSON is an array of 64 ints (full
        secret_bytes); we use the full 64-byte content as IKM (HKDF accepts
        arbitrary length and the additional public-key half adds defense
        against any future representation change).

        Raises if soul / wallet_path is missing — caller should handle.
        """
        if self.soul is None or not getattr(self.soul, "wallet_path", None):
            raise RuntimeError("soul not booted; identity wallet path unavailable")
        import json as _json
        from pathlib import Path as _Path
        path = _Path(self.soul.wallet_path)
        if not path.exists():
            raise FileNotFoundError(f"identity keypair file not found: {path}")
        data = _json.loads(path.read_text())
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"identity keypair file shape unexpected: {path}")
        return bytes(int(b) & 0xFF for b in data)

    # ── Microkernel v2 §A.4 S5 amendment — state snapshot publisher ──
    # The api_subprocess subscribes to STATE_SNAPSHOT_RESPONSE bus messages
    # and updates its CachedState dict. Endpoint code reads from the cache
    # via TitanStateAccessor — no sync RPC needed for state reads. This
    # publisher emits a bulk snapshot periodically (every 2 seconds by
    # default) plus on-request when api_subprocess sends
    # STATE_SNAPSHOT_REQUEST at boot.
    #
    # See PLAN_microkernel_phase_a_s5_amendment.md.

    def _build_state_snapshot(self) -> dict:
        """Assemble a flat dict of {cache_key: value} for api_subprocess.

        Kernel-side authoritative read of plugin attributes. Each key
        matches the BusSubscriber's EVENT_TO_CACHE_KEY mapping in
        titan_plugin/api/bus_subscriber.py — endpoints read via
        titan_state.<sub>.<attr>, which under the hood reads
        cache.get("<sub>.<attr_path>").
        """
        plugin = getattr(self, "_plugin_ref", None)
        if plugin is None:
            return {}

        snapshot: dict = {}

        # network info — pubkey, rpc_urls, premium_rpc (no balance — too slow)
        try:
            net = getattr(plugin, "network", None)
            if net is not None:
                snapshot["network.info"] = {
                    "pubkey": str(getattr(net, "pubkey", "") or ""),
                    "rpc_urls": list(getattr(net, "rpc_urls", []) or []),
                    "premium_rpc": getattr(net, "premium_rpc", None),
                }
        except Exception as e:
            logger.warning("[StateSnapshot] network.info error: %s", e)

        # soul state — maker_pubkey, nft_address, current_gen, directives.
        # NOTE: get_active_directives may be a coroutine on some Soul impls;
        # if so we skip it (cache returns empty list — endpoint code that
        # needs directives can publish a refresh request via commands).
        try:
            soul = getattr(plugin, "soul", None)
            if soul is not None:
                directives: list = []
                try:
                    raw = getattr(soul, "get_active_directives", lambda: [])()
                    if not inspect.iscoroutine(raw):
                        directives = list(raw or [])
                except Exception:
                    pass
                snapshot["soul.state"] = {
                    "maker_pubkey": str(getattr(soul, "_maker_pubkey", "") or ""),
                    "nft_address": str(getattr(soul, "_nft_address", "") or ""),
                    "current_gen": int(getattr(soul, "current_gen", 0) or 0),
                    "active_directives": directives,
                }
        except Exception as e:
            logger.warning("[StateSnapshot] soul.state error: %s", e)

        # guardian status — module health
        try:
            guardian = getattr(plugin, "guardian", None)
            if guardian is not None and hasattr(guardian, "get_status"):
                snapshot["guardian.status"] = guardian.get_status() or {}
        except Exception as e:
            logger.warning("[StateSnapshot] guardian.status error: %s", e)

        # bus stats — backs `plugin.bus.stats` legacy callsites via _BusShim
        try:
            bus = getattr(self, "bus", None) or getattr(plugin, "bus", None)
            if bus is not None and hasattr(bus, "_stats"):
                # DivineBus._stats has published/dropped/routed counters
                snapshot["bus.stats"] = dict(bus._stats)
        except Exception as e:
            logger.warning("[StateSnapshot] bus.stats error: %s", e)

        # gatekeeper status — sovereignty / output-verifier subsystem.
        # Microkernel v2 amendment 2026-04-26: previously absent from
        # snapshot, causing /health subsystems.gatekeeper = ABSENT in
        # all observatory views. Publish minimal liveness + sovereignty
        # score so the System tab's Subsystem Health renders correctly.
        try:
            gk = getattr(plugin, "gatekeeper", None) if plugin else None
            if gk is None:
                gk = getattr(plugin, "_output_verifier", None) if plugin else None
            if gk is not None:
                snapshot["gatekeeper.status"] = {
                    "alive": True,
                    "sovereignty_score": float(
                        getattr(gk, "sovereignty_score", 0.0) or 0.0),
                    "verified_count": int(
                        getattr(gk, "verified_count", 0) or 0),
                    "rejected_count": int(
                        getattr(gk, "rejected_count", 0) or 0),
                }
        except Exception as e:
            logger.warning("[StateSnapshot] gatekeeper.status error: %s", e)

        # agency stats
        try:
            agency = getattr(plugin, "_agency", None)
            if agency is not None and hasattr(agency, "get_stats"):
                snapshot["agency.stats"] = agency.get_stats() or {}
        except Exception as e:
            logger.warning("[StateSnapshot] agency.stats error: %s", e)

        # plugin private attributes (cache-resident)
        for attr in (
            "_limbo_mode", "_dream_inbox", "_current_user_id",
            "_pending_self_composed", "_pending_self_composed_confidence",
            "_last_execution_mode", "_last_commit_signature",
            "_last_research_sources", "_start_time", "_is_meditating",
        ):
            try:
                if hasattr(plugin, attr):
                    snapshot[f"plugin.{attr}"] = getattr(plugin, attr)
            except Exception:
                pass

        # state_register-derived rich state (D1: rFP_microkernel S5 amendment).
        # OuterState lives on the kernel and aggregates data published by all
        # workers (body/mind/spirit/cgn/etc.). This is the single fastest
        # source of "live" state for the api_subprocess. Reads are O(deepcopy)
        # under a lock — typically <1ms for the full snapshot.
        try:
            sr = getattr(self, "state_register", None)
            if sr is not None:
                snap = sr.snapshot()
                # state_register.snapshot() only returns the _state dict; the
                # neuromod_state / cgn_state / expression_composites /
                # neural_nervous_system_stats live as separate attributes for
                # historical reasons. Fold them back in for /v4/state-snapshot
                # consumers (frontend single-poll architecture).
                snap["neuromod_state"] = dict(getattr(sr, "neuromod_state", {}) or {})
                snap["cgn_state"] = dict(getattr(sr, "cgn_state", {}) or {})
                snap["expression_composites"] = dict(
                    getattr(sr, "expression_composites", {}) or {})
                snap["neural_nervous_system"] = dict(
                    getattr(sr, "neural_nervous_system_stats", {}) or {})
                snapshot["state_register.full"] = snap
                snapshot["state_register.age_seconds"] = sr.age_seconds()
                # Per-domain keys aligned with TitanStateAccessor sub-accessors.
                snapshot["body.tensor"] = {
                    "tensor": list(snap.get("body_tensor", []) or []),
                    "outer": list(snap.get("outer_body", []) or []),
                    "focus": list(snap.get("focus_body", []) or []),
                    "filter_down": list(snap.get("filter_down_body", []) or []),
                    "details": dict(snap.get("body_details", {}) or {}),
                    "center_dist": float(snap.get("body_center_dist", 0.0) or 0.0),
                }
                snapshot["mind.tensor"] = {
                    "tensor": list(snap.get("mind_tensor", []) or []),
                    "outer": list(snap.get("outer_mind", []) or []),
                    "focus": list(snap.get("focus_mind", []) or []),
                    "filter_down": list(snap.get("filter_down_mind", []) or []),
                    "center_dist": float(snap.get("mind_center_dist", 0.0) or 0.0),
                }
                snapshot["spirit.tensor"] = list(snap.get("spirit_tensor", []) or [])
                snapshot["spirit.outer"] = list(snap.get("outer_spirit", []) or [])
                snapshot["spirit.consciousness"] = dict(snap.get("consciousness", {}) or {})
                snapshot["spirit.unified_spirit"] = dict(snap.get("unified_spirit", {}) or {})
                snapshot["spirit.sphere_clocks"] = dict(snap.get("sphere_clocks", {}) or {})
                snapshot["spirit.resonance"] = dict(snap.get("resonance", {}) or {})
                snapshot["spirit.observables_30d"] = list(snap.get("observables_30d", []) or [])
                snapshot["spirit.observables_dict"] = dict(snap.get("observables_dict", {}) or {})
                # rFP_observatory_data_loading_v1 §3.2 (2026-04-26):
                # state_register.metabolic is initialized to UNKNOWN/0 and
                # never updated (no producer wires it). Pull live values
                # from MetabolismController on the kernel side instead.
                # Falls back to state_register's snapshot (UNKNOWN/0) if
                # the controller isn't reachable.
                _metab_live = dict(snap.get("metabolic", {}) or {})
                try:
                    metabolism = getattr(plugin, "metabolism", None)
                    if metabolism is not None:
                        if hasattr(metabolism, "get_metabolic_tier"):
                            _metab_live["energy_state"] = str(
                                metabolism.get_metabolic_tier())
                        _bal = getattr(metabolism, "_last_balance", None)
                        if isinstance(_bal, (int, float)):
                            _metab_live["sol_balance"] = float(_bal)
                        _bal_pct = (
                            metabolism._last_balance_pct()
                            if hasattr(metabolism, "_last_balance_pct") else None)
                        if isinstance(_bal_pct, (int, float)):
                            _metab_live["balance_pct"] = float(_bal_pct)
                except Exception as _met_err:
                    logger.debug("[StateSnapshot] metabolism live read failed: %s",
                                 _met_err)
                snapshot["metabolism.state"] = _metab_live
                snapshot["neuromods.state"] = dict(getattr(sr, "neuromod_state", {}) or {})
                snapshot["cgn.state"] = dict(getattr(sr, "cgn_state", {}) or {})
                # Microkernel v2 amendment 2026-04-26: pulled from
                # SPIRIT_STATE.payload["v4"] by state_register.
                snapshot["spirit.expression_composites"] = dict(
                    getattr(sr, "expression_composites", {}) or {})
                snapshot["spirit.neural_nervous_system"] = dict(
                    getattr(sr, "neural_nervous_system_stats", {}) or {})
                # Mood label/valence + tier_info — derived from metabolic + neuromods.
                # Endpoints that read titan_state.mood_engine.X read these cache keys.
                # Use live _metab_live (populated above from MetabolismController)
                # so all downstream readers see the same fresh state.
                _metab = _metab_live
                snapshot["mood_engine.get_mood_label"] = "active"
                snapshot["mood_engine.get_mood_valence"] = float(
                    _metab.get("mood_valence", 0.5) or 0.5)
                snapshot["metabolism.get_current_state"] = _metab
                snapshot["metabolism.get_tier_info"] = {
                    "tier": str(_metab.get("energy_state", "UNKNOWN") or "UNKNOWN"),
                    "sol_balance": float(_metab.get("sol_balance", 0.0) or 0.0),
                    "balance_pct": float(_metab.get("balance_pct", 1.0) or 1.0),
                }
                # Synthesized "spirit.coordinator" — the legacy aggregate key
                # that many /v4/* endpoints expect (inner-trinity, neuromods,
                # expression-composites, hormonal-system). Until spirit_worker
                # publishes its coordinator output as a dedicated bus event,
                # we synthesize what we can from state_register-known parts.
                # Missing keys (meta_reasoning, msl, pi_heartbeat, dreaming,
                # neural_nervous_system) ship as empty dicts and the frontend
                # handles them gracefully — these will populate once the
                # per-event publisher work (separate task) lands.
                _expr = dict(getattr(sr, "expression_composites", {}) or {})
                _nns = dict(getattr(sr, "neural_nervous_system_stats", {}) or {})
                snapshot["spirit.coordinator"] = {
                    "consciousness": dict(snap.get("consciousness", {}) or {}),
                    "sphere_clocks": dict(snap.get("sphere_clocks", {}) or {}),
                    "unified_spirit": dict(snap.get("unified_spirit", {}) or {}),
                    "resonance": dict(snap.get("resonance", {}) or {}),
                    "observables": dict(snap.get("observables_dict", {}) or {}),
                    "neuromodulators": dict(getattr(sr, "neuromod_state", {}) or {}),
                    "cgn": dict(getattr(sr, "cgn_state", {}) or {}),
                    "metabolic": dict(_metab or {}),
                    # rFP_observatory_data_loading_v1 §3.2 (2026-04-26):
                    # prefer the full extended tensors (15D mind, 45D spirit)
                    # when state_register has them, fall back to base 5D.
                    # Frontend TrinityMatrix renders MiniHeatmap when len > 5;
                    # before this fix, coord.outer_trinity always used 5D so
                    # the Outer Mind / Outer Spirit heatmaps fell through to
                    # "awaiting full tensor" placeholder.
                    "outer_trinity": {
                        "body": list(snap.get("outer_body", []) or []),
                        "mind": list(
                            snap.get("outer_mind_15d") or snap.get("outer_mind", []) or []),
                        "spirit": list(
                            snap.get("outer_spirit_45d") or snap.get("outer_spirit", []) or []),
                    },
                    "trinity": {
                        "body": list(snap.get("body_tensor", []) or []),
                        "mind": list(
                            snap.get("mind_tensor_15d") or snap.get("mind_tensor", []) or []),
                        "spirit": list(
                            snap.get("spirit_tensor_45d") or snap.get("spirit_tensor", []) or []),
                    },
                    "expression_composites": _expr,
                    "neural_nervous_system": _nns,
                    # Spirit_worker-owned aggregates not yet wired to v4 block;
                    # ship full default-shaped objects (not just {}) so the
                    # frontend's React components don't crash on undefined
                    # property access while waiting for spirit_worker to
                    # publish real values via per-event publishers.
                    "meta_reasoning": {
                        "total_chains": 0,
                        "total_eurekas": 0,
                        "primitives": [],
                        "rewards": {},
                    },
                    "msl": {
                        "i_confidence": 0.0,
                        "i_depth": 0.0,
                        "i_depth_components": {
                            "source_diversity": 0.0,
                            "concept_network": 0.0,
                            "emotional_range": 0.0,
                            "wisdom_depth": 0.0,
                            "memory_bridge": 0.0,
                        },
                        "convergence_count": 0,
                        "concept_confidences": {},
                        "attention_weights": {},
                    },
                    "pi_heartbeat": {
                        "total_epochs_observed": 0,
                        "developmental_age": 0,
                        "heartbeat_ratio": 0.0,
                    },
                    "dreaming": {
                        "is_dreaming": False,
                        "cycle_count": 0,
                        "fatigue": 0.0,
                        "recovery_progress": 0.0,
                        "developmental_age": 0,
                    },
                    # Chi Life Force — placeholder shape MUST match the real
                    # per-layer shape produced by life_force.py L369-373:
                    # {raw, effective, weight, thinking, feeling, willing}.
                    # Frontend ChiLifeForce TrinityBar reads effective/weight/
                    # thinking/feeling/willing per layer; sending the wrong
                    # shape (value/components) caused the 2026-04-26
                    # weight.toFixed undefined crash. Real values populated
                    # when spirit_worker publishes CHI_UPDATED.
                    "chi": {
                        "total": 0.0,
                        "spirit": {"raw": 0.0, "effective": 0.0, "weight": 0.0,
                                   "thinking": 0.0, "feeling": 0.0, "willing": 0.0},
                        "mind":   {"raw": 0.0, "effective": 0.0, "weight": 0.0,
                                   "thinking": 0.0, "feeling": 0.0, "willing": 0.0},
                        "body":   {"raw": 0.0, "effective": 0.0, "weight": 0.0,
                                   "thinking": 0.0, "feeling": 0.0, "willing": 0.0},
                        "circulation": 0.0,
                        "state": "BOOTSTRAP",
                        "developmental_phase": "BIRTH",
                        "weights": {"spirit": 0.0, "mind": 0.0, "body": 0.0},
                        "contemplation": {
                            "active": False,
                            "phase": 0,
                            "conviction": 0.0,
                            "conviction_threshold": 0.5,
                            "mature_enough": False,
                        },
                    },
                }
                snapshot["spirit.inner_trinity"] = snapshot["spirit.coordinator"]
                snapshot["spirit.nervous_system"] = _nns or {
                    "is_firing": False,
                    "transitions": 0,
                    "steps": 0,
                    "programs": {},
                }
        except Exception as e:
            logger.warning(
                "[StateSnapshot] state_register error: %s", e, exc_info=False)

        # v3 status
        try:
            if hasattr(plugin, "get_v3_status"):
                snapshot["v3.status"] = plugin.get_v3_status() or {}
        except Exception as e:
            logger.debug("[StateSnapshot] v3.status error: %s", e)

        # config (full) — once at boot is sufficient but cheap to repeat
        try:
            snapshot["config.full"] = self._config
        except Exception:
            pass

        return snapshot

    def _publish_state_snapshot(self) -> None:
        """Build snapshot + publish on the bus. Sender thread.

        Microkernel v2 §A.4 amendment 2026-04-28: emits unconditionally so
        the in-process BusSubscriber bridge (legacy api_process_separation=
        false mode, wired in titan_plugin/api/__init__.py) receives the
        bootstrap snapshot. When api_process_separation flips to true, the
        api_subprocess receives the same snapshot via the Unix-socket
        broker — single publisher, two transport-equivalent consumers.
        Forward-compat with B.3 mp.Queue retirement + Phase C Rust bus.
        """
        try:
            from titan_plugin.bus import STATE_SNAPSHOT_RESPONSE, make_msg
            snapshot = self._build_state_snapshot()
            if not snapshot:
                return
            bus = getattr(self, "bus", None) or getattr(
                self._plugin_ref, "bus", None)
            if bus is None:
                return
            msg = make_msg(STATE_SNAPSHOT_RESPONSE, "kernel", "api", snapshot)
            bus.publish(msg)
        except Exception as e:
            logger.warning(
                "[StateSnapshot] publish failed: %s", e, exc_info=False)

    def _start_state_snapshot_publisher(self) -> None:
        """Background thread emitting state snapshots every 2 seconds.

        Plus listens for STATE_SNAPSHOT_REQUEST events from the api
        and responds immediately (bootstrap path).

        Microkernel v2 §A.4 amendment 2026-04-28: starts unconditionally.
        Whichever consumer is wired (in-process BusSubscriber bridge OR
        api_subprocess) reads the same dst="api" snapshot from the bus.
        """
        self._snapshot_stop = threading.Event()

        def _publisher_loop():
            # First emit happens immediately so api_subprocess bootstrap
            # completes ASAP after boot.
            self._publish_state_snapshot()
            while not self._snapshot_stop.wait(2.0):
                self._publish_state_snapshot()

        t = threading.Thread(
            target=_publisher_loop,
            daemon=True,
            name="state-snapshot-publisher",
        )
        t.start()
        self._snapshot_thread = t
        logger.info(
            "[StateSnapshot] publisher started — emits every 2s on bus "
            "(STATE_SNAPSHOT_RESPONSE → api)")

    def _stop_state_snapshot_publisher(self) -> None:
        evt = getattr(self, "_snapshot_stop", None)
        if evt is not None:
            evt.set()

    # ------------------------------------------------------------------
    # SOL balance publisher (M1-H4) — periodic fetch + bus emit
    # ------------------------------------------------------------------
    def _start_balance_publisher(self) -> None:
        """Background thread: fetch SOL balance every 60s + publish
        SOLANA_BALANCE_UPDATED on the bus. api_subprocess BusSubscriber
        maps this to the network.balance cache key, which NetworkAccessor.
        balance reads.

        Uses the SYNC solana.rpc.api.Client (same pattern as
        spirit_loop._maybe_anchor_trinity:L988) — NOT the async
        HybridNetworkClient — because event loops can't be safely shared
        across threads. The async client gets bound to whichever loop
        first calls it; running it from a freshly-spawned thread loop
        causes the api_subprocess to fall over (observed 2026-04-26 first
        deploy attempt: uvicorn died, port 7777 unbound, api in
        crash-loop). The sync RPC call is bounded (60s cadence) and
        cheap (~50-500ms).

        rFP_observatory_data_loading_v1 §3.5 fix (2026-04-26): first
        publish is delayed by `microkernel.balance_publisher_first_delay_s`
        (default 30s). The first deploy crashed api_subprocess uvicorn
        on T2/T3 because the kernel-side balance publish fired during the
        api boot window — multiprocessing.Queue.put + uvicorn startup +
        solana SDK requests pool init all racing for the same resources.
        Waiting until api_subprocess is reliably up (Guardian heartbeats
        the api module after ~10-15s; 30s gives safety margin) avoids
        the race entirely. Cleaner than waiting on a MODULE_READY signal
        because Guardian can also restart api mid-flight; a fresh-after-30s
        publish lands on a stable subprocess in either case.
        """
        if self._limbo_mode or self.network is None:
            logger.info(
                "[BalancePublisher] skipped — limbo mode (no network client)")
            return

        self._balance_publisher_stop = threading.Event()

        first_delay_s = float(self._config.get("microkernel", {}).get(
            "balance_publisher_first_delay_s", 30.0))
        publish_interval_s = float(self._config.get("microkernel", {}).get(
            "balance_publisher_interval_s", 60.0))

        def _balance_loop():
            # Delayed first publish (rFP §3.5) — let api_subprocess
            # uvicorn fully come up before the kernel→api bus traffic
            # spike of a sync solana RPC + bus.publish + queue put.
            if first_delay_s > 0:
                if self._balance_publisher_stop.wait(first_delay_s):
                    return  # stopped during delay
            try:
                from solana.rpc.api import Client as SolanaClient
            except Exception as e:
                logger.warning(
                    "[BalancePublisher] solana SDK unavailable: %s", e)
                return
            net_cfg = self._config.get("network", {})
            rpc_url = net_cfg.get(
                "premium_rpc_url",
                (net_cfg.get("public_rpc_urls", [
                    "https://api.mainnet-beta.solana.com"]) or [
                    "https://api.mainnet-beta.solana.com"])[0])
            client = SolanaClient(rpc_url)
            pubkey = None
            try:
                pubkey = self.network.pubkey()
            except Exception:
                pubkey = getattr(self.network, "_pubkey", None)
            if pubkey is None:
                logger.warning(
                    "[BalancePublisher] no pubkey on network client — "
                    "publisher exiting")
                return
            # First fetch after the boot-stabilization delay.
            self._publish_balance_once(client, pubkey)
            while not self._balance_publisher_stop.wait(publish_interval_s):
                self._publish_balance_once(client, pubkey)

        t = threading.Thread(
            target=_balance_loop, daemon=True, name="balance-publisher")
        t.start()
        self._balance_publisher_thread = t
        logger.info(
            "[BalancePublisher] started — first publish in %.0fs, then every "
            "%.0fs (sync solana client)",
            first_delay_s, publish_interval_s)

    def _publish_balance_once(self, client, pubkey) -> None:
        """One sync balance fetch + bus publish. Wrapped for try/except."""
        try:
            resp = client.get_balance(pubkey)
            lamports = getattr(resp, "value", 0) or 0
            balance = float(lamports) / 1_000_000_000
            payload = {"balance": balance}
            self.bus.publish(make_msg(
                SOLANA_BALANCE_UPDATED, "kernel", "all", payload))
        except Exception as e:
            logger.warning(
                "[BalancePublisher] fetch/publish failed: %s", e)

    def _stop_balance_publisher(self) -> None:
        evt = getattr(self, "_balance_publisher_stop", None)
        if evt is not None:
            evt.set()

    # ------------------------------------------------------------------
    # Read-only @property accessors (KernelView interface contract)
    # ------------------------------------------------------------------
    # Narrow, typed read surface for upper layers. Mutation of kernel
    # state goes through explicit methods (boot, start_modules, shutdown
    # — added in commit 2). See titan_plugin.core.kernel_interface.

    @property
    def config(self) -> dict:
        """Read-only view of the loaded config.

        Plugin + upper-layer read access. Kernel owns the canonical dict;
        mutation is discouraged. For hot-reload flows (CONFIG_RELOAD bus
        message), mutation happens via dedicated reload paths, not here.
        """
        return self._config

    @property
    def titan_id(self) -> str:
        """Resolved titan identifier (T1 / T2 / T3 / ...).

        Source of truth: data/titan_identity.json (canonical precedence
        chain). Set once at __init__, immutable for the process lifetime.
        """
        return self._titan_id

    @property
    def limbo_mode(self) -> bool:
        """True when no keypair could be resolved — degraded operation.

        In limbo: self.soul is None, self.network is None. Kernel still
        boots and runs L0 services (bus, guardian, shm); L2/L3 subsystems
        that depend on the wallet degrade gracefully.
        """
        return self._limbo_mode

    # ------------------------------------------------------------------
    # Microkernel v2 Phase B.1 — Shadow Core Swap (rFP §347-357)
    # ------------------------------------------------------------------

    @property
    def kernel_version(self) -> str:
        """Short identifier for the kernel code currently running.

        Used in RuntimeSnapshot.kernel_version + the system-fork TimeChain
        block (kernel_version_from / kernel_version_to). Falls back to a
        sentinel if git is unavailable.

        Cached on first access — cheap; we only need a single read per
        hibernate event.
        """
        cached = getattr(self, "_kernel_version_cache", None)
        if cached:
            return cached
        version = "unknown"
        try:
            import subprocess
            here = os.path.dirname(__file__)
            result = subprocess.run(
                ["git", "rev-parse", "--short=8", "HEAD"],
                cwd=here, capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                version = result.stdout.strip() or "unknown"
        except Exception as e:
            logger.debug("[TitanKernel] kernel_version git lookup failed: %s", e)
        self._kernel_version_cache = version
        return version

    def dump_heap(self, top_types: int = 30, top_containers: int = 20) -> dict:
        """Live heap snapshot of the parent (kernel) process.

        Aggregates `gc.get_objects()` by type and surfaces the largest
        unbounded containers. Aggregate-only — never object content.

        Cost: ~1-5 s wall-clock on a 2 GB heap. Endpoint callers MUST
        run via `asyncio.to_thread` to avoid blocking the event loop.

        Returns the dict produced by `take_heap_snapshot()` plus
        `pid` and `process="parent"`.
        """
        from titan_plugin.core.profiler import take_heap_snapshot
        snap = take_heap_snapshot(top_types=top_types,
                                   top_containers=top_containers)
        snap["pid"] = os.getpid()
        snap["process"] = "parent"
        return snap

    def dump_tracemalloc(self, top_n: int = 30,
                         key_type: str = "filename",
                         diff: bool = True) -> dict:
        """Live tracemalloc snapshot of the parent (kernel) process.

        Returns top file:line allocators by size (or by growth since
        boot if `diff=True`). Requires tracemalloc to have been started
        at boot via `[profiling] tracemalloc_enabled = true`.

        Per worker_stability_audit 2026-04-27: this is the canonical
        path to find C-level memory leaks invisible to gc.get_objects().

        Args:
          top_n:    number of top allocators to return
          key_type: "filename" or "lineno"
          diff:     if True, return growth since boot baseline

        Returns dict with: `pid`, `process="parent"`, `tracemalloc_active`,
        and either `top` (sorted by size) or `diff` (sorted by growth).
        """
        import tracemalloc as _tm
        result: dict = {
            "pid": os.getpid(),
            "process": "parent",
            "tracemalloc_active": _tm.is_tracing(),
        }
        if not _tm.is_tracing():
            result["error"] = ("tracemalloc not running — set "
                                "[profiling] tracemalloc_enabled=true + restart")
            return result
        # `_profiling_collector` is set on the plugin, not kernel; locate it
        # via the parent process's plugin_ref (set when kernel_rpc started).
        collector = None
        plugin_ref = getattr(self, "_plugin_ref", None)
        if plugin_ref is not None:
            collector = getattr(plugin_ref, "_profiling_collector", None)
        if collector is None:
            # Fall back to ad-hoc snapshot (no diff baseline available)
            snap = _tm.take_snapshot()
            stats = snap.statistics(key_type)
            result["fallback"] = "no _profiling_collector — boot-time baseline missing"
            result["top"] = [
                {"file": str(s.traceback), "size_mb": round(s.size / 1048576, 2),
                 "size_bytes": s.size, "count": s.count}
                for s in stats[:top_n]
            ]
            return result
        if diff:
            result["diff"] = collector.get_diff_stats(n=top_n, key_type=key_type)
        else:
            result["top"] = collector.get_top_stats(n=top_n, key_type=key_type)
        result["summary"] = collector.get_summary()
        return result

    def dump_thread_inventory(self) -> dict:
        """Live thread inventory of the parent (kernel) process.

        Snapshots `threading.enumerate()` and groups by name-prefix to
        surface where the parent's threads come from. Used by `arch_map
        thread-pool --parent` for the rFP A.8 §6 measurement-driven
        residency audit, and by tests/test_a8_thread_count.py as the
        regression baseline.

        Returns:
          {
            "pid": <parent pid>,
            "process": "parent",
            "total": <int>,
            "threads": [
              {"name": str, "ident": int, "daemon": bool, "alive": bool},
              ...
            ],
            "by_prefix": {<prefix>: <count>, ...},
          }
        """
        import threading as _threading
        threads = list(_threading.enumerate())
        rows = []
        by_prefix: dict[str, int] = {}
        for t in threads:
            name = t.name or "<unnamed>"
            rows.append({
                "name": name,
                "ident": t.ident,
                "daemon": bool(t.daemon),
                "alive": t.is_alive(),
            })
            # Group by trimming a per-instance ID suffix at the LAST
            # "-" / ":" / "_" separator (e.g., "shadow-swap-deadbeef" →
            # "shadow-swap"). rpartition isolates the trailing token; we
            # only collapse if the trailing token is ≥6 chars of hex.
            prefix = name
            for sep in ("-", ":", "_"):
                if sep in prefix:
                    head, _, tail = prefix.rpartition(sep)
                    if len(tail) >= 6 and all(
                            c in "0123456789abcdef" for c in tail.lower()):
                        prefix = head
                    break
            by_prefix[prefix] = by_prefix.get(prefix, 0) + 1
        return {
            "pid": os.getpid(),
            "process": "parent",
            "total": len(rows),
            "threads": rows,
            "by_prefix": dict(sorted(
                by_prefix.items(), key=lambda kv: -kv[1])),
        }

    def hibernate_runtime(
        self,
        event_id: str,
        snapshot_path: Optional[str] = None,
    ) -> str:
        """Serialize kernel runtime state to disk for a shadow swap.

        Called by the shadow-swap orchestrator (scripts/shadow_swap.py)
        after readiness wait completes and before sending HIBERNATE to
        workers. The orchestrator passes in the upgrade `event_id` (UUID4)
        so this snapshot's path links to the system-fork TimeChain block.

        Returns the path the snapshot was written to (for the orchestrator
        to pass to the shadow kernel via --restore-from).

        This method does NOT pause the bus or the kernel. Workers continue
        normally; the snapshot is just a metadata file. The actual quiescence
        happens via the HIBERNATE bus message handled per-worker.
        """
        from titan_plugin.core import shadow_protocol as sp

        # ── Collect snapshot fields ──
        soul_gen = 0
        if self.soul is not None:
            soul_gen = int(getattr(self.soul, "current_gen", 0) or 0)

        # Registry seqs (informational — shadow re-opens same /dev/shm files
        # so seqs naturally continue. Dict captured for OBS-fidelity diagnostics).
        registry_seqs: dict[str, int] = {}
        bank = getattr(self, "registry_bank", None)
        if bank is not None:
            for name, writer in getattr(bank, "_writers", {}).items():
                seq = getattr(writer, "_seq", None)
                if seq is None:
                    seq = getattr(writer, "last_seq", 0)
                try:
                    registry_seqs[name] = int(seq)
                except (TypeError, ValueError):
                    registry_seqs[name] = 0

        # Guardian module roster — capture names so shadow can verify superset
        guardian_modules: list[str] = []
        if self.guardian is not None:
            modules_dict = getattr(self.guardian, "_modules", None)
            if modules_dict:
                guardian_modules = sorted(modules_dict.keys())

        # Bus subscriber count — informational; shadow re-subscribes on boot.
        bus_subscriber_count = 0
        if self.bus is not None:
            subs = getattr(self.bus, "_subscribers", {})
            bus_subscriber_count = sum(len(v) for v in subs.values())

        snap = sp.RuntimeSnapshot(
            kernel_version=self.kernel_version,
            soul_current_gen=soul_gen,
            titan_id=self._titan_id,
            registry_seqs=registry_seqs,
            guardian_modules=guardian_modules,
            bus_subscriber_count=bus_subscriber_count,
            written_at=time.time(),
            event_id=event_id,
        )

        # ── Serialize ──
        if snapshot_path is None:
            target = sp.default_snapshot_path()
        else:
            from pathlib import Path
            target = Path(snapshot_path)
        path_written = sp.serialize_snapshot(snap, target)

        logger.info(
            "[TitanKernel] hibernate_runtime: event_id=%s kernel_version=%s "
            "soul_gen=%d titan_id=%s modules=%d registries=%d bus_subs=%d → %s",
            event_id[:8], snap.kernel_version, soul_gen, self._titan_id,
            len(guardian_modules), len(registry_seqs), bus_subscriber_count,
            path_written,
        )
        return str(path_written)

    def restore_from_snapshot(
        self,
        snapshot_path: str,
        *,
        max_age_seconds: float = 300.0,
    ) -> dict:
        """Verify a runtime snapshot at boot time (--restore-from).

        Called from `scripts/titan_main.py` during shadow-kernel boot,
        AFTER soul + guardian are initialized but BEFORE start_modules().

        Returns a dict with the snapshot's event_id + verification result.
        Caller (titan_main) is responsible for:
          - Logging the verification outcome to the brain log
          - Publishing SYSTEM_RESUMED on the bus once start_modules completes
            (we publish later, not here, so the bus is fully wired first)

        If verify_compatible() fails, this method LOGS the reason but does
        NOT raise — the kernel continues a clean boot. The orchestrator's
        rollback path triggers via HIBERNATE_CANCEL on the OLD kernel.
        Refusing to boot would leave the system without a kernel at all.
        """
        from titan_plugin.core import shadow_protocol as sp

        try:
            snap = sp.deserialize_snapshot(snapshot_path)
        except FileNotFoundError:
            logger.warning(
                "[TitanKernel] restore_from_snapshot: file not found at %s — "
                "continuing clean boot", snapshot_path,
            )
            return {"verified": False, "reason": "file_not_found", "event_id": ""}
        except Exception as e:
            logger.warning(
                "[TitanKernel] restore_from_snapshot: deserialize failed (%s) — "
                "continuing clean boot", e,
            )
            return {"verified": False, "reason": f"deserialize_error:{e}", "event_id": ""}

        # Compat check — collect target's module roster
        target_modules: list[str] = []
        if self.guardian is not None:
            target_modules = sorted(getattr(self.guardian, "_modules", {}).keys())

        ok, reason = sp.verify_compatible(
            snap,
            target_titan_id=self._titan_id,
            target_modules=target_modules,
            max_age_seconds=max_age_seconds,
        )

        if ok:
            logger.info(
                "[TitanKernel] restore_from_snapshot: VERIFIED event_id=%s "
                "kernel_version_from=%s soul_gen_from=%d age=%.1fs",
                snap.event_id[:8], snap.kernel_version, snap.soul_current_gen,
                time.time() - snap.written_at,
            )
        else:
            logger.warning(
                "[TitanKernel] restore_from_snapshot: REFUSED event_id=%s reason=%s — "
                "continuing clean boot (orchestrator should detect via missing SYSTEM_RESUMED)",
                snap.event_id[:8] if snap.event_id else "?", reason,
            )

        return {
            "verified": ok,
            "reason": reason,
            "event_id": snap.event_id,
            "kernel_version_from": snap.kernel_version,
            "soul_gen_from": snap.soul_current_gen,
            "age_seconds": time.time() - snap.written_at,
        }

    def shadow_swap_orchestrate(self, reason: str = "manual",
                                grace: float = 120.0,
                                b2_1_forced: bool = False) -> dict:
        """B.1 §7 — KICKOFF entrypoint for the shadow swap protocol.

        Spawns the orchestrator in a background thread + returns
        immediately with {event_id, outcome="started", ...}. The kernel's
        main thread continues normally (Guardian drain, RPC service,
        bus routing) so workers can ACK the swap protocol without
        starvation.

        Caller polls /maker/upgrade-status (or kernel.shadow_swap_status)
        to track progress + retrieve the final result.

        Refuses if microkernel.shadow_swap_enabled=false OR if another
        swap is currently active.
        """
        # Flag check — refuse if shadow_swap_enabled is false (default)
        flag = (self._config.get("microkernel", {})
                            .get("shadow_swap_enabled", False))
        if not bool(flag):
            return {
                "outcome": "error",
                "failure_reason": "shadow_swap_enabled_flag_off",
                "phase": "preflight",
                "event_id": "",
                "elapsed_seconds": 0.0,
            }

        # Thread-safe single-active-swap guard
        if not hasattr(self, "_shadow_swap_lock"):
            self._shadow_swap_lock = threading.Lock()
            self._shadow_swap_active = None  # event_id of running swap
            self._shadow_swap_progress = {}  # event_id → live progress dict
            self._shadow_swap_history = {}   # event_id → final result

        with self._shadow_swap_lock:
            if self._shadow_swap_active:
                return {
                    "outcome": "error",
                    "failure_reason": "another_swap_active",
                    "active_event_id": self._shadow_swap_active,
                    "phase": "preflight",
                    "event_id": "",
                }
            from titan_plugin.core import shadow_protocol as _sp
            event_id = _sp.new_event_id()
            self._shadow_swap_active = event_id
            # Phase A retrofit (2026-04-27): clear done event so threads
            # entering wait_for_swap_completion() during the swap actually
            # block. Set again in _run_swap finally on completion.
            self._shadow_swap_done_event.clear()
            self._shadow_swap_progress[event_id] = {
                "event_id": event_id,
                "outcome": "running",
                "phase": "preflight",
                "reason": reason,
                "started_at": time.time(),
                "elapsed_seconds": 0.0,
            }

        def _run_swap():
            try:
                from titan_plugin.core.shadow_orchestrator import orchestrate_shadow_swap
                result = orchestrate_shadow_swap(
                    self, reason=reason, grace=grace, event_id=event_id,
                    b2_1_forced=b2_1_forced,
                )
                with self._shadow_swap_lock:
                    self._shadow_swap_history[event_id] = result
                    # Keep only last 5 in memory; full history is in audit jsonl
                    if len(self._shadow_swap_history) > 5:
                        oldest = sorted(
                            self._shadow_swap_history.keys(),
                            key=lambda k: self._shadow_swap_history[k].get("started_at", 0),
                        )[0]
                        self._shadow_swap_history.pop(oldest, None)
            except Exception as e:
                logger.exception("[shadow_swap] background thread crashed: %s", e)
                with self._shadow_swap_lock:
                    self._shadow_swap_history[event_id] = {
                        "event_id": event_id, "outcome": "error",
                        "failure_reason": f"thread_crashed:{e}",
                        "phase": "?", "started_at": time.time(),
                    }
            finally:
                with self._shadow_swap_lock:
                    if self._shadow_swap_active == event_id:
                        self._shadow_swap_active = None
                    self._shadow_swap_progress.pop(event_id, None)
                # Phase A retrofit (2026-04-27): signal swap completion to
                # any threads blocked on wait_for_swap_completion() — proxy
                # lazy-starts deferred during the swap can now proceed.
                self._shadow_swap_done_event.set()

        thread = threading.Thread(
            target=_run_swap, daemon=False,
            name=f"shadow-swap-{event_id[:8]}",
        )
        thread.start()

        # Return kickoff result immediately — caller polls for progress.
        return {
            "event_id": event_id,
            "outcome": "started",
            "phase": "preflight",
            "reason": reason,
            "started_at": time.time(),
            "elapsed_seconds": 0.0,
            "poll_endpoint": "/maker/upgrade-status",
        }

    def shadow_swap_status(self, event_id: str = "") -> dict:
        """B.1 §7 — read live progress / final result of a shadow swap.

        With event_id: returns that swap's state (live if active, or
        from history if completed). Without: returns the most recent
        active swap, or last completed swap.
        """
        if not hasattr(self, "_shadow_swap_lock"):
            return {"outcome": "no_swaps_yet", "history": []}

        with self._shadow_swap_lock:
            active = self._shadow_swap_active
            if event_id:
                if event_id in self._shadow_swap_progress:
                    return dict(self._shadow_swap_progress[event_id])
                if event_id in self._shadow_swap_history:
                    return dict(self._shadow_swap_history[event_id])
                return {"outcome": "not_found", "event_id": event_id}
            # No event_id — return current active OR last completed
            if active and active in self._shadow_swap_progress:
                return dict(self._shadow_swap_progress[active])
            if self._shadow_swap_history:
                last_eid = max(
                    self._shadow_swap_history.keys(),
                    key=lambda k: self._shadow_swap_history[k].get("started_at", 0),
                )
                return dict(self._shadow_swap_history[last_eid])
            return {"outcome": "no_swaps_yet"}

    # ------------------------------------------------------------------
    # Static helpers (lifted from TitanCore for verbatim semantics)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_full_config() -> dict:
        """Load the full merged Titan config (config.toml + ~/.titan/secrets.toml)."""
        from titan_plugin.config_loader import load_titan_config
        return load_titan_config()

    def _resolve_wallet(self, wallet_path: str) -> Optional[str]:
        """Resolve wallet keypair (lifted from TitanCore._resolve_wallet).

        Precedence:
          1. Hardware-bound encrypted keypair (data/soul_keypair.enc) —
             decrypt via utils.crypto.decrypt_for_machine and persist to
             runtime_keypair.json.
          2. Plain wallet_path if it exists on disk.
          3. Genesis-record-only fallback (limbo mode) — returns None.
          4. wallet_path (non-existent) — degraded mode.
        """
        enc_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "soul_keypair.enc")
        )
        if os.path.exists(enc_path):
            try:
                from titan_plugin.utils.crypto import decrypt_for_machine
                with open(enc_path, "rb") as f:
                    encrypted = f.read()
                key_bytes = decrypt_for_machine(encrypted)
                import json
                runtime_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "data", "runtime_keypair.json"
                )
                with open(runtime_path, "w") as f:
                    json.dump(list(key_bytes), f)
                logger.info("[TitanKernel] Warm reboot: hardware-bound keypair decrypted.")
                return runtime_path
            except Exception as e:
                logger.warning("[TitanKernel] Hardware-bound keypair failed: %s", e)

        if os.path.exists(wallet_path):
            return wallet_path

        genesis_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "genesis_record.json")
        )
        if os.path.exists(genesis_path):
            return None

        logger.info("[TitanKernel] No keypair at %s — degraded mode.", wallet_path)
        return wallet_path
