"""
RL/Sage Module Proxy — lazy bridge to the supervised RL process.

Provides interfaces matching SageScholar, SageGatekeeper, and SageRecorder,
routing all calls through the Divine Bus to the RL module process.
This proxy alone saves ~2GB RSS by keeping TorchRL mmap out of Core.

§A.8.7 (2026-04-28) — Scholar + Gatekeeper consolidation:
    Adds `dream(...)`, `decide_execution_mode(...)`, `sovereignty_score`,
    plus parent-side `action_embedder` / `projection_layer` / `buffer`
    accessors backed by SageEncoder. Legacy `__init__.py` parent path now
    uses RLProxy for all three of `self.recorder`/`self.scholar`/`self.gatekeeper`
    when `microkernel.a8_sage_scholar_gatekeeper_subprocess_enabled=true`,
    cutting the LazyMemmapStorage (~2GB) entirely out of the host process.
    V6 path's existing wiring gains the same methods, restoring
    Sovereign/Collaborative routing (silently disabled by `agno_hooks.py:633`
    `hasattr` guard since RLProxy first shipped).

    `dream` uses `await self._bus.request_async(...)` for async-friendly
    behavior under LLM-time-scale latency (mirrors A.8.6 AgencyProxy
    pattern). `request_async` routes through the dedicated bus_ipc_pool
    (RCA 2026-04-29) — isolates LLM-time RPC waits from the default
    asyncio pool that serves Observatory snapshots. `decide_execution_mode`
    is sync (sub-second KNN+Q-net forward pass — millisecond-scale; A.8.3
    OutputVerifierProxy pattern is fine).
"""
import asyncio
import logging
from typing import Optional

from ..bus import DivineBus
from ..guardian import Guardian

logger = logging.getLogger(__name__)


class RLProxy:
    """
    Drop-in proxy for RL subsystems (Scholar, Gatekeeper, Recorder).
    Delegates to the supervised RL module via Divine Bus.
    """

    # Bus-request timeouts.
    _GATE_TIMEOUT_S = 5.0      # KNN + 2 forward passes — sub-second typical
    _DREAM_TIMEOUT_S = 120.0   # IQL training (epochs=50, batch=256) ~30-90s
    _STATS_TIMEOUT_S = 5.0
    _RECORD_TIMEOUT_S = 5.0

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        self._reply_queue = bus.subscribe("rl_proxy", reply_only=True)
        self._started = False
        # SAGE_STATS broadcast cache — refreshed every 60s by rl_worker.
        # Used by `sovereignty_score` property + dashboard reads without
        # per-call bus round-trip.
        self._stats_cache: dict = {
            "buffer_len": 0,
            "storage_len": 0,
            "buffer_size": 0,
            "sovereignty_score": 0.0,
            "decision_history_len": 0,
        }
        self._stats_subscription = None
        try:
            # Subscribe to SAGE_STATS broadcasts (dst="all"). The kernel-side
            # snapshot loop (or any later wiring in core/plugin.py) is
            # responsible for calling self.update_cached_stats(payload) when
            # the broadcast arrives. Pattern matches AgencyProxy.update_cached_stats.
            #
            # Option B (2026-04-29): declare types=["SAGE_STATS"] so every
            # other broadcast (~150 msg types) is dropped at publish time
            # instead of filling our queue and being discarded by the
            # consumer-side `if msg.get("type") == SAGE_STATS` filter at
            # plugin.py:2162. Closes the rl_proxy_stats queue-full flood
            # (T2: 308k drops, T3: 342k drops, primary v2 bus regression).
            # Targeted dst="rl_proxy_stats" RPC replies bypass the filter.
            from titan_plugin.bus import SAGE_STATS as _SAGE_STATS
            self._stats_subscription = bus.subscribe(
                "rl_proxy_stats", types=[_SAGE_STATS])
        except Exception:
            # Bus may not support broadcast subscription in unit tests; fall
            # through silently — sovereignty_score returns 0.0 in that case.
            pass

        # Lazy SageEncoder for parent-side action embedding + projection.
        # Replaces parent's SageRecorder for the gatekeeper-routing path
        # (decide_execution_mode is bus-routed; only encoding stays parent-local).
        self._encoder = None

    # ── Lifecycle ──────────────────────────────────────────────────

    def _ensure_started(self) -> None:
        # Async-safe Guardian.start() — see _start_safe.py for rationale.
        from ._start_safe import ensure_started_async_safe
        if ensure_started_async_safe(
            self._guardian, "rl", id(self), proxy_label="RLProxy"
        ):
            self._started = True

    # ── §A.8.7 Gatekeeper-routing (sync, sub-second) ──────────────

    def evaluate(self, state_tensor, prompt: str = "") -> dict:
        """Back-compat: returns dict with mode/advantage. Calls into the
        same handler as decide_execution_mode but unpacks differently for
        legacy callers that expected dict-shape return."""
        self._ensure_started()
        # Allow tensor or list input.
        state_list = self._coerce_state(state_tensor)
        reply = self._bus.request(
            "rl_proxy", "rl",
            {
                "action": "decide_execution_mode",
                "state_tensor": state_list,
                "raw_prompt": prompt,
            },
            timeout=self._GATE_TIMEOUT_S,
            reply_queue=self._reply_queue,
        )
        if reply:
            body = reply.get("payload", {}) or {}
            return {
                "mode": body.get("mode", "Shadow"),
                "advantage": float(body.get("advantage", 0.0)),
                "confidence": float(body.get("advantage", 0.0)),
            }
        # Hard-fail Shadow mode (defer to LLM).
        return {"mode": "Shadow", "advantage": 0.0, "confidence": 0.0}

    def decide_execution_mode(
        self, state_tensor, raw_prompt: str = "",
    ) -> tuple:
        """SageGatekeeper.decide_execution_mode(state_tensor, raw_prompt) drop-in.
        Returns (mode, advantage, decoded_text) tuple. Hard-fail on bus
        timeout returns (\"Shadow\", 0.0, \"\") so the chat path never breaks."""
        self._ensure_started()
        state_list = self._coerce_state(state_tensor)
        reply = self._bus.request(
            "rl_proxy", "rl",
            {
                "action": "decide_execution_mode",
                "state_tensor": state_list,
                "raw_prompt": raw_prompt,
            },
            timeout=self._GATE_TIMEOUT_S,
            reply_queue=self._reply_queue,
        )
        if not reply:
            logger.warning("[RLProxy] decide_execution_mode timeout — Shadow fallback")
            return ("Shadow", 0.0, "")
        body = reply.get("payload", {}) or {}
        if "error" in body:
            logger.warning("[RLProxy] decide_execution_mode worker error: %s", body["error"])
            return ("Shadow", 0.0, "")
        # Cache sovereignty_score from the response (refreshes between SAGE_STATS broadcasts).
        try:
            sov = body.get("sovereignty_score")
            if sov is not None:
                self._stats_cache["sovereignty_score"] = float(sov)
        except Exception:
            pass
        return (
            body.get("mode", "Shadow"),
            float(body.get("advantage", 0.0)),
            body.get("decoded_text", "") or "",
        )

    # ── §A.8.7 Scholar.dream (async-friendly, LLM-time-scale) ─────

    async def dream(self, epochs: int = 50, batch_size: int = 256) -> dict:
        """SageScholar.dream(...) drop-in, async-friendly. Yields to event
        loop while rl_worker runs IQL. Hard-fail returns 0-loss dict so the
        meditation cycle continues regardless of bus availability.

        Bus IPC reply via dedicated bus_ipc_pool — isolated from default
        64-worker pool so Observatory snapshot bursts can't queue this
        long-running (30-90s) request behind heavy work (RCA 2026-04-29)."""
        self._ensure_started()
        try:
            reply = await self._bus.request_async(
                "rl_proxy", "rl",
                {
                    "action": "dream",
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                },
                self._DREAM_TIMEOUT_S,
                self._reply_queue,
            )
        except Exception as e:
            logger.warning("[RLProxy] dream bus.request raised: %s", e)
            return {"loss_actor": 0.0, "loss_qvalue": 0.0, "loss_value": 0.0}
        if not reply:
            logger.warning("[RLProxy] dream timeout — returning zero-loss")
            return {"loss_actor": 0.0, "loss_qvalue": 0.0, "loss_value": 0.0}
        body = reply.get("payload", {}) or {}
        if "error" in body:
            logger.warning("[RLProxy] dream worker error: %s", body["error"])
            return {"loss_actor": 0.0, "loss_qvalue": 0.0, "loss_value": 0.0}
        return {
            "loss_actor": float(body.get("loss_actor", 0.0)),
            "loss_qvalue": float(body.get("loss_qvalue", 0.0)),
            "loss_value": float(body.get("loss_value", 0.0)),
            "buffer_len": int(body.get("buffer_len", 0)),
            "epochs": int(body.get("epochs", epochs)),
        }

    # ── §A.8.7 Cached stats (SAGE_STATS broadcast subscription) ───

    @property
    def sovereignty_score(self) -> float:
        """SageGatekeeper.sovereignty_score drop-in. Reads cached value
        from last SAGE_STATS broadcast — never blocks. Updated every 60s
        by rl_worker, plus on every decide_execution_mode response."""
        return float(self._stats_cache.get("sovereignty_score", 0.0))

    def update_cached_stats(self, payload: dict) -> None:
        """Called by parent's broadcast handler when SAGE_STATS arrives."""
        try:
            if isinstance(payload, dict):
                self._stats_cache.update(payload)
        except Exception:
            pass

    def get_stats(self) -> dict:
        """Cached stats — fast path (no bus round-trip). For force-fresh
        snapshot, use refresh_stats()."""
        return dict(self._stats_cache)

    def refresh_stats(self) -> dict:
        """Force-fetch fresh stats via synchronous bus.request. NOT for
        hot-path use — diagnostics / explicit refresh only."""
        self._ensure_started()
        reply = self._bus.request(
            "rl_proxy", "rl",
            {"action": "stats"},
            timeout=self._STATS_TIMEOUT_S,
            reply_queue=self._reply_queue,
        )
        if reply:
            body = reply.get("payload", {}) or {}
            self._stats_cache.update(body)
        return dict(self._stats_cache)

    # ── Layer-2 record_transition (back-compat dict path) ─────────

    def record_transition(self, observation, action, reward, next_obs=None, done=False) -> int:
        """Record an RL transition. Replaces SageRecorder.record_transition().
        For the canonical bus path, see Layer 2 SAGE_RECORD_TRANSITION
        publish helper in titan_plugin/__init__.py."""
        self._ensure_started()
        reply = self._bus.request(
            "rl_proxy", "rl",
            {
                "action": "record",
                "observation": observation,
                "action_idx": action,
                "reward": reward,
                "next_observation": next_obs or [],
                "done": done,
            },
            timeout=self._RECORD_TIMEOUT_S,
            reply_queue=self._reply_queue,
        )
        if reply:
            return int(reply.get("payload", {}).get("transition_id", -1))
        return -1

    # ── §A.8.7 Encoder accessors (parent-side, lazy SageEncoder) ──

    @property
    def action_embedder(self):
        """SageRecorder.action_embedder drop-in — backed by parent-safe
        SageEncoder (no LazyMemmapStorage). Lazy-loaded on first access."""
        return self._get_encoder().action_embedder

    @property
    def projection_layer(self):
        """SageRecorder.projection_layer drop-in — 3072→128 nn.Linear,
        frozen, parent-safe."""
        return self._get_encoder().projection_layer

    @property
    def buffer(self):
        """SageRecorder.buffer drop-in. Returns None — parent has no buffer
        in subprocess mode. Legacy callers that check `if self.recorder.buffer`
        fall through to their `else` branch (e.g. `transition_id = -1`)."""
        return None

    @property
    def storage(self):
        """SageRecorder.storage drop-in. Empty list — parent has no storage
        in subprocess mode. SageGatekeeper's KNN-decode walks `len(storage)`
        and the loop body — empty list → graceful Shadow fallback (which
        the gatekeeper inside rl_worker handles correctly)."""
        return []

    @property
    def dynamic_embedding_dim(self) -> int:
        """SageRecorder.dynamic_embedding_dim drop-in (config-driven dim)."""
        return self._get_encoder().dynamic_embedding_dim

    def _get_encoder(self):
        """Lazy SageEncoder. Imported on first access — keeps the heavy
        SageEncoder import out of proxy construction in tests / boot."""
        if self._encoder is None:
            from ..core.sage.recorder import SageEncoder
            self._encoder = SageEncoder()
        return self._encoder

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _coerce_state(state) -> list:
        """Accepts torch.Tensor / list / np.ndarray; returns plain Python
        list[float] safe for msgpack bus payload."""
        if isinstance(state, list):
            return state
        try:
            # torch.Tensor — check via duck-typing to avoid hard import
            tolist = getattr(state, "tolist", None)
            if callable(tolist):
                return tolist()
        except Exception:
            pass
        try:
            return list(state)
        except Exception:
            return [0.0] * 128
