"""
titan_plugin/api/state_accessor.py — TitanStateAccessor + sub-accessors.

Microkernel v2 Phase A §A.4 S5 amendment (2026-04-25).

The single object endpoint code talks to. Composes:
  - ShmReaderBank — direct shm reads (zero IPC) for kernel state
  - CachedState — bus-event-populated cache for non-shm-backed values
  - CommandSender — fire-and-forget bus.publish for write-side ops
  - ConfigAccessor — static config read once at boot

Endpoint code rewrite (post-codemod):
  state = request.app.state.titan_state
  bal = state.network.balance
  trinity = state.trinity.read()
  state.commands.reload_api()

Future transport changes (Phase C Rust kernel, gRPC, shared-memory queues)
touch only this file. Endpoint code stays put.

Sub-accessor responsibilities (locked in PLAN v2 Q4):
  shm-direct       trinity, neuromods, epoch, spirit, sphere_clocks,
                   chi, titanvm, identity (after S4)
  bus-cached       network.balance, network.info, guardian.status,
                   agency.stats, reasoning.stats, dreaming.state,
                   cgn.stats, language.stats, meta_teacher.stats,
                   social.stats, soul.state
  static           config (loaded once at boot from full_config dict)
  commands         CommandSender (fire-and-forget bus.publish)
"""
from __future__ import annotations

import logging
from typing import Any

from titan_plugin.api.cached_state import CachedState
from titan_plugin.api.command_sender import CommandSender
from titan_plugin.api.shm_reader_bank import ShmReaderBank

logger = logging.getLogger(__name__)


# ── Base mixin for graceful fallback ──────────────────────────────────


class _SubAccessorBase:
    """Base class for typed sub-accessors. Provides __getattr__ fallback
    so endpoint code calling unrecognized methods doesn't 500 — instead
    we synthesize a cache-backed getter or no-op callable.

    Per Q4 (PLAN v2): the typed accessors cover the common methods, but
    endpoint code referencing rarely-used methods (e.g. .get_nervous_system,
    .get_topology_extra) falls through to this fallback. Returns a
    _CacheGetter that proxies further attribute access into cache.get().
    """

    def __getattr__(self, name: str):
        # Skip Python internals + private attrs (real AttributeError).
        if name.startswith("_") or name.startswith("__"):
            raise AttributeError(name)
        # Build a cache-key-prefixed getter using the class name as namespace.
        cls = type(self).__name__.replace("Accessor", "").lower()
        cache = getattr(self, "_cache", None)
        if cache is None:
            return _empty_callable
        return _CacheGetter(cls, cache)


# ── Sub-accessors ─────────────────────────────────────────────────────


class NetworkAccessor(_SubAccessorBase):
    """Solana network state — bus-cached. Kernel publishes
    SOLANA_BALANCE_UPDATED + NETWORK_INFO_UPDATED periodically."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    @property
    def balance(self) -> float:
        """Current SOL balance — last cached value (typically <30s old)."""
        info = self._cache.get("network.balance", {})
        if isinstance(info, dict):
            return float(info.get("balance", 0.0))
        if isinstance(info, (int, float)):
            return float(info)
        return 0.0

    @property
    def pubkey(self) -> str:
        info = self._cache.get("network.info", {})
        return str(info.get("pubkey", "")) if isinstance(info, dict) else ""

    @property
    def rpc_urls(self) -> list[str]:
        info = self._cache.get("network.info", {})
        urls = info.get("rpc_urls", []) if isinstance(info, dict) else []
        return list(urls) if isinstance(urls, list) else []

    @property
    def premium_rpc(self) -> str | None:
        info = self._cache.get("network.info", {})
        return info.get("premium_rpc") if isinstance(info, dict) else None

    @property
    def rpc_endpoint(self) -> str:
        urls = self.rpc_urls
        return urls[0] if urls else "https://api.mainnet-beta.solana.com"

    @property
    def is_available(self) -> bool:
        return self._cache.has("network.info")

    def get_raw_account_data(self, pda: str) -> dict | None:
        """Return cached raw account data for a vault PDA, if recently fetched.
        Endpoints that need fresh data publish SOLANA_ACCOUNT_REFRESH_REQUEST
        via state.commands.publish(...) and read on next bus tick."""
        return self._cache.get(f"network.account.{pda}")


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
    """Inner Spirit state — shm-direct (45D fast tensor) + bus-cached
    composite stats (full inner-trinity payload, expression composites,
    π-heartbeat, dream distillation, etc.)."""


    def __init__(self, shm: ShmReaderBank, cache: CachedState) -> None:
        self._shm = shm
        self._cache = cache

    def read_45d(self) -> dict | None:
        """SAT-15 + CHIT-15 + ANANDA-15 fast tensor (S3b)."""
        return self._shm.read_inner_spirit_45d()

    def read_inner_trinity(self) -> dict:
        """Composite inner-trinity payload — bus-cached
        (kernel publishes INNER_TRINITY_UPDATED periodically)."""
        return self._cache.get("spirit.inner_trinity", {}) or {}

    def get_trinity(self) -> dict:
        """Legacy method name — callers from old endpoint code expect dict."""
        return self.read_inner_trinity()

    def get_coordinator(self) -> dict:
        """Composite coordinator view — kernel snapshot baseline + per-event
        cache key overlay.

        rFP_observatory_data_loading_v1 Phase 4 fix (2026-04-26):
        ``spirit.coordinator`` from the kernel snapshot ships placeholder
        zero-valued blocks for chi/msl/dreaming/meta_reasoning/etc. so the
        frontend never crashes pre-Phase-4-publishers. Workers publish the
        real values via per-event ``*_UPDATED`` bus messages
        (spirit_loop._publish_coord_subdomains) which BusSubscriber writes
        to dedicated cache keys (chi.state, msl.state, dreaming.state, ...).
        Without overlay, every endpoint reading get_coordinator() saw the
        placeholders forever — Trinity tab 0.50, I-Depth empty, lifetime
        metrics zero, sphere clocks 0.50 in the UI, etc.

        This overlay reads each per-event cache key and replaces the
        corresponding placeholder block in the snapshot. Snapshot wins
        for keys with no per-event publisher (sphere_clocks, resonance,
        unified_spirit, outer_trinity, trinity tensors, observables,
        consciousness, neuromodulators) — those continue to flow via
        kernel snapshot.
        """
        coord = dict(self._cache.get("spirit.coordinator", {}) or {})
        # Per-event overlays — only overwrite when the per-event key is
        # actually populated (the BusSubscriber writes the payload from
        # the *_UPDATED message). Empty/missing → keep placeholder.
        overlay_map = {
            "chi": "chi.state",
            "msl": "msl.state",
            "pi_heartbeat": "pi_heartbeat.state",
            "dreaming": "dreaming.state",
            "meta_reasoning": "meta_reasoning.state",
            "reasoning": "reasoning.state",
            "expression_composites": "expression.composites",
            "neuromodulators": "neuromods.full",
            "language": "language.stats",
            # Batch E: 30D space topology — TopologyPanel consumer
            "topology": "topology.state",
        }
        for coord_key, cache_key in overlay_map.items():
            real = self._cache.get(cache_key, None)
            if isinstance(real, dict) and real:
                coord[coord_key] = real
        return coord

    def get_v4_state(self) -> dict:
        return self._cache.get("spirit.v4_state", {}) or {}

    def get_sphere_clocks(self) -> dict:
        # Prefer S4 shm if available, fall back to bus cache
        shm_data = self._shm.read_sphere_clocks()
        if shm_data is not None:
            return shm_data
        return self._cache.get("spirit.sphere_clocks", {}) or {}

    def get_nervous_system(self) -> dict:
        """Microkernel v2 amendment: read the spirit_worker-published
        neural_nervous_system stats from the api cache."""
        return self._cache.get("spirit.neural_nervous_system", {}) or {}

    def get_expression_composites(self) -> dict:
        """Microkernel v2 amendment: read spirit_worker-published
        expression composites (SPEAK/ART/MUSIC/SOCIAL/KIN/LONGING)."""
        return self._cache.get("spirit.expression_composites", {}) or {}

    def get_resonance(self) -> dict:
        """Microkernel v2 amendment: read spirit_worker-published
        resonance detector stats."""
        return self._cache.get("spirit.resonance", {}) or {}

    def get_unified_spirit(self) -> dict:
        """Microkernel v2 amendment: read spirit_worker-published
        unified spirit stats."""
        return self._cache.get("spirit.unified_spirit", {}) or {}


class BodyAccessor(_SubAccessorBase):
    """Body tensor + sense state — bus-cached (S7 will add shm-direct).
    Until S7 sensor decoupling, body tensor lives in cache populated by
    BODY_TENSOR_UPDATED events."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    def get_body_tensor(self) -> dict:
        return self._cache.get("body.tensor", {}) or {}


class MindAccessor(_SubAccessorBase):
    """Mind tensor — bus-cached (S7 will add shm-direct for 15D mind)."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    def get_mind_tensor(self) -> dict:
        return self._cache.get("mind.tensor", {}) or {}


class IdentityAccessor(_SubAccessorBase):
    """Titan identity — shm-direct after S4 flag flips, bus-cached
    fallback before. titan_id, maker_pubkey, kernel_instance_nonce."""


    def __init__(self, shm: ShmReaderBank, cache: CachedState) -> None:
        self._shm = shm
        self._cache = cache

    @property
    def titan_id(self) -> str:
        data = self._shm.read_identity()
        if data:
            return data.get("titan_id", "")
        # Fallback: bus-cached or shm bank's resolved id
        return self._shm.titan_id

    @property
    def maker_pubkey(self) -> str:
        data = self._shm.read_identity()
        if data and data.get("maker_pubkey"):
            return data["maker_pubkey"]
        soul = self._cache.get("soul.state", {}) or {}
        return str(soul.get("maker_pubkey", ""))

    @property
    def kernel_instance_nonce(self) -> str:
        data = self._shm.read_identity()
        return data.get("kernel_instance_nonce", "") if data else ""


class SoulAccessor(_SubAccessorBase):
    """Soul governance state — bus-cached. Includes maker_pubkey,
    nft_address, current_gen."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    @property
    def maker_pubkey(self) -> str:
        soul = self._cache.get("soul.state", {}) or {}
        return str(soul.get("maker_pubkey", ""))

    @property
    def nft_address(self) -> str:
        soul = self._cache.get("soul.state", {}) or {}
        return str(soul.get("nft_address", ""))

    @property
    def current_gen(self) -> int:
        soul = self._cache.get("soul.state", {}) or {}
        return int(soul.get("current_gen", 0))

    def get_active_directives(self) -> list:
        soul = self._cache.get("soul.state", {}) or {}
        return list(soul.get("active_directives", []))


class CGNAccessor(_SubAccessorBase):
    """CGN state — bus-cached stats + (post-S4) shm-direct weights."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    def get_stats(self) -> dict:
        return self._cache.get("cgn.stats", {}) or {}


class ReasoningAccessor(_SubAccessorBase):
    """Reasoning engine state — bus-cached."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    def get_stats(self) -> dict:
        return self._cache.get("reasoning.stats", {}) or {}


class DreamingAccessor(_SubAccessorBase):
    """Dreaming state — bus-cached."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    def get_state(self) -> dict:
        return self._cache.get("dreaming.state", {}) or {}

    @property
    def is_dreaming(self) -> bool:
        return bool(self.get_state().get("is_dreaming", False))


class GuardianAccessor(_SubAccessorBase):
    """Guardian/module state — bus-cached."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    def get_status(self) -> dict:
        return self._cache.get("guardian.status", {}) or {}

    def get_modules_by_layer(self, layer: str) -> list:
        status = self.get_status()
        return [
            name for name, info in status.items()
            if isinstance(info, dict) and info.get("layer") == layer
        ]


class AgencyAccessor(_SubAccessorBase):
    """Agency state — bus-cached."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    def get_stats(self) -> dict:
        return self._cache.get("agency.stats", {}) or {}


class LanguageAccessor(_SubAccessorBase):
    """Language teacher state — bus-cached."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    def get_stats(self) -> dict:
        return self._cache.get("language.stats", {}) or {}


class MetaTeacherAccessor(_SubAccessorBase):
    """Meta teacher state — bus-cached."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    def get_stats(self) -> dict:
        return self._cache.get("meta_teacher.stats", {}) or {}


class SocialAccessor(_SubAccessorBase):
    """Social/persona state — bus-cached."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    def get_stats(self) -> dict:
        return self._cache.get("social.stats", {}) or {}


class MemoryAccessor(_SubAccessorBase):
    """Memory state — bus-cached. Mirrors plugin._proxies['memory']
    (MemoryProxy) interface for legacy endpoint code."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    def get_memory_status(self) -> dict:
        return self._cache.get("memory.status", {}) or {}

    def get_topology(self) -> dict:
        return self._cache.get("memory.topology", {}) or {}

    def get_top_memories(self, n: int = 10) -> list:
        # Producer (memory_worker) wraps as {"items": [...], "count": N};
        # legacy/empty path returns either [] or {} — handle all shapes.
        raw = self._cache.get("memory.top", {})
        if isinstance(raw, dict):
            items = raw.get("items", []) or []
        elif isinstance(raw, list):
            items = raw
        else:
            items = []
        return list(items[:n]) if isinstance(items, list) else []

    def get_neuromod_state(self) -> dict:
        return self._cache.get("memory.neuromod_state", {}) or {}

    def get_ns_state(self) -> dict:
        return self._cache.get("memory.ns_state", {}) or {}

    def get_reasoning_state(self) -> dict:
        return self._cache.get("memory.reasoning_state", {}) or {}

    def get_persistent_count(self) -> int:
        # Direct cache key first (legacy); fall back to memory.status.persistent_count
        # which the memory_worker periodic publisher populates (M5 fix 2026-04-26).
        direct = self._cache.get("memory.persistent_count", None)
        if isinstance(direct, (int, float)):
            return int(direct)
        status = self._cache.get("memory.status", {}) or {}
        if isinstance(status, dict):
            return int(status.get("persistent_count", 0) or 0)
        return 0

    def fetch_mempool(self) -> list:
        # Producer wraps as {"items": [...], "count": N}; handle all shapes.
        raw = self._cache.get("memory.mempool", {})
        if isinstance(raw, dict):
            return raw.get("items", []) or []
        return raw if isinstance(raw, list) else []

    def fetch_social_metrics(self) -> dict:
        return self._cache.get("memory.social_metrics", {}) or {}

    def get_knowledge_graph(self, limit: int = 200) -> dict:
        # Microkernel v2: cache contains the worker-pre-built graph. The
        # `limit` arg is preserved for endpoint API compatibility but
        # node count is whatever the worker computed (snapshot freshness
        # > per-request slicing).
        return self._cache.get("memory.knowledge_graph", {}) or {}

    @property
    def _cognee_ready(self) -> bool:
        status = self.get_memory_status()
        return bool(status.get("cognee_ready", False))


class RLAccessor(_SubAccessorBase):
    """RL state — bus-cached."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    def get_stats(self) -> dict:
        return self._cache.get("rl.stats", {}) or {}


class LLMAccessor(_SubAccessorBase):
    """LLM state — bus-cached."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    def get_stats(self) -> dict:
        return self._cache.get("llm.stats", {}) or {}


class MediaAccessor(_SubAccessorBase):
    """Media state — bus-cached."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    def get_stats(self) -> dict:
        return self._cache.get("media.stats", {}) or {}


class TimechainAccessor(_SubAccessorBase):
    """TimeChain state — bus-cached."""


    def __init__(self, cache: CachedState) -> None:
        self._cache = cache

    def get_stats(self) -> dict:
        return self._cache.get("timechain.stats", {}) or {}


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
        cache: CachedState,
        commands: CommandSender,
        full_config: dict | None = None,
    ) -> None:
        self.shm = shm
        self.cache = cache
        self.commands = commands

        # Sub-accessors
        self.network = NetworkAccessor(cache)
        self.trinity = TrinityAccessor(shm)
        self.neuromods = NeuromodAccessor(shm)
        self.epoch = EpochAccessor(shm)
        self.spirit = SpiritAccessor(shm, cache)
        self.body = BodyAccessor(cache)
        self.mind = MindAccessor(cache)
        self.identity = IdentityAccessor(shm, cache)
        self.soul = SoulAccessor(cache)
        self.cgn = CGNAccessor(cache)
        self.reasoning = ReasoningAccessor(cache)
        self.dreaming = DreamingAccessor(cache)
        self.guardian = GuardianAccessor(cache)
        self.agency = AgencyAccessor(cache)
        self.language = LanguageAccessor(cache)
        self.meta_teacher = MetaTeacherAccessor(cache)
        self.social = SocialAccessor(cache)
        self.memory = MemoryAccessor(cache)
        self.rl = RLAccessor(cache)
        self.llm = LLMAccessor(cache)
        self.media = MediaAccessor(cache)
        self.timechain = TimechainAccessor(cache)
        self.config = ConfigAccessor(full_config or {})

        # Microkernel v2 D2 amendment (2026-04-26): legacy callsites
        # use `plugin.event_bus.emit(...)` and `plugin.bus.publish(...)` /
        # `plugin.bus.stats`. Provide compatible shims that route to
        # CommandSender / cache so the codemod doesn't have to rewrite
        # the call shape. Survives Phase B/C since the underlying
        # transport is bus.publish either way.
        self.event_bus = _EventBusShim(commands)
        self.bus = _BusShim(commands, cache)

        logger.info(
            "[TitanStateAccessor] initialized for titan_id=%s "
            "(shm registries=%d, cache_keys=%d, commands=%s)",
            shm.titan_id,
            sum(1 for v in shm.availability_report().values() if v),
            len(cache),
            "yes" if commands._send_queue is not None else "stub",
        )

    # -- fallback accessor for less-common sub-accessors ---------------

    def __getattr__(self, name: str):
        """Synthesize a CacheGetterAccessor for any unknown sub-accessor name.

        Endpoint code may reference titan_state.metabolism, titan_state.studio,
        titan_state.gatekeeper, etc. Rather than enumerate every one of the
        ~30 plugin sub-objects upfront, we synthesize a generic
        cache-backed accessor on demand. The kernel's snapshot publisher
        decides which keys to include in the cache; missing ones return
        empty defaults.

        For underscore-prefixed names (e.g. plugin._dream_inbox,
        plugin._current_user_id): map to cache lookup at "plugin.<name>"
        ONLY IF the kernel snapshot publishes that key. If the key is
        absent from cache, raise AttributeError so legacy code paths
        like `hasattr(plugin, "_proxies")` still return False (and
        downstream warmer threads don't start with the wrong assumptions).

        2026-04-26 fix: original D2 always-return-None broke the
        coordinator warmer's legacy detection on /v3/trinity, causing
        BodyAccessor.get_body_tensor() (which returns dict) to be
        cached as a list — endpoint then crashed on
        `(v - 0.5) ** 2` with str-float TypeError.
        """
        if name.startswith("_"):
            try:
                cache = object.__getattribute__(self, "cache")
            except AttributeError:
                raise AttributeError(name)
            cache_key = f"plugin.{name}"
            if cache.has(cache_key):
                return cache.get(cache_key)
            raise AttributeError(name)
        # Build a lazy CacheGetter that proxies common method calls
        # (get_stats, get_status, etc.) into cache reads.
        cache = self.cache
        return _CacheGetter(name, cache)

    # -- introspection ------------------------------------------------

    def availability(self) -> dict[str, Any]:
        """Per-accessor availability — used by /v4/api-status diagnostic."""
        return {
            "shm_registries": self.shm.availability_report(),
            "cache_keys": self.cache.keys(),
            "cache_bootstrap_done": self.cache.bootstrap_done,
            "titan_id": self.shm.titan_id,
        }


class _CacheGetter:
    """Generic cache-backed accessor for sub-accessors not explicitly typed.

    Used for `titan_state.metabolism`, `titan_state.studio`, etc. Forwards
    attribute reads to cache.get("<sub>.<attr>") and method calls to
    cache.get("<sub>.<method>", lambda: <empty>)().

    Returned by TitanStateAccessor.__getattr__ on demand.
    """


    def __init__(self, name: str, cache) -> None:
        self._name = name
        self._cache = cache

    def __getattr__(self, attr: str):
        if attr.startswith("_"):
            raise AttributeError(attr)
        cache_key = f"{self._name}.{attr}"
        cached_val = self._cache.get(cache_key, _MISSING)
        if cached_val is not _MISSING:
            return _CallableValue(cached_val)
        return _CacheGetter(cache_key, self._cache)

    def __call__(self, *args, **kwargs):
        cached = self._cache.get(self._name, _MISSING)
        return cached if cached is not _MISSING else {}

    def __await__(self):
        # Make the getter itself awaitable so legacy `await titan_state.X`
        # patterns (without method invocation) resolve to the cached value.
        # Note: `await titan_state.X()` (with parens) calls __call__ first,
        # which returns a raw dict, and `await raw_dict` errors — drop the
        # await in those callsites instead.
        cached = self._cache.get(self._name, _MISSING)
        return _CallableValue(cached if cached is not _MISSING else {}).__await__()

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
      - `.stats` → cache.get("bus.stats", {}) (kernel snapshot publishes)
      - bool truth (`if plugin.bus:`) → True (always available)

    Anything else falls through __getattr__ as a no-op (returns _empty_callable).
    """

    __slots__ = ("_commands", "_cache")

    def __init__(self, commands: CommandSender, cache: CachedState) -> None:
        self._commands = commands
        self._cache = cache

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
        return self._cache.get("bus.stats", {}) or {}

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
