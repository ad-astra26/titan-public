"""
titan_plugin/utils/network_monitor.py — peer entropy, ping variance, bus drop rate.

Produces three independent rich sources for outer_body V6 5DT composites:
- Peer entropy — Shannon entropy of Solana RPC cluster peer count buckets
  (measures topology diversity; stable cluster + even distribution = high
  entropy; monoculture = low)
- Ping variance — rolling variance of RPC request latencies (measures
  network chaos / jitter)
- Bus drop rate — DivineBus drops / published ratio (measures internal
  bus pressure)

Used by outer_trinity._collect_outer_body for:
  dim [1] proprioception: peer entropy (0.5) + helper availability (0.3) +
         bus module diversity (0.2)
  dim [3] entropy: ping variance (0.4) + bus drop rate (0.3) + error rate (0.3)

Sample-on-demand with rolling buffer for variance. Peer entropy uses a
10-min TTL cache (cluster changes rarely). One internal ping per
get_ping_variance() call (never more than once per 30s due to TTL).

Thread-safe. No background thread.
"""
import json
import logging
import math
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

# Cache TTLs
_PEER_ENTROPY_TTL_S = 600.0   # cluster peer list rarely changes; 10min
_PING_SAMPLE_TTL_S = 30.0     # one ping per 30s max

# Rolling buffers
_PING_BUFFER_MAX = 20         # ~10 min of samples at 30s intervals
_CLUSTER_NODE_VERSION_BUCKETS = 8  # bucket peers by validator version for entropy

# RPC timeouts
_RPC_TIMEOUT_S = 3.0

# State
_lock = threading.Lock()
_peer_entropy_cache: dict = {"value": None, "ts": 0.0}
_last_ping_ts: float = 0.0
_ping_buffer: deque = deque(maxlen=_PING_BUFFER_MAX)

# Default fallback RPC (caller SHOULD pass explicit rpc_url; this is safety net)
_FALLBACK_RPC_URL = "https://api.devnet.solana.com"


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v) if v == v else 0.5))


def _post_json(url: str, payload: dict, timeout: float = _RPC_TIMEOUT_S) -> Optional[dict]:
    """Minimal sync JSON-RPC POST. Returns parsed response dict or None on failure.

    Intentionally uses stdlib urllib instead of httpx/aiohttp so this module
    has zero heavy imports — it gets imported from outer_trinity hot path.
    """
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        return json.loads(body)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, OSError):
        return None


def _ping_once(rpc_url: str) -> Optional[float]:
    """Single getSlot ping. Returns latency in seconds or None on failure."""
    t0 = time.monotonic()
    resp = _post_json(rpc_url, {
        "jsonrpc": "2.0", "id": 1, "method": "getSlot", "params": [],
    })
    if resp is None or "result" not in resp:
        return None
    return time.monotonic() - t0


def get_ping_variance(rpc_url: Optional[str] = None) -> float:
    """Rolling normalized variance of recent RPC ping latencies.

    Returns [0, 1] where 0 = very stable network, 1 = highly chaotic.
    Triggers at most one RPC ping per 30s (TTL-gated).
    Returns 0.5 neutral if buffer is empty or all pings failed.
    """
    global _last_ping_ts

    rpc = rpc_url or _FALLBACK_RPC_URL
    now = time.monotonic()

    do_ping = False
    with _lock:
        if now - _last_ping_ts > _PING_SAMPLE_TTL_S:
            _last_ping_ts = now
            do_ping = True

    if do_ping:
        latency = _ping_once(rpc)
        if latency is not None:
            with _lock:
                _ping_buffer.append(latency)

    with _lock:
        samples = list(_ping_buffer)

    if len(samples) < 2:
        return 0.5  # Not enough data

    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / len(samples)
    stddev = math.sqrt(var)

    # Normalize: stddev of 0s = 0, 1s = 1. Typical healthy RPC 20-200ms stddev
    # should produce 0.02-0.2 range. Chaotic RPC with 1s+ jitter = ~1.0.
    return _clamp(stddev / 1.0)


def get_peer_entropy(rpc_url: Optional[str] = None) -> float:
    """Shannon entropy of Solana RPC cluster peer distribution.

    Queries getClusterNodes and buckets peers by version string to measure
    validator-version diversity (proxy for ecosystem health).

    Returns [0, 1] where:
      0.0 = all peers same version (monoculture, fragile)
      1.0 = maximum diversity across N buckets

    Cached for 10min (cluster peer list rarely changes).
    Returns 0.5 neutral on RPC failure.
    """
    rpc = rpc_url or _FALLBACK_RPC_URL

    with _lock:
        cached = _peer_entropy_cache["value"]
        cached_ts = _peer_entropy_cache["ts"]
    if cached is not None and time.time() - cached_ts < _PEER_ENTROPY_TTL_S:
        return cached

    resp = _post_json(rpc, {
        "jsonrpc": "2.0", "id": 1, "method": "getClusterNodes", "params": [],
    }, timeout=5.0)

    if resp is None or "result" not in resp:
        # Don't cache failures — retry next call
        return 0.5

    nodes = resp.get("result") or []
    if not nodes:
        return 0.5

    # Bucket by version string
    version_counts: dict[str, int] = {}
    for node in nodes:
        v = node.get("version") or "unknown"
        version_counts[v] = version_counts.get(v, 0) + 1

    total = sum(version_counts.values())
    if total == 0:
        return 0.5

    # Monoculture is a valid real observation, not missing data: 0.0 entropy.
    if len(version_counts) == 1:
        normalized = 0.0
    else:
        # Shannon entropy
        entropy = 0.0
        for c in version_counts.values():
            p = c / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by max-possible entropy for N buckets
        max_entropy = math.log2(min(len(version_counts), _CLUSTER_NODE_VERSION_BUCKETS))
        normalized = _clamp(entropy / max_entropy) if max_entropy > 0 else 0.0

    with _lock:
        _peer_entropy_cache["value"] = normalized
        _peer_entropy_cache["ts"] = time.time()

    return normalized


def get_bus_drop_rate(bus_stats: Optional[dict]) -> float:
    """DivineBus drop rate: dropped / max(1, published).

    Returns [0, 1] clamped. Healthy bus = 0.0 (no drops); saturated = near 1.0.
    """
    if not isinstance(bus_stats, dict):
        return 0.0

    dropped = int(bus_stats.get("dropped", 0))
    published = int(bus_stats.get("published", 0))

    if published <= 0:
        return 0.0

    return _clamp(dropped / published)


def get_bus_module_diversity(bus_stats: Optional[dict]) -> float:
    """Fraction of expected bus modules currently registered.

    Feeds outer_body[1] proprioception (ecosystem width). Expected 8+ modules
    in a healthy V6 Titan (body/mind/spirit/memory/llm/cgn/knowledge/media/...).
    """
    if not isinstance(bus_stats, dict):
        return 0.5

    modules = bus_stats.get("modules")
    if isinstance(modules, (set, list)):
        count = len(modules)
    else:
        return 0.5

    # 8 distinct modules = fully diverse; normalize
    return _clamp(count / 8.0)


def get_all_stats(rpc_url: Optional[str] = None,
                   bus_stats: Optional[dict] = None) -> dict:
    """Single-call snapshot for outer_trinity.

    `rpc_url` — Solana RPC endpoint (caller passes from NetworkClient)
    `bus_stats` — DivineBus.stats dict
    """
    return {
        "peer_entropy": get_peer_entropy(rpc_url),
        "ping_variance": get_ping_variance(rpc_url),
        "bus_drop_rate": get_bus_drop_rate(bus_stats),
        "bus_module_diversity": get_bus_module_diversity(bus_stats),
    }


def _reset_for_testing() -> None:
    """Clear caches + buffer. Test-only helper."""
    global _last_ping_ts
    with _lock:
        _peer_entropy_cache["value"] = None
        _peer_entropy_cache["ts"] = 0.0
        _ping_buffer.clear()
        _last_ping_ts = 0.0
