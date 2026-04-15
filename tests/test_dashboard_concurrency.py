"""
Concurrency smoke test for dashboard endpoints — rFP_meta_cgn_v3 § 15 item 6.

Verifies that the 9 HIGH-severity async-block sites identified in § 15 audit
remain non-blocking under concurrent load. Each test hits a dashboard endpoint
20 times concurrently and asserts no single request exceeds 2s.

Runs against a LIVE local T1 (http://localhost:7777). Skips gracefully if
T1 is unreachable — integration smoke test, not a unit test.

Rationale (why live, not mocked): § 15 fixed event-loop stalls caused by
sync proxy calls from async endpoints. Mocks would not reproduce the
real asyncio + multiprocessing.Queue boundary where the stall manifests.
A live test captures actual dashboard behaviour end-to-end. Deploy gate:
these tests should PASS before any PR touching dashboard.py async routes.

Budget: ~30s total runtime (9 endpoints × 20 concurrent requests, parallelized).
"""
import asyncio
import os
import time

import pytest

pytest_plugins = ["pytest_asyncio"]
pytestmark = pytest.mark.asyncio

T1_BASE = os.environ.get("TITAN_T1_URL", "http://localhost:7777")
CONCURRENT = 20
LATENCY_THRESHOLD_S = 2.0

# The endpoints that § 15 identified as async-block hotspots. All should now
# wrap proxy calls in asyncio.to_thread or use the cached warmer.
#
# /v4/mood-narrative is INTENTIONALLY excluded from the fast set: it makes an
# outbound Ollama LLM call (~2-5s typical) — the bottleneck is the upstream
# API, not the event loop. It's tested separately with a looser threshold.
ENDPOINTS = [
    "/health",
    "/status",
    "/status/mood",
    "/v3/trinity",
    "/v4/inner-trinity",
    "/v4/reasoning",
    "/v4/meta-reasoning",
    "/v4/bus-health",
]

# Endpoints with inherent external latency; tested with a looser budget so
# they still catch event-loop regressions without false-failing on LLM time.
SLOW_ENDPOINTS = ["/v4/mood-narrative"]
SLOW_ENDPOINT_THRESHOLD_S = 8.0  # LLM call + event-loop overhead budget


async def _titan_reachable() -> bool:
    """Return True if T1 /health returns 200 within 2s."""
    try:
        import httpx  # noqa: WPS433 — deferred import so missing httpx doesn't crash collection
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(T1_BASE + "/health")
            return r.status_code == 200
    except Exception:
        return False


async def _timed_get(client, path):
    t0 = time.time()
    try:
        r = await client.get(T1_BASE + path)
        return (path, time.time() - t0, r.status_code)
    except Exception as e:
        return (path, time.time() - t0, f"error: {e}")


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_endpoint_concurrent_latency(endpoint: str):
    """Endpoint must respond <2s under 20 concurrent requests."""
    if not await _titan_reachable():
        pytest.skip(f"T1 at {T1_BASE} unreachable — skipping live integration test")

    import httpx
    # Reasonable per-request timeout; test fails fast if a request hangs
    async with httpx.AsyncClient(timeout=LATENCY_THRESHOLD_S + 3.0) as client:
        tasks = [_timed_get(client, endpoint) for _ in range(CONCURRENT)]
        results = await asyncio.gather(*tasks, return_exceptions=False)

    max_latency = max(r[1] for r in results)
    failures = [r for r in results if not isinstance(r[2], int) or r[2] >= 500]

    # Per-endpoint assertions with context for debugging
    assert max_latency < LATENCY_THRESHOLD_S, (
        f"{endpoint}: max latency {max_latency:.2f}s exceeds {LATENCY_THRESHOLD_S}s "
        f"under {CONCURRENT}-concurrent load. Indicates async-block regression — "
        f"check that the route wraps proxy/sync calls in asyncio.to_thread. "
        f"Full latencies (sorted): {sorted(r[1] for r in results)}"
    )
    assert not failures, (
        f"{endpoint}: {len(failures)}/{CONCURRENT} requests returned 5xx or error: "
        f"{failures[:3]}"
    )


@pytest.mark.parametrize("endpoint", SLOW_ENDPOINTS)
async def test_slow_endpoint_concurrent_latency(endpoint: str):
    """Endpoints with inherent external latency (LLM calls): tolerate higher
    threshold but still verify event-loop doesn't serialize further.
    """
    if not await _titan_reachable():
        pytest.skip(f"T1 at {T1_BASE} unreachable — skipping live integration test")

    import httpx
    async with httpx.AsyncClient(timeout=SLOW_ENDPOINT_THRESHOLD_S + 2.0) as client:
        tasks = [_timed_get(client, endpoint) for _ in range(CONCURRENT)]
        results = await asyncio.gather(*tasks, return_exceptions=False)

    max_latency = max(r[1] for r in results)
    assert max_latency < SLOW_ENDPOINT_THRESHOLD_S, (
        f"{endpoint}: max latency {max_latency:.2f}s exceeds slow-endpoint "
        f"threshold {SLOW_ENDPOINT_THRESHOLD_S}s. LLM call is expected to "
        f"be slow, but not THIS slow — indicates event-loop serialization "
        f"of the outbound HTTP call (should be non-blocking via httpx). "
        f"Full latencies: {sorted(r[1] for r in results)}"
    )


async def test_all_endpoints_parallel_no_stall():
    """All fast endpoints hit 20× concurrently at the SAME TIME — simulates
    frontend opening all observatory tabs at once. Tests the event loop
    doesn't serialize across endpoints.
    """
    if not await _titan_reachable():
        pytest.skip(f"T1 at {T1_BASE} unreachable — skipping live integration test")

    import httpx
    async with httpx.AsyncClient(timeout=LATENCY_THRESHOLD_S + 3.0) as client:
        # Interleave requests across all fast endpoints: [e1 e2 e3 e1 e2 e3 ...]
        tasks = []
        for _ in range(CONCURRENT):
            for ep in ENDPOINTS:
                tasks.append(_timed_get(client, ep))
        t0 = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=False)
        total_wall = time.time() - t0

    # With 8 endpoints × 20 concurrent = 160 requests, if event loop is
    # non-blocking the total should finish in ~max_latency seconds (not
    # 160×latency). Budget: 10s total wall-clock.
    assert total_wall < 10.0, (
        f"All-endpoints parallel test took {total_wall:.1f}s total — "
        f"event loop appears to be serializing requests. "
        f"Expected ~2-4s max if async routes are properly non-blocking."
    )

    # No individual request should exceed threshold
    slowest = sorted(results, key=lambda r: -r[1])[:5]
    worst_latency = slowest[0][1]
    assert worst_latency < LATENCY_THRESHOLD_S, (
        f"Slowest request {slowest[0][0]} took {worst_latency:.2f}s. "
        f"Top 5 slowest: {[(r[0], round(r[1], 2)) for r in slowest]}"
    )
