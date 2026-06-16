"""RFP_supervision_lifecycle §7.B — the supervisor enforces rss_limit against
RssAnon (real private memory), NOT VmRSS.

WHY: VmRSS over-counts reclaimable file-backed mmap (FAISS/Kuzu/DuckDB/ONNX
page-cache) by 100-300MB on the DB-heavy + embedder workers → a VmRSS false-OOM
restart cascade (synthesis VmRSS 706 vs 700 cap / real RssAnon 498; agno VmRSS
880 vs 1000 / RssAnon 815), T1 mainnet 2026-06-15/16. `Orchestrator._get_rss_mb`
is the value `supervisor/core.py:343` compares to `rss_limit_mb`.

Run isolated: python -m pytest tests/test_supervisor_rss_anon_enforcement.py -v -p no:anchorpy
"""
import os

from titan_hcl.orchestrator.core import Orchestrator


def _read_status_mb(field: str) -> float:
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def test_get_rss_mb_returns_rss_anon_not_vmrss():
    # The enforced value must equal RssAnon, and RssAnon <= VmRSS (the mmap
    # delta is exactly what we must NOT count toward the limit).
    enforced = Orchestrator._get_rss_mb(os.getpid())
    rss_anon = _read_status_mb("RssAnon")
    vmrss = _read_status_mb("VmRSS")
    assert rss_anon > 0 and vmrss > 0
    # enforced tracks RssAnon (small drift if the process allocates between the
    # two reads; assert it's on the RssAnon side, well below VmRSS+slack).
    assert abs(enforced - rss_anon) < max(5.0, rss_anon * 0.1)
    assert enforced <= vmrss + 1.0          # never the inflated VmRSS
    assert rss_anon <= vmrss                 # RssAnon is the lower (real) figure


def test_get_rss_mb_dead_pid_is_zero():
    # A non-existent pid → 0.0 (no crash) — the running-only guard upstream
    # means this is belt-and-suspenders, but it must never raise.
    assert Orchestrator._get_rss_mb(2_147_483_000) == 0.0


def test_enforcement_uses_real_memory_not_mmap_inflation():
    # Regression intent: a worker whose VmRSS is inflated by reclaimable mmap
    # but whose RssAnon is under its limit must NOT be over-limit. Simulated by
    # asserting the enforced figure equals the anon (real) measurement, so a
    # 706MB-VmRSS / 498MB-RssAnon worker reads 498 (under a 700 cap), not 706.
    enforced = Orchestrator._get_rss_mb(os.getpid())
    rss_anon = _read_status_mb("RssAnon")
    assert abs(enforced - rss_anon) < max(5.0, rss_anon * 0.1)
