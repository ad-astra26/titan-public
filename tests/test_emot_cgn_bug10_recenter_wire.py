"""
Regression tests for BUG #10: legacy k-means recenter was never invoked
in production since Phase 1.6h cutover (2026-04-20).

BUG #10 discovery (2026-04-24): dreaming.py::end_dreaming() had the wire
`if emot_cgn is not None: clusterer.maybe_recenter(force=False)`, but
neither inner_coordinator.py::end_dreaming() call site (:201, :296)
passed `emot_cgn` — which was legitimately None post-Phase-1.6h cutover
(EMOT-CGN moved to standalone subprocess). Consequence: last_recenter_ts
stayed at 0.0 on all 3 Titans, RNG-seeded centroids never drifted,
WONDER monoculture locked in, LOVE never fired on T1.

Fix: invoke maybe_recenter() periodically in emot_cgn_worker main loop
with hourly tick; internal 7-day gate in maybe_recenter() throttles
actual centroid updates per the existing design spec (Q3 TRANSITIONAL
directive — no tuning of interval itself).
"""

import time
from unittest.mock import MagicMock

import pytest


def test_emotion_clusterer_maybe_recenter_fires_when_last_recenter_ts_zero():
    """Root-cause regression: a just-booted EmotionClusterer with
    last_recenter_ts=0 MUST accept the next maybe_recenter() invocation
    (internal gate `(now - 0) < interval` evaluates False because
    now >> interval for any positive interval). Prior to BUG #10 fix
    this was the condition that would have fired ONCE at first
    dream-cycle end, but dreaming.py never reached this call."""
    from titan_plugin.logic.emotion_cluster import EmotionClusterer
    clusterer = EmotionClusterer(save_dir=f"/tmp/test_emotion_cluster_b10_{id(object())}")

    # Fresh state — never recentered
    assert clusterer._last_recenter_ts == 0.0

    # Populate observation buffer enough for recenter to have work to do
    import numpy as np
    for _ in range(60):  # > 50 threshold inside maybe_recenter
        vec = np.random.rand(150).astype(np.float32)
        clusterer._observation_buffer.append(("WONDER", vec))

    # Force a known recenter interval (avoid touching global default)
    clusterer._recenter_interval_s = 7 * 86400.0

    fired = clusterer.maybe_recenter(force=False)
    assert fired, "maybe_recenter should fire when last_recenter_ts=0 regardless of interval"
    assert clusterer._last_recenter_ts > 0, "last_recenter_ts should be populated post-fire"


def test_emotion_clusterer_first_fire_override_threshold_lowered_to_10():
    """BUG #10 follow-up (2026-04-24): when last_recenter_ts==0.0 (fresh
    post-restart state), the observation buffer threshold drops from 50→10
    so the first recenter fires within ~1h post-boot rather than ~6h.
    After first fire, threshold reverts to 50 for steady-state operation."""
    from titan_plugin.logic.emotion_cluster import EmotionClusterer
    import numpy as np

    clusterer = EmotionClusterer(save_dir=f"/tmp/test_emotion_cluster_b10_ff_{id(object())}")
    clusterer._recenter_interval_s = 86400.0  # 1 day
    assert clusterer._last_recenter_ts == 0.0

    # Populate exactly 10 observations — would fail old 50-threshold,
    # succeeds with 10-threshold first-fire override
    for _ in range(10):
        clusterer._observation_buffer.append(
            ("WONDER", np.random.rand(150).astype(np.float32)))

    fired = clusterer.maybe_recenter(force=False)
    assert fired, "first fire should succeed with only 10 observations (override)"
    assert clusterer._last_recenter_ts > 0


def test_emotion_clusterer_post_first_fire_threshold_restored_to_50():
    """After the first fire, the buffer threshold goes back to 50 for
    normal operation. Verify by forcing interval elapsed + buffer of 20."""
    from titan_plugin.logic.emotion_cluster import EmotionClusterer
    import numpy as np, time

    clusterer = EmotionClusterer(save_dir=f"/tmp/test_emotion_cluster_b10_n_{id(object())}")
    clusterer._recenter_interval_s = 1.0  # 1 second so we can test normal-threshold path

    # Simulate post-first-fire state: last_recenter_ts is non-zero
    clusterer._last_recenter_ts = time.time() - 10.0  # 10s ago (interval passed)

    # 20 observations — enough for first-fire override but NOT enough
    # for normal 50-threshold
    for _ in range(20):
        clusterer._observation_buffer.append(
            ("WONDER", np.random.rand(150).astype(np.float32)))

    fired = clusterer.maybe_recenter(force=False)
    assert not fired, \
        "post-first-fire, 20 obs should fail the 50-threshold even if interval elapsed"


def test_emotion_clusterer_gate_throttles_subsequent_calls():
    """After a successful recenter, subsequent calls within the gate
    window return False until interval elapses."""
    from titan_plugin.logic.emotion_cluster import EmotionClusterer
    clusterer = EmotionClusterer(save_dir=f"/tmp/test_emotion_cluster_b10b_{id(object())}")
    clusterer._recenter_interval_s = 7 * 86400.0

    import numpy as np
    for _ in range(60):
        clusterer._observation_buffer.append(
            ("WONDER", np.random.rand(150).astype(np.float32)))

    first = clusterer.maybe_recenter(force=False)
    assert first

    # Repopulate buffer (maybe_recenter clears it on success)
    for _ in range(60):
        clusterer._observation_buffer.append(
            ("WONDER", np.random.rand(150).astype(np.float32)))

    second = clusterer.maybe_recenter(force=False)
    assert not second, "second call within interval must return False"


def test_worker_config_exposes_k_recenter_check_interval_s():
    """BUG #10 fix: emot_cgn_worker reads emot_k_recenter_check_interval_s
    config key; default = 3600.0 (hourly check; internal 7d gate throttles
    actual recenter). This test verifies the config key name matches what
    documentation + deploy scripts expect."""
    # Static source verification — ensures the config key is preserved
    # across refactors. We grep the source; parsing with importlib would
    # require booting the whole worker.
    from pathlib import Path
    source = Path(
        __file__).parent.parent / "titan_plugin" / "modules" / "emot_cgn_worker.py"
    content = source.read_text()
    assert "emot_k_recenter_check_interval_s" in content, \
        "worker must read config key `emot_k_recenter_check_interval_s`"
    assert "K_RECENTER_CHECK_INTERVAL_S" in content, \
        "worker must define K_RECENTER_CHECK_INTERVAL_S constant"
    assert "_last_k_recenter_check_ts" in content, \
        "worker must track last check timestamp"


def test_worker_main_loop_invokes_maybe_recenter():
    """The BUG #10 fix adds an explicit `emot_cgn._clusterer.maybe_recenter(force=False)`
    invocation in the main loop. Source-level check ensures the pattern
    is preserved (actual subprocess-integration test is covered by
    test_emot_cgn_bus_contract.py smoke tests)."""
    from pathlib import Path
    source = Path(
        __file__).parent.parent / "titan_plugin" / "modules" / "emot_cgn_worker.py"
    content = source.read_text()
    assert "emot_cgn._clusterer.maybe_recenter(force=False)" in content, \
        "worker main loop must invoke maybe_recenter() — BUG #10 fix wire"
    assert "BUG #10" in content, \
        "BUG #10 reference comment must remain as breadcrumb for future sessions"


def test_dreaming_emot_cgn_branch_documented_as_dead():
    """The legacy `if emot_cgn is not None:` branch in dreaming.py is dead
    code post-Phase-1.6h. Ensure the comment explaining why remains — if
    someone removes the comment, the dead code becomes a mystery."""
    from pathlib import Path
    source = Path(
        __file__).parent.parent / "titan_plugin" / "logic" / "dreaming.py"
    content = source.read_text()
    assert "DEAD PATH since Phase 1.6h cutover" in content, \
        "dreaming.py must document the dead emot_cgn branch"
    assert "BUG #10 fix" in content, \
        "dreaming.py must reference BUG #10 fix for discoverability"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-p", "no:anchorpy"])
