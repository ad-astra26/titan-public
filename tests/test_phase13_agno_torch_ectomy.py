"""Phase 13 §3J.3 — agno torch-ectomy regression guard.

The reasoning-tier gatekeeper encode moved host-side (recorder worker). agno must
NOT trigger a local torch SageEncoder. These assert the source-level contract so
torch can't silently creep back into agno's process (the ~800MB→1GB-restart cause).
"""
import re


def _src(path):
    return open(path, encoding="utf-8").read()


def test_prehook_does_not_import_torch():
    src = _src("titan_hcl/modules/agno_hooks.py")
    # The pre-hook must not `import torch` (it did at line ~461 pre-§3J).
    assert "\n        import torch\n" not in src, \
        "agno pre-hook must not import torch (§3J host-side encode)"


def test_prehook_no_local_encode_or_projection():
    src = _src("titan_hcl/modules/agno_hooks.py")
    # No local embed/projection in agno — that's what built the torch SageEncoder.
    assert "action_embedder.encode(" not in src, "agno must not encode locally (§3J)"
    assert "projection_layer(" not in src, "agno must not run the torch projection (§3J)"


def test_prehook_uses_host_side_decide():
    src = _src("titan_hcl/modules/agno_hooks.py")
    assert "decide_execution_mode_from_prompt(" in src, \
        "agno must route the gatekeeper encode host-side via the proxy"


def test_rl_proxy_has_host_side_method():
    src = _src("titan_hcl/proxies/rl_proxy.py")
    assert "def decide_execution_mode_from_prompt(" in src
    assert "encode_host_side" in src


def test_recorder_supports_host_side_encode():
    src = _src("titan_hcl/modules/recorder_worker.py")
    assert "encode_host_side" in src
    assert "observation_vector" in src
