"""Phase 13 §3J.3 — agno torch-ectomy regression guard.

agno must NOT carry torch in its process (the ~800MB→1GB-restart cause). These
assert the source-level contract so torch can't silently creep back into agno.

(The host-side gatekeeper/recorder encode tests were removed when the offline-RL
subsystem — gatekeeper/scholar/recorder/rl_proxy — was retired in
RFP_synthesis_decision_authority P1. The torch-ectomy guards below stand on their
own: agno carries no torch regardless of what does the routing.)
"""


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
