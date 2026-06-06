"""
Lazy import enforcement tests.

These tests verify that heavy ML libraries (torch, transformers, faiss) are
NOT loaded at worker module import time. Each Guardian-managed worker starts
as its own Python process and must boot quickly with minimal RSS before
doing any actual cognitive work.

History:
- Commit 7f01125 (Mar 2026) applied lazy imports to titan_hcl/__init__.py
  via PEP 562 __getattr__. Saved ~860MB × 9 workers = ~2.3GB.
- Commit 09fcdf5 (Apr 5 2026) added cgn_consumer_client.py with module-level
  `import torch` at the top, bypassing the lazy fix. Saved ~200MB lost.
- Commit b351965 (Apr 8 2026) refactored cgn_consumer_client.py to lazy
  torch import. Saved 178MB at boot.
- This test file (Apr 8 2026 audit) locks in the current state so future
  commits that add eager imports are caught immediately.

If any of these tests fail, you probably added `import torch` (or a heavy
ML library) at the module level of a file that's imported during worker
boot. Move the import inside the function/method that uses it, or use
PEP 562 __getattr__ for module-level exports.

See:
- memory/feedback_lazy_imports_titan_hcl.md
- memory/tuning_012_compound_rewards.md (related scaffolding)
- titan-docs/codebase_audit_08042026.md (Cluster 1)
"""
import sys
import pytest


HEAVY_LIBS = [
    "torch",
    "torchvision",
    "transformers",
    "faiss",
    "sentence_transformers",
    "triton",
]

# Each entry: (module_path, entry_fn_name, display_name)
WORKERS = [
    ("titan_hcl.modules.body_worker", "body_worker_main", "body_worker"),
    ("titan_hcl.modules.mind_worker", "mind_worker_main", "mind_worker"),
    # spirit_worker retired (D-SPEC-116) — engines live in cognitive_worker.
    ("titan_hcl.modules.cognitive_worker", "cognitive_worker_main", "cognitive_worker"),
    ("titan_hcl.modules.language_worker", "language_worker_main", "language_worker"),
    ("titan_hcl.modules.knowledge_worker", "knowledge_worker_main", "knowledge_worker"),
    ("titan_hcl.modules.memory_worker", "memory_worker_main", "memory_worker"),
    ("titan_hcl.modules.llm_worker", "llm_worker_main", "llm_worker"),
    ("titan_hcl.modules.media_worker", "media_worker_main", "media_worker"),
    ("titan_hcl.modules.cgn_worker", "cgn_worker_main", "cgn_worker"),
    ("titan_hcl.modules.timechain_worker", "timechain_worker_main", "timechain_worker"),
]


def _clear_torch_and_titan():
    """Remove torch/titan_hcl from sys.modules so we can test fresh imports."""
    for key in list(sys.modules.keys()):
        if key == "torch" or key.startswith("torch.") or key.startswith("titan_hcl"):
            del sys.modules[key]
        for lib in HEAVY_LIBS:
            if key == lib or key.startswith(lib + "."):
                sys.modules.pop(key, None)


@pytest.mark.parametrize("module_path,entry_fn,display_name", WORKERS)
def test_worker_does_not_leak_torch(module_path, entry_fn, display_name):
    """Each worker module must be importable without loading torch."""
    _clear_torch_and_titan()
    assert "torch" not in sys.modules, (
        f"torch was already loaded before {display_name} import "
        "(test isolation problem — file a bug)"
    )
    # Import the worker module
    mod = __import__(module_path, fromlist=[entry_fn])
    assert hasattr(mod, entry_fn), (
        f"{module_path} does not export {entry_fn}"
    )
    # The import MUST NOT have loaded torch
    assert "torch" not in sys.modules, (
        f"{display_name} import loaded torch at module level. "
        "Move torch imports inside the functions/methods that use them, "
        "or use lazy-load pattern from cgn_consumer_client.py. "
        "See memory/feedback_lazy_imports_titan_hcl.md"
    )


@pytest.mark.parametrize("module_path,entry_fn,display_name", WORKERS)
def test_worker_does_not_leak_heavy_libs(module_path, entry_fn, display_name):
    """Each worker must not leak any heavy ML library at import time."""
    _clear_torch_and_titan()
    mod = __import__(module_path, fromlist=[entry_fn])
    leaked = [lib for lib in HEAVY_LIBS if lib in sys.modules]
    assert not leaked, (
        f"{display_name} import leaked heavy libs: {leaked}. "
        "Move these imports inside the functions/methods that use them."
    )


def test_cgn_consumer_client_construction_lazy():
    """CGNConsumerClient construction must not load torch.

    The class __init__ stores config but defers all torch-dependent setup
    to _ensure_initialized() which runs on first ground()/infer_action() call.
    This is the Apr 8 2026 fix from commit b351965.
    """
    _clear_torch_and_titan()
    from titan_hcl.logic.cgn_consumer_client import CGNConsumerClient
    assert "torch" not in sys.modules, (
        "Importing CGNConsumerClient class loaded torch"
    )
    client = CGNConsumerClient("language")
    assert "torch" not in sys.modules, (
        "CGNConsumerClient construction loaded torch. "
        "_ensure_initialized() should defer torch to first ground() call."
    )


def test_titan_hcl_root_import_lazy():
    """Importing titan_hcl top-level must not load torch.

    This verifies the PEP 562 __getattr__ lazy imports from commit 7f01125
    are still working.
    """
    _clear_torch_and_titan()
    import titan_hcl
    assert "torch" not in sys.modules, (
        "Importing titan_hcl leaked torch. "
        "Check titan_hcl/__init__.py __getattr__ lazy imports."
    )
