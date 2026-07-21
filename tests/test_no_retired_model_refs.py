"""Guard: no references to provider-retired LLM models anywhere in titan_hcl/**.py.

A stale retired-model reference deceives a future reader (it looks current) and
can 410 at runtime as a fallback default. See scripts/check_retired_models.py for
the registry. FALSIFIER: adding a retired model id (e.g. "deepseek-v3.1:671b") as
a literal/comment anywhere under titan_hcl/ turns this RED.

Run: python -m pytest tests/test_no_retired_model_refs.py -q -p no:anchorpy
"""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

from check_retired_models import scan, RETIRED_MODELS  # noqa: E402


def test_no_retired_model_references():
    hits = scan(_ROOT / "titan_hcl")
    detail = "\n".join(f"  {p}:{ln} [{m}] {t}" for p, ln, m, t in hits)
    assert not hits, (
        f"{len(hits)} reference(s) to provider-retired models "
        f"({', '.join(RETIRED_MODELS)}) found under titan_hcl/ — purge them "
        f"(retired models deceive future readers + 410 at runtime):\n{detail}"
    )
