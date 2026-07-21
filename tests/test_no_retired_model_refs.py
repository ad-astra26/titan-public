"""Guard: no references to provider-retired LLM models anywhere in titan_hcl/**.py.

A stale retired-model reference deceives a future reader (it looks current) and
can 410 at runtime as a fallback default. See scripts/check_retired_models.py for
the registry. FALSIFIER: adding a retired model id (e.g. "deepseek-v3.1:671b") as
a literal/comment anywhere under titan_hcl/ turns this RED.

Run: python -m pytest tests/test_no_retired_model_refs.py -q -p no:anchorpy
"""
import importlib.util
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Load the guard script by path — do NOT insert scripts/ onto sys.path (that
# shadows same-named modules other conftest imports pull in → ModuleNotFoundError).
_spec = importlib.util.spec_from_file_location(
    "check_retired_models", _ROOT / "scripts" / "check_retired_models.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
scan, RETIRED_MODELS = _mod.scan, _mod.RETIRED_MODELS


def test_no_retired_model_references():
    hits = scan(_ROOT / "titan_hcl")
    detail = "\n".join(f"  {p}:{ln} [{m}] {t}" for p, ln, m, t in hits)
    assert not hits, (
        f"{len(hits)} reference(s) to provider-retired models "
        f"({', '.join(RETIRED_MODELS)}) found under titan_hcl/ — purge them "
        f"(retired models deceive future readers + 410 at runtime):\n{detail}"
    )
