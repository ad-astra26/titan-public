"""Phase 10C gate (rFP §3G.3) — ``logic/spirit_helpers.py`` must be a pure,
light import surface: importing it pulls NEITHER ``torch`` NOR any ``cgn``
package. The spirit reflex + trajectory + felt/bridge/bus helpers were
extracted out of the retiring ``modules/spirit_loop.py`` precisely so that
``agno_hooks`` (via ``logic/reflex_intuition``) and ``inner_spirit_sidecar``
can consume them without paying the boot-time heavy-import cost that the
Phase 11 orchestrator/supervisor split is designed to eliminate.

Run in a FRESH subprocess so the assertion reflects only what
``import titan_hcl.logic.spirit_helpers`` itself drags in — the in-process
pytest interpreter has almost certainly already loaded torch via other
fixtures, which would mask a regression.
"""
import subprocess
import sys


_PROBE = r"""
import sys
import titan_hcl.logic.spirit_helpers  # noqa: F401

heavy = sorted(
    m for m in sys.modules
    if m == "torch" or m.startswith("torch.")
    or m == "cognee" or "cgn" in m.lower()
)
# The 9 relocated helpers must all be present on the module.
import titan_hcl.logic.spirit_helpers as sh
expected = {
    "_load_birth_state", "_compute_spirit_reflex_intuition",
    "_compute_trajectory", "_send_msg", "_send_response",
    "_send_heartbeat", "_build_felt_snapshot",
    "_load_bridge_dedup", "_save_bridge_dedup",
}
missing = sorted(n for n in expected if not hasattr(sh, n))

print("HEAVY=" + ",".join(heavy))
print("MISSING=" + ",".join(missing))
"""


def _run_probe():
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"probe import failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    heavy, missing = "", ""
    for line in proc.stdout.splitlines():
        if line.startswith("HEAVY="):
            heavy = line[len("HEAVY="):].strip()
        elif line.startswith("MISSING="):
            missing = line[len("MISSING="):].strip()
    return heavy, missing


def test_spirit_helpers_imports_without_torch_or_cgn():
    heavy, _ = _run_probe()
    assert heavy == "", (
        "importing titan_hcl.logic.spirit_helpers loaded heavy modules "
        f"(must be torch/cgn-free): {heavy}"
    )


def test_spirit_helpers_exposes_all_relocated_functions():
    _, missing = _run_probe()
    assert missing == "", (
        f"spirit_helpers is missing relocated helper(s): {missing}"
    )
