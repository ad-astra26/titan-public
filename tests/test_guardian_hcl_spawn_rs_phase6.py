"""
test_guardian_hcl_spawn_rs_phase6 — verify the kernel-rs spawn.rs change.

Phase 6 / SPEC §11.B.4 / D-SPEC-135 / v1.62.0. Source-level check that
the Rust kernel spawns scripts/guardian_hcl.py (NOT scripts/titan_hcl.py
--server) so guardian_hcl is the L1 child of kernel-rs per INV-PROC-3.
"""
import pathlib


SPAWN_RS = pathlib.Path(__file__).resolve().parent.parent / "titan-rust" / \
    "crates" / "titan-kernel-rs" / "src" / "spawn.rs"


def test_spawn_rs_invokes_guardian_hcl_script():
    """The spawn command argument must be `scripts/guardian_hcl.py`,
    NOT `scripts/titan_hcl.py`."""
    assert SPAWN_RS.exists(), f"missing {SPAWN_RS}"
    src = SPAWN_RS.read_text()
    assert '.arg("scripts/guardian_hcl.py")' in src, (
        "Phase 6 INV-PROC-3: kernel-rs must spawn scripts/guardian_hcl.py")
    # Pre-Phase-6 path is gone — no --server arg, no titan_hcl.py invocation.
    assert '.arg("scripts/titan_hcl.py")' not in src, (
        "Phase 6 PURE CUTOVER: scripts/titan_hcl.py is no longer the "
        "kernel-rs spawn target")
    assert '.arg("--server")' not in src, (
        "Phase 6: guardian_hcl is non-interactive by default — no --server flag")


def test_spawn_rs_renamed_function_and_field():
    src = SPAWN_RS.read_text()
    assert "pub fn spawn_guardian_hcl(" in src
    assert "pub spawn_guardian_hcl: bool" in src
    # Old names purged
    assert "pub fn spawn_python_main(" not in src
    assert "pub spawn_python_main: bool" not in src


def test_spawn_rs_daemon_name_is_guardian_hcl():
    """The build_child_env call passes 'guardian_hcl' as the daemon name
    so TITAN_DAEMON_NAME=guardian_hcl in the child environment."""
    src = SPAWN_RS.read_text()
    assert 'build_child_env("guardian_hcl")' in src
    assert 'build_child_env("titan_HCL")' not in src, (
        "Phase 6: the spawned child is guardian_hcl, not titan_HCL")
