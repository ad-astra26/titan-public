"""Phase B / Phase D-offline tests — scripts/setup_titan/provision.py.

Gate G1: provision logic correct — mode-gating, idempotency skip, block-on-
failure, post-install verification, PATH/profile wiring. Every real installer is
mocked (no toolchain is touched). Live install (G3) is Phase E on the T4 box.
"""
from __future__ import annotations

import os

import pytest

from scripts.setup_titan import provision
from scripts.setup_titan import toolchain as tc
from scripts.setup_titan.modes import Mode


# ── mode / action gating (INV-PROV-2) ────────────────────────────────────────

def test_required_tools_local_is_empty():
    assert provision.required_tools(Mode.LOCAL) == []


def test_required_tools_devnet_full_set_in_order():
    assert provision.required_tools(Mode.DEVNET) == ["rust", "solana", "anchor", "node"]


def test_required_tools_mainnet_full_set():
    assert provision.required_tools(Mode.MAINNET) == ["rust", "solana", "anchor", "node"]


def test_required_tools_resurrect_drops_anchor():
    # resurrect = rust + solana + node (Arweave/Irys), NEVER anchor (no deploy).
    assert provision.required_tools(Mode.MAINNET, resurrect=True) == ["rust", "solana", "node"]


def test_required_tools_accepts_str_mode():
    assert provision.required_tools("local") == []


# ── fake host: tool_present_at + installers the test controls ────────────────

class FakeHost:
    """Simulates a box: `present` = tools currently compatible; running an
    installer marks its tool present (unless overridden)."""

    def __init__(self, present=()):
        self.present = set(present)
        self.installed: list[tuple[str, str]] = []

    def status(self, tool, version):
        ok = tool in self.present
        return tc.ToolStatus(tool, version, tc.EXECUTABLE[tool], present=ok,
                             found=version if ok else None, compatible=ok,
                             detail="present" if ok else "absent")

    def installer(self, tool, *, makes_present=True):
        def _do(version):
            self.installed.append((tool, version))
            if makes_present:
                self.present.add(tool)
        return _do

    def installers(self, **overrides):
        m = {t: self.installer(t) for t in ("rust", "solana", "anchor", "node")}
        m.update(overrides)
        return m


@pytest.fixture
def quiet_phase(monkeypatch):
    """Neutralize the real host-mutating side effects for phase tests."""
    monkeypatch.setattr(provision, "ensure_path_for_run", lambda: None)
    monkeypatch.setattr(provision, "write_profile_lines", lambda profile=None: False)


def _wire(monkeypatch, host: FakeHost, **installer_overrides):
    monkeypatch.setattr(provision.toolchain, "tool_present_at", host.status)
    monkeypatch.setattr(provision, "_INSTALLERS", host.installers(**installer_overrides))


# ── run_provision_phase ──────────────────────────────────────────────────────

def test_local_mode_installs_nothing(quiet_phase, tmp_path):
    results = provision.run_provision_phase(tmp_path, Mode.LOCAL, tc.PINS)
    assert len(results) == 1 and results[0].severity == "ok"
    assert "no toolchain required" in results[0].detail


def test_idempotent_skip_when_all_present(monkeypatch, quiet_phase, tmp_path):
    host = FakeHost(present={"rust", "solana", "anchor", "node"})
    _wire(monkeypatch, host)
    results = provision.run_provision_phase(tmp_path, Mode.MAINNET, tc.PINS)
    assert host.installed == []                              # INV-PROV-1: nothing reinstalled
    assert all(r.severity == "ok" for r in results)
    assert any("already present" in r.detail for r in results)


def test_installs_only_missing(monkeypatch, quiet_phase, tmp_path):
    host = FakeHost(present={"rust"})                        # rust already there
    _wire(monkeypatch, host)
    results = provision.run_provision_phase(tmp_path, Mode.DEVNET, tc.PINS)
    assert [t for t, _ in host.installed] == ["solana", "anchor", "node"]
    assert all(r.severity != "fail" for r in results)
    # the right pins were handed to the installers
    assert dict(host.installed) == {"solana": "3.1.10", "anchor": "0.32.1", "node": "22"}


def test_block_on_installer_failure_halts(monkeypatch, quiet_phase, tmp_path):
    host = FakeHost()

    def boom(version):
        raise provision.ProvisionError("solana exploded", "do the manual thing")

    _wire(monkeypatch, host, solana=boom)
    results = provision.run_provision_phase(tmp_path, Mode.MAINNET, tc.PINS)
    installed = [t for t, _ in host.installed]
    assert "rust" in installed                              # ran before the failure
    assert "anchor" not in installed and "node" not in installed   # halted (INV-PROV-5)
    fail = [r for r in results if r.severity == "fail"]
    assert len(fail) == 1 and fail[0].name == "solana"
    assert "solana exploded" in fail[0].detail
    assert "--resume" in fail[0].remediation


def test_post_install_verification_failure(monkeypatch, quiet_phase, tmp_path):
    host = FakeHost()
    # installer "runs" but does NOT make the tool compatible (wrong version, etc.)
    _wire(monkeypatch, host, rust=host.installer("rust", makes_present=False))
    results = provision.run_provision_phase(tmp_path, Mode.DEVNET, tc.PINS)
    fail = [r for r in results if r.severity == "fail"]
    assert fail and fail[0].name == "rust"
    assert "verification failed" in fail[0].detail


def test_version_override_reaches_installer(monkeypatch, quiet_phase, tmp_path):
    host = FakeHost()
    _wire(monkeypatch, host)
    pins = tc.resolve_versions({"solana_version": "3.2.0"})
    provision.run_provision_phase(tmp_path, Mode.DEVNET, pins)
    assert dict(host.installed)["solana"] == "3.2.0"


def test_resurrect_provisions_rust_solana_node_not_anchor(monkeypatch, quiet_phase, tmp_path):
    host = FakeHost()
    _wire(monkeypatch, host)
    provision.run_provision_phase(tmp_path, Mode.MAINNET, tc.PINS, resurrect=True)
    assert [t for t, _ in host.installed] == ["rust", "solana", "node"]


# ── PATH wiring (export for the run + persist for next login, §5) ────────────

def test_ensure_path_for_run_prepends_and_is_idempotent(monkeypatch):
    fake_dirs = [provision.Path("/opt/a/bin"), provision.Path("/opt/b/bin")]
    monkeypatch.setattr(provision, "_toolchain_bin_dirs", lambda: fake_dirs)
    monkeypatch.setenv("PATH", "/usr/bin")
    provision.ensure_path_for_run()
    after = os.environ["PATH"].split(os.pathsep)
    assert after[:2] == ["/opt/a/bin", "/opt/b/bin"]        # order preserved, prepended
    assert "/usr/bin" in after
    provision.ensure_path_for_run()                         # idempotent — no dupes
    assert os.environ["PATH"].split(os.pathsep).count("/opt/a/bin") == 1


def test_write_profile_lines_writes_once(tmp_path):
    profile = tmp_path / ".profile"
    assert provision.write_profile_lines(profile) is True
    body = profile.read_text()
    assert provision._PROFILE_MARKER_BEGIN in body
    assert ".cargo/bin" in body and "solana" in body and ".avm/bin" in body
    # idempotent — a second call writes nothing and the marker appears once
    assert provision.write_profile_lines(profile) is False
    assert profile.read_text().count(provision._PROFILE_MARKER_BEGIN) == 1
