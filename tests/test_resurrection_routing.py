"""G3 routing tests — resurrection.phase_2_3_restore defaults to the SOVEREIGN
v=3 chain (no manifest) and uses the legacy --manifest path ONLY when a manifest
is explicitly supplied (INV-MBR-12). No network — the body engines are stubbed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(_REPO / "scripts"))

import resurrection  # noqa: E402
import backup_restore_sovereign as sov  # noqa: E402


def _fresh_install(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    return tmp_path


def test_phase_2_3_defaults_to_sovereign_chain(tmp_path):
    """No manifest → restore_body_from_chain is called; the manifest path is NOT."""
    install = _fresh_install(tmp_path)
    fake_result = sov.ResurrectionResult(status="resurrected", events_applied=3)

    with patch.object(sov, "restore_body_from_chain",
                      return_value=fake_result) as chain, \
         patch.object(resurrection, "_restore_via_manifest") as legacy:
        out = resurrection.phase_2_3_restore(
            b"\x00" * 64, "PUBKEY", "T1", install_root=str(install),
            manifest_path=None, network="mainnet", verify_zk=False, force=True)

    assert chain.called
    assert not legacy.called
    # chain engine called with commit=True (restores into data/)
    _, kwargs = chain.call_args
    assert kwargs["commit"] is True
    assert kwargs["titan_pubkey"] == "PUBKEY"
    assert out is fake_result


def test_phase_2_3_sovereign_halt_exits(tmp_path):
    """A halted sovereign restore is fatal (sys.exit)."""
    install = _fresh_install(tmp_path)
    halted = sov.ResurrectionResult(status="halted", halt_reason="no_chain",
                                    errors=["no v=3 memos"])
    with patch.object(sov, "restore_body_from_chain", return_value=halted):
        with pytest.raises(SystemExit):
            resurrection.phase_2_3_restore(
                b"\x00" * 64, "PUBKEY", "T1", install_root=str(install),
                manifest_path=None, network="mainnet", verify_zk=False, force=True)


def test_phase_2_3_manifest_path_uses_legacy(tmp_path):
    """--manifest supplied → the legacy manifest path is used; chain is NOT."""
    install = _fresh_install(tmp_path)
    with patch.object(resurrection, "_restore_via_manifest",
                      return_value="legacy-ok") as legacy, \
         patch.object(sov, "restore_body_from_chain") as chain:
        out = resurrection.phase_2_3_restore(
            b"\x00" * 64, "PUBKEY", "T1", install_root=str(install),
            manifest_path="/some/off-site/manifest.json", network="mainnet",
            verify_zk=False, force=True)

    assert legacy.called
    assert not chain.called
    assert out == "legacy-ok"
