"""RFP_g18_high_rate_state_shm_migration §7.A — G-PARITY instrument tests.

The probe is env-gated instrumentation (default OFF) that, on each trinity
bus event, reads the matching SHM component slot and logs bus-tensor vs
SHM-tensor + slot freshness. These tests pin its three load-bearing
behaviors: it detects parity, it flags an EMPTY/missing slot, and it never
alters the cache write (INV: no behavior change vs the pre-probe path).

Run isolated (TorchRL mmap): pytest tests/test_g18_parity_probe.py -p no:anchorpy
"""
import logging

import titan_hcl.modules.cognitive_worker as cw


class _FakeBank:
    """Minimal ShmReaderBank stand-in returning the component readers'
    real shape: {"values": [...], "age_seconds": float, "seq": int}."""

    def read_inner_mind_15d(self):
        return {"values": [0.1] * 15, "age_seconds": 0.02, "seq": 42}

    def read_inner_spirit_45d(self):
        return {"values": [0.9] + [0.5] * 44, "age_seconds": 0.01, "seq": 7}

    def read_inner_body_5d(self):
        return None  # EMPTY slot — the case the migration must NOT cut over on


def _dispatch(refs, values, *, dim, inner_key, type_label, src="inner"):
    cw._dispatch_trinity_state(
        refs, {"src": src, "values": values}, dim=dim,
        inner_key=inner_key, outer_key="_outer_x", type_label=type_label)


def test_parity_match_logs_zero_diff(monkeypatch, caplog):
    monkeypatch.setattr(cw, "_G18_PARITY_PROBE", True)
    monkeypatch.setattr(cw, "_G18_PARITY_INTERVAL_S", 0.0)
    cw._G18_PARITY_LAST_LOG.clear()
    refs = {"_shm_reader_bank": _FakeBank()}
    with caplog.at_level(logging.INFO, logger=cw.logger.name):
        _dispatch(refs, [0.1] * 15, dim=15,
                  inner_key="_inner_mind_state", type_label="MIND_STATE")
    line = next(r.getMessage() for r in caplog.records if "G18-PARITY" in r.getMessage())
    assert "MIND_STATE" in line and "max_abs_diff=0.000000" in line
    assert "shm_seq=42" in line


def test_parity_mismatch_surfaces_diff(monkeypatch, caplog):
    monkeypatch.setattr(cw, "_G18_PARITY_PROBE", True)
    monkeypatch.setattr(cw, "_G18_PARITY_INTERVAL_S", 0.0)
    cw._G18_PARITY_LAST_LOG.clear()
    refs = {"_shm_reader_bank": _FakeBank()}
    with caplog.at_level(logging.INFO, logger=cw.logger.name):
        _dispatch(refs, [0.1] + [0.5] * 44, dim=45,
                  inner_key="_inner_spirit_state", type_label="SPIRIT_STATE")
    line = next(r.getMessage() for r in caplog.records if "G18-PARITY" in r.getMessage())
    assert "max_abs_diff=0.800000" in line  # bus[0]=0.1 vs shm[0]=0.9


def test_empty_slot_warns(monkeypatch, caplog):
    monkeypatch.setattr(cw, "_G18_PARITY_PROBE", True)
    monkeypatch.setattr(cw, "_G18_PARITY_INTERVAL_S", 0.0)
    cw._G18_PARITY_LAST_LOG.clear()
    refs = {"_shm_reader_bank": _FakeBank()}
    with caplog.at_level(logging.WARNING, logger=cw.logger.name):
        _dispatch(refs, [0.2] * 5, dim=5,
                  inner_key="_inner_body_state", type_label="BODY_STATE")
    line = next(r.getMessage() for r in caplog.records if "G18-PARITY" in r.getMessage())
    assert "SHM slot EMPTY" in line and "BODY_STATE" in line


def test_throttle_suppresses_repeat(monkeypatch, caplog):
    monkeypatch.setattr(cw, "_G18_PARITY_PROBE", True)
    monkeypatch.setattr(cw, "_G18_PARITY_INTERVAL_S", 10.0)
    cw._G18_PARITY_LAST_LOG.clear()
    refs = {"_shm_reader_bank": _FakeBank()}
    with caplog.at_level(logging.INFO, logger=cw.logger.name):
        _dispatch(refs, [0.3] * 15, dim=15,
                  inner_key="_inner_mind_state", type_label="MIND_STATE")
        _dispatch(refs, [0.7] * 15, dim=15,
                  inner_key="_inner_mind_state", type_label="MIND_STATE")
    parity_lines = [r for r in caplog.records if "G18-PARITY" in r.getMessage()]
    assert len(parity_lines) == 1  # second call throttled


def test_no_behavior_change_cache_still_written(monkeypatch):
    """The probe is read-only: the cache slot still holds the bus value,
    identical to the pre-probe path."""
    monkeypatch.setattr(cw, "_G18_PARITY_PROBE", True)
    monkeypatch.setattr(cw, "_G18_PARITY_INTERVAL_S", 0.0)
    cw._G18_PARITY_LAST_LOG.clear()
    refs = {"_shm_reader_bank": _FakeBank()}
    _dispatch(refs, [0.7] * 15, dim=15,
              inner_key="_inner_mind_state", type_label="MIND_STATE")
    assert refs["_inner_mind_state"] == [0.7] * 15


def test_probe_off_is_inert(monkeypatch, caplog):
    monkeypatch.setattr(cw, "_G18_PARITY_PROBE", False)
    cw._G18_PARITY_LAST_LOG.clear()
    refs = {"_shm_reader_bank": _FakeBank()}
    with caplog.at_level(logging.INFO, logger=cw.logger.name):
        _dispatch(refs, [0.1] * 15, dim=15,
                  inner_key="_inner_mind_state", type_label="MIND_STATE")
    assert not any("G18-PARITY" in r.getMessage() for r in caplog.records)
    assert refs["_inner_mind_state"] == [0.1] * 15  # cache still written


def test_gate_env_var_enables(monkeypatch):
    monkeypatch.setenv("TITAN_G18_PARITY_PROBE", "1")
    assert cw._g18_parity_enabled() is True


def test_gate_marker_file_enables(monkeypatch, tmp_path):
    monkeypatch.delenv("TITAN_G18_PARITY_PROBE", raising=False)
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    assert cw._g18_parity_enabled() is False  # no marker yet
    (tmp_path / ".g18_parity_probe").touch()
    assert cw._g18_parity_enabled() is True


def test_gate_off_when_neither(monkeypatch, tmp_path):
    monkeypatch.delenv("TITAN_G18_PARITY_PROBE", raising=False)
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TITAN_KERNEL_DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)  # cwd-relative "data" fallback also absent
    assert cw._g18_parity_enabled() is False


def test_missing_bank_is_safe(monkeypatch):
    monkeypatch.setattr(cw, "_G18_PARITY_PROBE", True)
    monkeypatch.setattr(cw, "_G18_PARITY_INTERVAL_S", 0.0)
    cw._G18_PARITY_LAST_LOG.clear()
    refs = {}  # no _shm_reader_bank
    _dispatch(refs, [0.4] * 5, dim=5,
              inner_key="_inner_body_state", type_label="BODY_STATE")
    assert refs["_inner_body_state"] == [0.4] * 5  # cache write unaffected
