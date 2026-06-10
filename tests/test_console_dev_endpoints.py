"""Tests for the dev-only Console Agent endpoints (RFP_titan_mobile_app §7.0).

Verifies the dev log-sink + APK-serve behaviour AND the prod-safety gate: with
`dev_enabled=False` the routes must not exist / must not expose the APK.
"""
from titan_console import dev_endpoints
from titan_console.agent import dispatch
from titan_console.context import Context


def _ctx(tmp_path, dev=True):
    return Context(install_root=tmp_path, dev_enabled=dev)


# ── log sink ─────────────────────────────────────────────────────────────
def test_ingest_log_raw_text(tmp_path):
    log = tmp_path / "device.log"
    status, payload = dev_endpoints.ingest_log(_ctx(tmp_path), b"hello world", log_path=log)
    assert status == 200 and payload["ok"]
    assert log.read_text() == "hello world\n"


def test_ingest_log_json_lines(tmp_path):
    log = tmp_path / "device.log"
    status, _ = dev_endpoints.ingest_log(
        _ctx(tmp_path), b'{"lines": ["a", "b", "c"]}', log_path=log
    )
    assert status == 200
    assert log.read_text() == "a\nb\nc\n"


def test_ingest_log_appends(tmp_path):
    log = tmp_path / "device.log"
    dev_endpoints.ingest_log(_ctx(tmp_path), b"line1", log_path=log)
    dev_endpoints.ingest_log(_ctx(tmp_path), b"line2", log_path=log)
    assert log.read_text() == "line1\nline2\n"


def test_ingest_log_empty_rejected(tmp_path):
    status, _ = dev_endpoints.ingest_log(_ctx(tmp_path), b"", log_path=tmp_path / "x.log")
    assert status == 400


# ── apk serve ────────────────────────────────────────────────────────────
def test_serve_apk_present(tmp_path):
    apk = dev_endpoints.apk_path(_ctx(tmp_path))
    apk.parent.mkdir(parents=True)
    apk.write_bytes(b"PK\x03\x04APKDATA")
    status, payload = dev_endpoints.serve_apk(_ctx(tmp_path))
    assert status == 200 and payload == b"PK\x03\x04APKDATA"


def test_serve_apk_absent(tmp_path):
    status, _ = dev_endpoints.serve_apk(_ctx(tmp_path))
    assert status == 404


# ── prod-safety gate (the important invariant) ───────────────────────────
def test_dev_log_route_off_in_prod(tmp_path):
    status, _ = dispatch(_ctx(tmp_path, dev=False), "POST", "/console/dev/log", {}, b"hi", {})
    assert status == 404  # route does not exist without --dev


def test_dev_apk_route_gated(tmp_path):
    apk = dev_endpoints.apk_path(_ctx(tmp_path))
    apk.parent.mkdir(parents=True)
    apk.write_bytes(b"APKDATA")
    # dev ON → serves the apk bytes
    s_on, p_on = dispatch(_ctx(tmp_path, dev=True), "GET", "/dev/latest.apk", {}, b"", {})
    assert s_on == 200 and p_on == b"APKDATA"
    # dev OFF → does NOT serve the apk (falls through to static; never exposes it)
    _, p_off = dispatch(_ctx(tmp_path, dev=False), "GET", "/dev/latest.apk", {}, b"", {})
    assert p_off != b"APKDATA"
