"""Phase H tests — research services (services.py) + the [research] pip install
(phases._pip_install_research). Gates G8/G9/G10 (offline). Every shell-out is
mocked; the live SearXNG/health check is Phase E.
"""
from __future__ import annotations

import subprocess

import pytest

from scripts.setup_titan import phases
from scripts.setup_titan import services as svc
from scripts.setup_titan.modes import Mode


# ── run_services_phase: SearXNG + OCR (INV-PROV-8) ───────────────────────────

def _wire(monkeypatch, *, state="absent", apt_ok=True, docker=True, run_ok=True):
    """Fake host: container `state`, apt success, docker presence, docker-run success."""
    calls: list[list[str]] = []

    def fake_run(cmd, *, shell=False):
        calls.append(list(cmd))
        if cmd[:2] == ["sudo", "apt-get"] and not apt_ok:
            raise subprocess.CalledProcessError(1, cmd)
        if cmd and cmd[0] == "docker" and not run_ok:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(svc, "_run", fake_run)
    monkeypatch.setattr(svc, "_container_state", lambda name: state)
    monkeypatch.setattr(svc.shutil, "which",
                        lambda n: "/usr/bin/docker" if (n == "docker" and docker) else None)
    monkeypatch.setattr(svc, "_ensure_searxng_settings", lambda root: root / "searxng")
    return calls


def test_minimal_skips_services(tmp_path):
    out = svc.run_services_phase(tmp_path, minimal=True)
    assert len(out) == 1 and out[0].severity == "warn" and "minimal" in out[0].detail.lower()


def test_absent_installs_ocr_docker_and_runs_searxng(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, state="absent")
    out = svc.run_services_phase(tmp_path)
    assert all(r.severity != "fail" for r in out)
    apt = [c for c in calls if c[:2] == ["sudo", "apt-get"]][0]
    for p in (*svc.OCR_APT_DEPS, "docker.io"):
        assert p in apt
    run = [c for c in calls if c and c[0] == "docker" and "run" in c][0]
    assert svc.SEARXNG_IMAGE in run and "8080:8080" in " ".join(run)
    assert "--restart=unless-stopped" in run


def test_stopped_container_is_started_not_rerun(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, state="stopped")
    svc.run_services_phase(tmp_path)
    docker = [c for c in calls if c and c[0] == "docker"]
    assert ["docker", "start", svc.SEARXNG_CONTAINER] in docker
    assert not any("run" in c for c in docker)


def test_running_container_is_noop(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, state="running")
    out = svc.run_services_phase(tmp_path)
    assert not any(c and c[0] == "docker" and ("run" in c or "start" in c) for c in calls)
    assert any(r.name == "searxng" and r.severity == "ok" and "already running" in r.detail for r in out)


def test_apt_failure_hard_blocks_before_docker(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, state="absent", apt_ok=False)
    out = svc.run_services_phase(tmp_path)
    fail = [r for r in out if r.severity == "fail"]
    assert fail and fail[0].name == "services" and "--resume" in fail[0].remediation
    assert not any(c and c[0] == "docker" for c in calls)


def test_docker_missing_after_apt_blocks(monkeypatch, tmp_path):
    _wire(monkeypatch, state="absent", docker=False)
    out = svc.run_services_phase(tmp_path)
    fail = [r for r in out if r.severity == "fail"]
    assert fail and fail[-1].name == "searxng" and "docker not on PATH" in fail[-1].detail


def test_searxng_settings_json_format_and_stable_secret(tmp_path):
    cfg = svc._ensure_searxng_settings(tmp_path)
    body = (cfg / "settings.yml").read_text()
    assert "- json" in body and "secret_key" in body
    assert "base_url: http://localhost:8080/" in body
    first = body
    svc._ensure_searxng_settings(tmp_path)          # idempotent — secret unchanged
    assert (cfg / "settings.yml").read_text() == first


# ── _pip_install_research: [research] extra, CPU-only, --minimal-gated (G8/G10) ─

def test_research_install_minimal_skips(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(phases.subprocess, "check_call", lambda *a, **k: called.append(a))
    out = phases._pip_install_research(tmp_path / "pip", tmp_path, minimal=True)
    assert out[0].severity == "warn" and not called


def test_research_install_uses_cpu_index_and_extra(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(phases.subprocess, "check_call",
                        lambda cmd, *a, **k: captured.update(cmd=cmd))
    out = phases._pip_install_research(tmp_path / "pip", tmp_path, minimal=False)
    cmd = captured["cmd"]
    assert "--extra-index-url" in cmd and phases.TORCH_CPU_INDEX in cmd   # never CUDA
    assert f"{tmp_path}[research]" in cmd                                  # the extra, editable
    assert out[0].severity == "ok"


def test_research_install_failure_is_fail_result(monkeypatch, tmp_path):
    def boom(cmd, *a, **k):
        raise subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(phases.subprocess, "check_call", boom)
    out = phases._pip_install_research(tmp_path / "pip", tmp_path, minimal=False)
    assert out[0].severity == "fail" and "[research]" in out[0].detail
