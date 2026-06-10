"""Soul-Diary P5 — scaffolding self-inspection tests.

RFP_titan_authored_soul_diary §7.P5 / INV-SD-9: read-only, bounded self-inspection
(journal errors + error→code correlation + structural glance) → grounds the diary
GATHER + is promoted as a `domain="self"` self-Engram (source="self_inspect").
"""
from __future__ import annotations

import os
import tempfile

import titan_hcl.bus as bus
from titan_hcl.core import self_inspect
from titan_hcl.core.soul_diary import SoulDiaryOrchestrator
from titan_hcl.modules import soul_diary_worker as sdw


class _FakeQueue:
    def __init__(self):
        self.msgs = []

    def put(self, msg):
        self.msgs.append(msg)


# ── journal read (runner-injected — no real journalctl needed) ──────

def test_read_journal_lines_splits_and_strips():
    raw = "boot ok\nERROR synthesis WAL\n\n  \nWARN slow tick\n"
    lines = self_inspect.read_journal_lines(
        "T1", runner=lambda argv: raw)
    assert lines == ["boot ok", "ERROR synthesis WAL", "WARN slow tick"]


def test_read_journal_lines_runner_exception_soft_fails():
    def _boom(argv):
        raise RuntimeError("no journal")
    assert self_inspect.read_journal_lines("T1", runner=_boom) == []


def test_read_journal_builds_correct_unit():
    seen = {}
    self_inspect.read_journal_lines(
        "myagent", runner=lambda argv: seen.update(argv=argv) or "")
    assert "titan-myagent.service" in seen["argv"]


# ── error extraction ────────────────────────────────────────────────

def test_extract_error_observations_filters_and_dedups():
    lines = ["normal line", "ERROR boom", "INFO fine", "ERROR boom",
             "WARNING slow", "Traceback (most recent call last)"]
    out = self_inspect.extract_error_observations(lines)
    assert "ERROR boom" in out
    assert "WARNING slow" in out
    assert "Traceback (most recent call last)" in out
    assert "normal line" not in out and "INFO fine" not in out
    assert out.count("ERROR boom") == 1          # deduped


# ── error→code correlation + bounding ───────────────────────────────

def _tmp_repo():
    d = tempfile.mkdtemp()
    pkg = os.path.join(d, "titan_hcl")
    os.makedirs(pkg)
    with open(os.path.join(pkg, "foo.py"), "w", encoding="utf-8") as f:
        f.write("".join(f"line {i}\n" for i in range(1, 21)))
    return d


def test_correlate_to_code_reads_window():
    repo = _tmp_repo()
    errors = ["ERROR something failed at titan_hcl/foo.py:10 boom"]
    corr = self_inspect.correlate_to_code(errors, repo_root=repo)
    assert len(corr) == 1
    assert corr[0]["file"] == "titan_hcl/foo.py" and corr[0]["line"] == 10
    assert "line 10" in corr[0]["snippet"]
    assert "line 7" in corr[0]["snippet"]        # context above


def test_correlate_to_code_rejects_paths_outside_repo():
    repo = _tmp_repo()
    # An absolute .py OUTSIDE the repo must be refused (traversal guard).
    outside = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
    outside.write(b"secret\n")
    outside.close()
    errors = [f"ERROR at {outside.name}:1"]
    assert self_inspect.correlate_to_code(errors, repo_root=repo) == []
    os.unlink(outside.name)


def test_correlate_to_code_skips_missing_file():
    repo = _tmp_repo()
    errors = ["ERROR at titan_hcl/does_not_exist.py:5"]
    assert self_inspect.correlate_to_code(errors, repo_root=repo) == []


# ── self-source read + structure glance (bounded) ───────────────────

def test_read_self_source_bounded_and_guarded():
    repo = _tmp_repo()
    assert "line 1" in self_inspect.read_self_source(
        "titan_hcl/foo.py", repo_root=repo)
    # Traversal attempt → "".
    assert self_inspect.read_self_source(
        "../../../etc/passwd", repo_root=repo) == ""


def test_glance_self_structure_counts_py_files():
    repo = _tmp_repo()
    st = self_inspect.glance_self_structure(repo)
    assert st["package"] == "titan_hcl"
    assert st["py_files"] >= 1


# ── end-to-end gather + summarize ───────────────────────────────────

def test_gather_self_observations_end_to_end():
    repo = _tmp_repo()
    raw = "ERROR synthesis crash at titan_hcl/foo.py:12 boom\nWARN slow tick"
    obs = self_inspect.gather_self_observations(
        "T1", repo_root=repo, runner=lambda argv: raw)
    assert obs["journal_errors"]
    assert obs["correlations"][0]["file"] == "titan_hcl/foo.py"
    assert obs["structure"]["py_files"] >= 1


def test_summarize_observations_text_and_empty():
    obs = {"journal_errors": ["ERROR boom"],
           "correlations": [{"file": "titan_hcl/foo.py", "line": 12, "snippet": "x"}]}
    s = self_inspect.summarize_observations(obs)
    assert "error/warning" in s and "titan_hcl/foo.py:12" in s
    assert self_inspect.summarize_observations(
        {"journal_errors": [], "correlations": []}) == ""


# ── worker enrich (self-inspect → domain=self thought) ──────────────

def test_enrich_self_inspection_publishes_self_inspect_thought():
    q = _FakeQueue()
    sdw._enrich_self_inspection(q, "soul_diary", "2026-06-10",
                                {"summary": "2 warnings passed through my substrate"})
    assert len(q.msgs) == 1
    m = q.msgs[0]
    assert m["type"] == bus.MEMORY_MEMPOOL_ADD and m["dst"] == "memory"
    assert m["payload"]["source"] == "self_inspect"
    assert "domain:self" in m["payload"]["tags"]
    assert "warnings" in m["payload"]["agent_response"]


def test_enrich_self_inspection_noop_on_empty():
    q = _FakeQueue()
    sdw._enrich_self_inspection(q, "soul_diary", "2026-06-10", {"summary": ""})
    sdw._enrich_self_inspection(q, "soul_diary", "2026-06-10", {})
    assert q.msgs == []


# ── orchestrator renders infra into the grounded prompt ─────────────

def test_render_facts_includes_infra_observations():
    bundle = SoulDiaryOrchestrator.build_bundle(
        sovereignty={"s": 0.5}, outcome={"promoted": 3}, felt={},
        engrams_today=[], memory={}, social={}, onchain={},
        infra={"summary": "1 synthesis WAL warning; looked at the checkpoint guard",
               "structure": {"py_files": 900, "subsystems": ["core", "synthesis"]}})
    facts = SoulDiaryOrchestrator.build_compose_prompts(bundle)["user_prompt"]
    assert "my own substrate" in facts
    assert "checkpoint guard" in facts
    assert "900 source files" in facts
