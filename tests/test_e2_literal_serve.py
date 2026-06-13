"""§7.E (E.2) — the hot-path literal-serve glue: _e2_literal_lookup + the cached
BackstopResult rendering. Proves a stored durable verified answer is recalled on the
agno side (lock-free) and injected as a recall block (zero sandbox), and that volatile/
miss fall through. The store→reader round-trip is covered in test_prompt_signature_cache.

Run: python -m pytest tests/test_e2_literal_serve.py -v -p no:anchorpy
"""
import hashlib

import duckdb
import numpy as np
import pytest

from titan_hcl.synthesis.prompt_signature import (
    EMBEDDING_DIM, PromptSignatureStore, prompt_template,
)
from titan_hcl.synthesis.tool_backstop import BackstopResult, _e2_literal_lookup

faiss = pytest.importorskip("faiss")


def _tmpl_embed(text: str):
    key = prompt_template(text)
    h = hashlib.sha256(key.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


class _Plugin:
    def __init__(self, prompt_vec, e_prompt_cache=None):
        self._last_prompt_vec = prompt_vec
        self._full_config = {"synthesis": {"tool_backstop": {
            "e_prompt_cache": e_prompt_cache if e_prompt_cache is not None
            else {"enabled": True, "sim_floor": 0.93}}}}


def _cfg(plugin):
    return plugin._full_config["synthesis"]["tool_backstop"]


def _seed_store(tmp_path, prompt, answer, durability="durable", epoch=100.0):
    conn = duckdb.connect(str(tmp_path / "synth.duckdb"))
    s = PromptSignatureStore(
        conn, faiss_path=str(tmp_path / "prompt_signature_vectors.faiss"),
        snapshot_path=str(tmp_path / "prompt_signature_snapshot.json"),
        embedder=_tmpl_embed, writer=None, graph=None)
    s.write_signature(prompt=prompt, literal_answer=answer, solved_by="tx_perm",
                      durability=durability, created_epoch=epoch)
    return s


def test_e2_lookup_hit(tmp_path, monkeypatch):
    _seed_store(tmp_path, "order my 8 climbing routes", "40320")
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    p = _Plugin(_tmpl_embed("order my 8 climbing routes"))
    hit = _e2_literal_lookup(p, "order my 8 climbing routes", _cfg(p))
    assert hit is not None and hit["literal_answer"] == "40320"


def test_e2_lookup_param_differ_miss(tmp_path, monkeypatch):
    _seed_store(tmp_path, "order my 8 climbing routes", "40320")
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    p = _Plugin(_tmpl_embed("order my 10 climbing routes"))
    assert _e2_literal_lookup(p, "order my 10 climbing routes", _cfg(p)) is None


def test_e2_lookup_volatile_miss(tmp_path, monkeypatch):
    _seed_store(tmp_path, "current sol price", "$71", durability="volatile")
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    p = _Plugin(_tmpl_embed("current sol price"))
    assert _e2_literal_lookup(p, "current sol price", _cfg(p)) is None


def test_e2_lookup_no_vec(tmp_path, monkeypatch):
    _seed_store(tmp_path, "order 8 routes", "40320")
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    p = _Plugin(None)
    assert _e2_literal_lookup(p, "order 8 routes", _cfg(p)) is None


def test_e2_lookup_disabled(tmp_path, monkeypatch):
    _seed_store(tmp_path, "order 8 routes", "40320")
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    p = _Plugin(_tmpl_embed("order 8 routes"), e_prompt_cache={"enabled": False})
    assert _e2_literal_lookup(p, "order 8 routes", _cfg(p)) is None


def test_cached_verdict_block_renders_recall():
    bs = BackstopResult(fired=True, executed=True, success=True, verdict="true",
                        result_summary="40320", cached=True, reason="e2_literal")
    block = bs.verdict_block()
    assert "Recalled Verified Result" in block
    assert "40320" in block
    assert "Executed in the sandbox" not in block   # honest: no run happened


def test_non_cached_block_unchanged():
    bs = BackstopResult(fired=True, executed=True, success=True, verdict="true",
                        result_summary="40320", code="math.factorial(8)")
    block = bs.verdict_block()
    assert "Verified Result" in block and "Executed in the sandbox" in block
