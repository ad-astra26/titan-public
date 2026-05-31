"""Unit tests for the live RPC read seam (RFP 5J-3 / D3a).

Covers the two primitives the sovereign resurrection engine imports by name —
`get_signatures_for_address` (full pagination) and `get_memo_for_tx` (robust
SPL-Memo extraction) — plus the shared `_rpc_post` transport contract. No live
network: pagination/extraction are exercised via a stubbed `_rpc_post`, and the
transport itself is exercised once via `httpx.MockTransport`.
"""
import asyncio

import httpx
import pytest

from titan_hcl.utils import solana_client as sc

MEMO_PID = sc.MEMO_PROGRAM_ID


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── _extract_memo_from_tx (pure) ────────────────────────────────────────────

def test_extract_memo_parsed_instruction_string():
    tx = {"transaction": {"message": {"instructions": [
        {"program": "spl-memo", "programId": MEMO_PID, "parsed": "v=3;evt=abc"},
    ]}}}
    assert sc._extract_memo_from_tx(tx) == "v=3;evt=abc"


def test_extract_memo_by_program_id_when_program_name_absent():
    tx = {"transaction": {"message": {"instructions": [
        {"programId": MEMO_PID, "parsed": "v=3;tier=PT"},
    ]}}}
    assert sc._extract_memo_from_tx(tx) == "v=3;tier=PT"


def test_extract_memo_inner_instruction():
    tx = {
        "transaction": {"message": {"instructions": [
            {"program": "vote", "programId": "Vote111"},
        ]}},
        "meta": {"innerInstructions": [
            {"instructions": [
                {"program": "spl-memo", "programId": MEMO_PID, "parsed": "v=3;inner"},
            ]},
        ]},
    }
    assert sc._extract_memo_from_tx(tx) == "v=3;inner"


def test_extract_memo_log_message_fallback():
    tx = {
        "transaction": {"message": {"instructions": [
            {"program": "system", "programId": sc.SYSTEM_PROGRAM_ID},
        ]}},
        "meta": {"logMessages": [
            "Program MemoSq... invoke [1]",
            "Program log: Memo (len 11): v=3;fromlog",
            "Program MemoSq... success",
        ]},
    }
    assert sc._extract_memo_from_tx(tx) == "v=3;fromlog"


def test_extract_memo_none_when_no_memo():
    tx = {"transaction": {"message": {"instructions": [
        {"program": "system", "programId": sc.SYSTEM_PROGRAM_ID},
    ]}}, "meta": {"logMessages": ["Program log: hello"]}}
    assert sc._extract_memo_from_tx(tx) is None


def test_extract_memo_empty_tx():
    assert sc._extract_memo_from_tx(None) is None
    assert sc._extract_memo_from_tx({}) is None


# ── get_signatures_for_address (pagination over stubbed _rpc_post) ───────────

def test_get_signatures_paginates_newest_to_oldest(monkeypatch):
    # Two full pages (1000) + a short final page → walk stops on the short page.
    page1 = [{"signature": f"s{i}"} for i in range(sc._SIG_PAGE_LIMIT)]
    page2 = [{"signature": f"t{i}"} for i in range(sc._SIG_PAGE_LIMIT)]
    page3 = [{"signature": "u0"}, {"signature": "u1"}]
    pages = {None: page1, "s999": page2, "t999": page3}
    calls = []

    async def fake_post(url, method, params):
        assert method == "getSignaturesForAddress"
        before = params[1].get("before")
        calls.append(before)
        return {"result": pages[before]}

    monkeypatch.setattr(sc, "_rpc_post", fake_post)
    out = _run(sc.get_signatures_for_address("PUB", rpc_url="http://x"))
    assert len(out) == sc._SIG_PAGE_LIMIT * 2 + 2
    assert out[0] == "s0" and out[-1] == "u1"          # newest → oldest
    assert calls == [None, "s999", "t999"]              # cursor advanced by last sig


def test_get_signatures_respects_limit(monkeypatch):
    page = [{"signature": f"s{i}"} for i in range(sc._SIG_PAGE_LIMIT)]

    async def fake_post(url, method, params):
        return {"result": page}

    monkeypatch.setattr(sc, "_rpc_post", fake_post)
    out = _run(sc.get_signatures_for_address("PUB", rpc_url="http://x", limit=5))
    assert out == [f"s{i}" for i in range(5)]


def test_get_signatures_empty_wallet(monkeypatch):
    async def fake_post(url, method, params):
        return {"result": []}

    monkeypatch.setattr(sc, "_rpc_post", fake_post)
    assert _run(sc.get_signatures_for_address("PUB", rpc_url="http://x")) == []


def test_get_signatures_transport_error_propagates(monkeypatch):
    async def fake_post(url, method, params):
        raise RuntimeError("RPC getSignaturesForAddress error: rate limited")

    monkeypatch.setattr(sc, "_rpc_post", fake_post)
    with pytest.raises(RuntimeError, match="rate limited"):
        _run(sc.get_signatures_for_address("PUB", rpc_url="http://x"))


# ── get_memo_for_tx (over stubbed _rpc_post) ────────────────────────────────

def test_get_memo_for_tx_returns_memo(monkeypatch):
    async def fake_post(url, method, params):
        assert method == "getTransaction"
        return {"result": {"transaction": {"message": {"instructions": [
            {"program": "spl-memo", "programId": MEMO_PID, "parsed": "v=3;evt=z"},
        ]}}}}

    monkeypatch.setattr(sc, "_rpc_post", fake_post)
    assert _run(sc.get_memo_for_tx("SIG", rpc_url="http://x")) == "v=3;evt=z"


def test_get_memo_for_tx_missing_tx_returns_none(monkeypatch):
    async def fake_post(url, method, params):
        return {"result": None}

    monkeypatch.setattr(sc, "_rpc_post", fake_post)
    assert _run(sc.get_memo_for_tx("SIG", rpc_url="http://x")) is None


# ── _rpc_post transport contract (real httpx via MockTransport) ──────────────
#
# `_rpc_post` does a local `import httpx` then `httpx.AsyncClient(...)`. We patch
# the global `httpx.AsyncClient` to inject a MockTransport so the real request/
# response + error-raising path runs with zero network.

def _patch_mock_transport(monkeypatch, body):
    def handler(request):
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def patched(*a, **k):
        k["transport"] = transport
        return real_async_client(*a, **k)

    monkeypatch.setattr(httpx, "AsyncClient", patched)


def test_rpc_post_parses_body(monkeypatch):
    _patch_mock_transport(monkeypatch, {"jsonrpc": "2.0", "id": 1, "result": ["ok"]})
    body = _run(sc._rpc_post("http://x", "getSignaturesForAddress", ["PUB", {}]))
    assert body["result"] == ["ok"]


def test_rpc_post_raises_on_rpc_error(monkeypatch):
    _patch_mock_transport(
        monkeypatch,
        {"jsonrpc": "2.0", "id": 1, "error": {"code": -32005, "message": "rate"}})
    with pytest.raises(RuntimeError, match="rate"):
        _run(sc._rpc_post("http://x", "getTransaction", ["SIG", {}]))
