"""RFP_chain_provider Phase A — ChainProvider data-plane contract tests.

Fake-backed (no network/daemon), per the Maker directive: build + green against
the Fake before any live wiring. The real ArweaveChainProvider's devnet path is
also exercised here (no network); its mainnet path is a live gate (§8 G2/G4).
"""
import os
import tracemalloc

import pytest

from titan_hcl.chain import ChainProvider, ArweaveChainProvider, FakeChainProvider


# ── data-plane round-trip (the core contract) ──────────────────────────────

@pytest.mark.asyncio
async def test_fake_put_get_bytes_roundtrip():
    cp = FakeChainProvider()
    tx = await cp.put(b"sovereign-state-bytes")
    assert tx and isinstance(tx, str)
    assert await cp.get_bytes(tx) == b"sovereign-state-bytes"
    # deterministic: same bytes → same tx
    assert await cp.put(b"sovereign-state-bytes") == tx


@pytest.mark.asyncio
async def test_fake_put_path_get_to_file_byte_identical(tmp_path):
    cp = FakeChainProvider()
    src = tmp_path / "event.tar.zst"
    payload = os.urandom(3 * 1024 * 1024)  # 3 MB
    src.write_bytes(payload)
    tx = await cp.put(str(src))                      # put a PATH (the build path)
    dest = tmp_path / "restored.tar.zst"
    assert await cp.get_to_file(tx, str(dest)) is True
    assert dest.read_bytes() == payload              # byte-identical round-trip


@pytest.mark.asyncio
async def test_get_to_file_is_streamed_not_whole_in_ram(tmp_path):
    """get_to_file must write in chunks (constant RAM, INV-CP-2) — peak Python
    alloc during the transfer must be a small fraction of the object size."""
    cp = FakeChainProvider()
    big = b"TITAN\x00\x00\x00" * (6 * 1024 * 1024)   # 48 MB, in the store
    tx = await cp.put(big)
    dest = tmp_path / "out.bin"
    tracemalloc.start()
    ok = await cp.get_to_file(tx, str(dest))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert ok and dest.read_bytes() == big
    # streamed → peak « 48 MB (the source blob is pre-existing, not traced)
    assert peak < 8 * 1024 * 1024, f"get_to_file peaked {peak//1024//1024}MB — not streamed"


# ── head classification (never cries wolf) ─────────────────────────────────

@pytest.mark.asyncio
async def test_head_present_missing_unverified():
    cp = FakeChainProvider()
    tx = await cp.put(b"x")
    assert await cp.head(tx) == "present"
    assert await cp.head("fake_does_not_exist") == "missing"
    cp.unverified.add(tx)
    assert await cp.head(tx) == "unverified"          # transient ≠ missing


@pytest.mark.asyncio
async def test_get_to_file_missing_returns_false(tmp_path):
    cp = FakeChainProvider()
    assert await cp.get_to_file("fake_nope", str(tmp_path / "x")) is False
    assert await cp.get_bytes("fake_nope") is None


# ── ABC contract: trust/funding not yet implemented (Phase B/C) ────────────

@pytest.mark.asyncio
async def test_real_provider_commit_needs_signer():
    """The real provider's commit_memo (Phase B) REQUIRES a network_client
    (signer) — without one it raises a clear RuntimeError, never a silent no-op."""
    cp = ArweaveChainProvider(keypair_path="/nonexistent", network="devnet")  # no network_client
    with pytest.raises(RuntimeError):
        await cp.commit_memo("memo")
    with pytest.raises(RuntimeError):                       # head-bundle path also needs the signer
        await cp.commit_memo("memo", state_root="deadbeef" * 8)


@pytest.mark.asyncio
async def test_real_provider_funding_devnet_is_noop():
    """Phase C: on devnet there is no real Irys deposit — balance() is +inf
    (treated as unlimited by runway logic) and fund() is a no-op (no spend)."""
    cp = ArweaveChainProvider(keypair_path="/nonexistent", network="devnet")
    assert await cp.balance() == float("inf")
    assert await cp.fund(0.01) is None
    assert await cp.fund(0.01, daily_cap_sol=0.05) is None


# ── Phase B — trust plane (Fake-backed) ─────────────────────────────────────

@pytest.mark.asyncio
async def test_commit_read_memo_roundtrip():
    """commit_memo → a sig; read_memo(sig) returns the memo text. The per-event
    contract: a HEAD commit (state_root set) and a TAIL commit (no state_root)
    both produce readable memos."""
    cp = FakeChainProvider()
    head_sig = await cp.commit_memo("v=3;e1;PT;url=tx_p", state_root="ab" * 32,
                                    sovereignty_bp=7000)
    tail_sig = await cp.commit_memo("v=3;e1;TC;url=tx_t")
    assert head_sig and tail_sig and head_sig != tail_sig
    assert await cp.read_memo(head_sig) == "v=3;e1;PT;url=tx_p"
    assert await cp.read_memo(tail_sig) == "v=3;e1;TC;url=tx_t"
    assert await cp.read_memo("fakesig_unknown") is None


@pytest.mark.asyncio
async def test_list_memos_newest_first():
    """list_memos returns sigs newest→oldest (the resurrection-walk order), capped
    at limit — mirroring getSignaturesForAddress."""
    cp = FakeChainProvider()
    sigs = [await cp.commit_memo(f"v=3;e{i}") for i in range(5)]
    listed = await cp.list_memos("Titan_pubkey", limit=3)
    assert listed == list(reversed(sigs))[:3]    # newest 3, newest-first


@pytest.mark.asyncio
async def test_commit_memo_state_root_changes_sig():
    """The head-bundle (state_root) is part of the committed tx — the Fake's sig
    reflects it, so a head commit is distinguishable from the same memo as a tail."""
    cp = FakeChainProvider()
    s_head = await cp.commit_memo("same-memo", state_root="cd" * 32)
    s_tail = await cp.commit_memo("same-memo")
    assert s_head != s_tail


# ── Phase C — funding plane ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fake_balance_fund_roundtrip():
    cp = FakeChainProvider()
    assert await cp.balance() == 1.0
    sig = await cp.fund(0.05)
    assert sig and sig.startswith("fakefund_")
    assert await cp.balance() == pytest.approx(1.05)
    assert cp.fund_log == [0.05]


def test_provider_fund_accumulator_atomic(tmp_path, monkeypatch):
    """Phase C INV-CP-5: the daily-fund accumulator records atomically (date +
    total_sol + tx_count) and appends an audit row — the bounded state the daily
    cap reads to trim `fund()`. (The daemon spend itself is a live gate.)"""
    monkeypatch.chdir(tmp_path)
    cp = ArweaveChainProvider(keypair_path="/nonexistent", network="mainnet")
    cp._record_fund(0.03, "tx1")
    _t, total, n = cp._fund_today_total()
    assert total == pytest.approx(0.03) and n == 1
    cp._record_fund(0.01, "tx2")
    _t2, total2, n2 = cp._fund_today_total()
    assert total2 == pytest.approx(0.04) and n2 == 2
    # the remaining-cap math fund() uses to trim a top-up
    assert max(0.0, 0.05 - total2) == pytest.approx(0.01)
    assert os.path.exists(os.path.join("data", "backups", "auto_fund_audit.jsonl"))


def test_chain_provider_is_abstract():
    with pytest.raises(TypeError):
        ChainProvider()  # cannot instantiate the ABC directly


# ── real provider, devnet path (no network) ────────────────────────────────

@pytest.mark.asyncio
async def test_rebirthbackup_fetch_routes_through_chain_provider(tmp_path):
    """Phase A tail: RebirthBackup's restore-fetch (`_build_fetch_to_file`) routes
    through the injected ChainProvider's `get_to_file` (not ArweaveStore). Inject
    a FakeChainProvider, point a manifest event at a tx in it, and confirm the
    built fetcher streams that tarball back byte-identical."""
    from titan_hcl.logic.backup import RebirthBackup

    fake = FakeChainProvider()
    backup = RebirthBackup(network_client=None, config={}, titan_id="T1",
                           chain_provider=fake, full_config={})
    assert backup._ensure_chain() is fake          # injection wins over lazy build

    tarball = b"component-tarball-bytes" * 4096     # ~90 KB
    tx = await fake.put(tarball)

    class _Manifest:
        events = [{"personality": {"tx_id": tx, "iv": None,
                                   "merkle_root": "deadbeef"}}]

    fetch = backup._build_fetch_to_file(_Manifest())
    dest = tmp_path / "fetched.tar"
    ok = await fetch(tx, str(dest))                 # Mode-A → chain.get_to_file
    assert ok and dest.read_bytes() == tarball
    assert tx in fake._store                        # came from the provider, not a store


@pytest.mark.asyncio
async def test_arweave_provider_devnet_roundtrip(tmp_path, monkeypatch):
    """The REAL ArweaveChainProvider on devnet uses the local cache — put → a
    `devnet_*` pseudo-tx → get_to_file/head read it back, no network (INV-CP-6)."""
    monkeypatch.chdir(tmp_path)                       # data/arweave_devnet/ under tmp
    cp = ArweaveChainProvider(keypair_path="/nonexistent", network="devnet")
    payload = os.urandom(1 * 1024 * 1024)
    tx = await cp.put(payload)
    assert tx.startswith("devnet_")
    assert await cp.head(tx) == "present"
    dest = tmp_path / "rt.bin"
    assert await cp.get_to_file(tx, str(dest)) is True
    assert dest.read_bytes() == payload
    assert await cp.head("devnet_absent") == "missing"
