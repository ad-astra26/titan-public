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
async def test_dataplane_only_provider_defers_trust_and_funding():
    """ArweaveChainProvider implements the Phase-A data plane; the trust +
    funding verbs raise NotImplementedError until Phases B/C (the ABC default)."""
    cp = ArweaveChainProvider(keypair_path="/nonexistent", network="devnet")
    with pytest.raises(NotImplementedError):
        await cp.commit_memo("memo")
    with pytest.raises(NotImplementedError):
        await cp.read_memo("sig")
    with pytest.raises(NotImplementedError):
        await cp.balance()
    with pytest.raises(NotImplementedError):
        await cp.fund(0.01)


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
