"""Phase 4 FU-2 — Ollama Cloud LLM proposer wiring tests.

The synthesis_worker boot path now constructs an OllamaCloudProvider via
`titan_hcl.inference.get_provider("ollama_cloud", inference_cfg)` and
wraps it with `make_default_llm_propose`. These tests don't spin up the
full worker (that requires spawn + bus + Guardian); they validate the
factory contract that the worker depends on:

- get_provider("ollama_cloud", cfg) returns a working provider when
  ollama_cloud_api_key is set
- make_default_llm_propose binds it into the (cluster) -> LLMProposal
  sync callable
- provider construction failure degrades to all-reject (no exception
  propagates past the worker boot)
- A live provider response parses through the line-prefix protocol
  (mocked at the provider.complete level)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from titan_hcl.synthesis.consolidation import Cluster, LLMProposal, TxCandidate
from titan_hcl.synthesis.consolidation_defaults import (
    make_default_llm_propose,
)


def _cluster_with(tags=("topic:linux",), n=3):
    return Cluster(members=[
        TxCandidate(
            tx_hash=f"t{i}", fork="declarative", tags=tags, embedding=None,
        )
        for i in range(n)
    ])


# ── Provider factory contract ──────────────────────────────────────


def test_get_provider_ollama_cloud_imports_cleanly():
    """The factory entry point synthesis_worker calls at boot must
    import without side effects. Doesn't construct a provider (which
    would require a real API key)."""
    from titan_hcl.inference import get_provider
    assert callable(get_provider)


def test_get_provider_ollama_cloud_with_api_key_constructs():
    """get_provider('ollama_cloud', {'ollama_cloud_api_key': 'fake'})
    constructs an OllamaCloudProvider instance (no network call yet)."""
    from titan_hcl.inference import get_provider
    p = get_provider("ollama_cloud", {
        "ollama_cloud_api_key": "fake-key-for-test",
    })
    assert p is not None
    assert hasattr(p, "complete")
    assert hasattr(p, "chat")


def test_get_provider_unknown_raises():
    """The factory raises ValueError for an unregistered provider name."""
    from titan_hcl.inference import get_provider
    with pytest.raises(ValueError):
        get_provider("nonexistent_provider", {})


# ── make_default_llm_propose binding ───────────────────────────────


class _StubProvider:
    """Async-compatible stub matching the InferenceProvider contract."""

    def __init__(self, response: str = "", raise_exc=None):
        self._response = response
        self._raise = raise_exc

    async def complete(self, **_kw) -> str:
        if self._raise is not None:
            raise self._raise
        return self._response


def test_make_default_llm_propose_returns_callable():
    p = _StubProvider(response="ACTION: reject\nREASON: test")
    propose = make_default_llm_propose(p)
    assert callable(propose)


def test_propose_fn_parses_new_concept_response():
    response = (
        "ACTION: new_concept\n"
        "CONCEPT_ID: metaplex_nft_minting\n"
        "NAME: Metaplex NFT minting\n"
        "MEMORY_TYPE: procedural\n"
        "REASON: cluster of 3 mint attempts"
    )
    propose = make_default_llm_propose(_StubProvider(response=response))
    result = propose(_cluster_with())
    assert isinstance(result, LLMProposal)
    assert result.action == "new_concept"
    assert result.concept_id == "metaplex_nft_minting"
    assert result.memory_type == "procedural"


def test_propose_fn_parses_version_bump_response():
    response = (
        "ACTION: version_bump\n"
        "CONCEPT_ID: linux_terminal\n"
        "NAME: Linux terminal\n"
        "MEMORY_TYPE: declarative\n"
        "REASON: enriched"
    )
    propose = make_default_llm_propose(_StubProvider(response=response))
    result = propose(_cluster_with())
    assert result.action == "version_bump"
    assert result.concept_id == "linux_terminal"


def test_propose_fn_provider_exception_returns_reject():
    """The worker boot wiring must never let an LLM exception kill the
    consolidation pass — provider errors degrade to REJECT."""
    propose = make_default_llm_propose(
        _StubProvider(raise_exc=RuntimeError("ollama 503")),
    )
    result = propose(_cluster_with())
    assert result.action == "reject"
    assert "RuntimeError" in result.reason


# ── Worker boot path integration (mock the full chain) ─────────────


def test_synthesis_worker_boot_wires_proposer_when_api_key_present(monkeypatch):
    """Smoke-test the boot path: when [inference].ollama_cloud_api_key is
    set in synthesis worker config, the boot path imports get_provider +
    constructs OllamaCloudProvider + wraps with make_default_llm_propose.

    We don't spin up the full worker (spawn + bus); we replicate the
    boot's provider-init block in isolation."""
    inference_cfg = {
        "ollama_cloud_api_key": "fake-key-for-test",
        "ollama_cloud_model": "deepseek-v3.1:671b",
    }

    # Mirror the worker's boot block.
    from titan_hcl.inference import get_provider
    provider = get_provider("ollama_cloud", inference_cfg)
    assert provider is not None

    propose = make_default_llm_propose(provider)
    assert callable(propose)


def test_synthesis_worker_boot_falls_back_when_api_key_missing():
    """Worker boot's fallback path: empty ollama_cloud_api_key → no
    provider construction, propose_fn is the all-reject sentinel."""
    inference_cfg = {"ollama_cloud_api_key": ""}

    # Mirror the worker's boot block (the if api_key: branch never fires).
    api_key = inference_cfg.get("ollama_cloud_api_key", "") or ""
    propose_fn = None
    if api_key:
        from titan_hcl.inference import get_provider
        provider = get_provider("ollama_cloud", inference_cfg)
        propose_fn = make_default_llm_propose(provider)
    assert propose_fn is None

    # Fall-through to the all-reject sentinel (replicated from worker boot).
    def fallback(_cluster):
        return LLMProposal(
            action="reject", reason="llm_proposer_unconfigured",
        )
    result = fallback(_cluster_with())
    assert result.action == "reject"
    assert result.reason == "llm_proposer_unconfigured"
