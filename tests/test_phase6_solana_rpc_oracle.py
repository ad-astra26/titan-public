"""Phase 6 — `solana_rpc` TruthOraclePlug tests (§P6.C).

Covers `titan_hcl/synthesis/oracles/solana_rpc_oracle.py` against
SPEC §25.3 + arch §11.1 + INV-Syn-13 metered cost class:

- can_handle covers the three Solana claim domains and nothing else
- cost_class == "metered" (gated by daily_sol_budget["helius_rpc"])
- solana_tx_confirmed verdict mapping (confirmed/finalized→true;
  err→false; processed/null→unknown/false)
- solana_account_balance_gte verdict mapping (≥ threshold→true; <→false)
- solana_program_invoked verdict mapping (program in accountKeys→true;
  not present→false; signature not found→false)
- defensive: missing payload fields return clean "unknown"
- defensive: no RPC URL configured → "unknown" with evidence_ref="no_rpc_configured"
- defensive: RPC timeout / HTTP non-200 / JSON-RPC error / connection error
  → "unknown" with evidence_ref="rpc_unreachable"
- defensive: domain not supported → "unknown" with evidence_ref="domain_unsupported"
- evidence_ref shape (signature for tx; "<account>@<slot>" for balance;
  "<sig>@<program>" for program-invoked)
- latency_ms + ts populated
- HTTPS calls are mocked — no actual network traffic
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from titan_hcl.synthesis.oracles.solana_rpc_oracle import (
    DEFAULT_PER_CALL_COST_SOL,
    SUPPORTED_DOMAINS,
    SolanaRpcOracle,
)
from titan_hcl.synthesis.plugs import OracleClaim


# ─────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────


def _mock_resp(payload: dict, status: int = 200) -> MagicMock:
    """Construct a requests.Response stand-in."""
    m = MagicMock()
    m.status_code = status
    m.json.return_value = payload
    return m


@pytest.fixture
def oracle():
    return SolanaRpcOracle(rpc_url="https://mainnet.helius-rpc.com/?api-key=test")


# ─────────────────────────────────────────────────────────────────────────
# Protocol surface
# ─────────────────────────────────────────────────────────────────────────


def test_oracle_id_and_cost_class(oracle):
    assert oracle.oracle_id == "helius_rpc"
    assert oracle.cost_class == "metered"


def test_supported_domains_set():
    assert SUPPORTED_DOMAINS == frozenset(
        {
            "solana_tx_confirmed",
            "solana_account_balance_gte",
            "solana_program_invoked",
        }
    )


def test_can_handle_supported_domains(oracle):
    for d in SUPPORTED_DOMAINS:
        assert oracle.can_handle(d) is True


def test_can_handle_rejects_others(oracle):
    for d in ("code_correctness", "web_fact", "x_event_real"):
        assert oracle.can_handle(d) is False


def test_verify_unsupported_domain_returns_unknown(oracle):
    v = oracle.verify(OracleClaim(domain="not_a_real_domain", payload={}))
    assert v.verdict == "unknown"
    assert v.evidence_ref == "domain_unsupported"
    assert v.oracle_id == "helius_rpc"


def test_no_rpc_url_configured_returns_unknown():
    o = SolanaRpcOracle()
    v = o.verify(
        OracleClaim(
            domain="solana_tx_confirmed",
            payload={"signature": "abc"},
        )
    )
    assert v.verdict == "unknown"
    assert v.evidence_ref == "no_rpc_configured"


# ─────────────────────────────────────────────────────────────────────────
# solana_tx_confirmed
# ─────────────────────────────────────────────────────────────────────────


def test_tx_confirmed_finalized(oracle):
    with patch.object(
        requests,
        "post",
        return_value=_mock_resp(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "context": {"slot": 12345},
                    "value": [
                        {
                            "slot": 12345,
                            "confirmations": None,
                            "err": None,
                            "confirmationStatus": "finalized",
                        }
                    ],
                },
            }
        ),
    ):
        v = oracle.verify(
            OracleClaim(
                domain="solana_tx_confirmed",
                payload={"signature": "5xY...sig"},
            )
        )
    assert v.verdict == "true"
    assert v.evidence_ref == "5xY...sig"
    assert v.cost == DEFAULT_PER_CALL_COST_SOL
    assert v.latency_ms >= 0


def test_tx_confirmed_confirmed_status_is_true(oracle):
    with patch.object(
        requests,
        "post",
        return_value=_mock_resp(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "context": {"slot": 12345},
                    "value": [
                        {"slot": 12345, "err": None, "confirmationStatus": "confirmed"}
                    ],
                },
            }
        ),
    ):
        v = oracle.verify(
            OracleClaim(
                domain="solana_tx_confirmed",
                payload={"signature": "sig"},
            )
        )
    assert v.verdict == "true"


def test_tx_confirmed_processed_only_is_unknown(oracle):
    """processed but not yet confirmed/finalized → unknown (caller re-verifies later)."""
    with patch.object(
        requests,
        "post",
        return_value=_mock_resp(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "context": {"slot": 12345},
                    "value": [
                        {"slot": 12345, "err": None, "confirmationStatus": "processed"}
                    ],
                },
            }
        ),
    ):
        v = oracle.verify(
            OracleClaim(
                domain="solana_tx_confirmed",
                payload={"signature": "sig"},
            )
        )
    assert v.verdict == "unknown"
    assert v.evidence_ref == "sig"


def test_tx_confirmed_err_non_null_is_false(oracle):
    """tx landed on-chain but failed → false (verifiably wrong claim)."""
    with patch.object(
        requests,
        "post",
        return_value=_mock_resp(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "context": {"slot": 12345},
                    "value": [
                        {"slot": 12345, "err": "InstructionError", "confirmationStatus": "finalized"}
                    ],
                },
            }
        ),
    ):
        v = oracle.verify(
            OracleClaim(
                domain="solana_tx_confirmed",
                payload={"signature": "sig"},
            )
        )
    assert v.verdict == "false"


def test_tx_confirmed_signature_not_found_is_false(oracle):
    with patch.object(
        requests,
        "post",
        return_value=_mock_resp(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"context": {"slot": 12345}, "value": [None]},
            }
        ),
    ):
        v = oracle.verify(
            OracleClaim(
                domain="solana_tx_confirmed",
                payload={"signature": "unknown_sig"},
            )
        )
    assert v.verdict == "false"
    assert v.evidence_ref == "unknown_sig"


def test_tx_confirmed_missing_signature_arg_is_unknown(oracle):
    with patch.object(requests, "post") as mock_post:
        v = oracle.verify(
            OracleClaim(domain="solana_tx_confirmed", payload={})
        )
    assert v.verdict == "unknown"
    assert v.evidence_ref == "missing_signature"
    mock_post.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────
# solana_account_balance_gte
# ─────────────────────────────────────────────────────────────────────────


def test_balance_gte_above_threshold_is_true(oracle):
    with patch.object(
        requests,
        "post",
        return_value=_mock_resp(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"context": {"slot": 99999}, "value": 12_000_000_000},
            }
        ),
    ):
        v = oracle.verify(
            OracleClaim(
                domain="solana_account_balance_gte",
                payload={"account": "myAcc", "lamports": 10_000_000_000},
            )
        )
    assert v.verdict == "true"
    assert v.evidence_ref == "myAcc@99999"
    assert v.cost == DEFAULT_PER_CALL_COST_SOL


def test_balance_gte_below_threshold_is_false(oracle):
    with patch.object(
        requests,
        "post",
        return_value=_mock_resp(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"context": {"slot": 100}, "value": 999},
            }
        ),
    ):
        v = oracle.verify(
            OracleClaim(
                domain="solana_account_balance_gte",
                payload={"account": "myAcc", "lamports": 1_000},
            )
        )
    assert v.verdict == "false"


def test_balance_gte_missing_account_is_unknown(oracle):
    with patch.object(requests, "post") as mock_post:
        v = oracle.verify(
            OracleClaim(
                domain="solana_account_balance_gte",
                payload={"lamports": 1000},
            )
        )
    assert v.verdict == "unknown"
    assert v.evidence_ref == "missing_account"
    mock_post.assert_not_called()


def test_balance_gte_bad_lamports_arg_is_unknown(oracle):
    with patch.object(requests, "post") as mock_post:
        v = oracle.verify(
            OracleClaim(
                domain="solana_account_balance_gte",
                payload={"account": "x", "lamports": "not-a-number"},
            )
        )
    assert v.verdict == "unknown"
    assert v.evidence_ref == "bad_lamports_arg"
    mock_post.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────
# solana_program_invoked
# ─────────────────────────────────────────────────────────────────────────


def test_program_invoked_present_in_keys_dict_shape(oracle):
    with patch.object(
        requests,
        "post",
        return_value=_mock_resp(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "transaction": {
                        "message": {
                            "accountKeys": [
                                {"pubkey": "11111111111111111111111111111111", "signer": True},
                                {"pubkey": "MY_PROGRAM_ID", "signer": False},
                            ]
                        }
                    }
                },
            }
        ),
    ):
        v = oracle.verify(
            OracleClaim(
                domain="solana_program_invoked",
                payload={"signature": "sig123", "program_id": "MY_PROGRAM_ID"},
            )
        )
    assert v.verdict == "true"
    assert v.evidence_ref == "sig123@MY_PROGRAM_ID"


def test_program_invoked_present_in_keys_string_shape(oracle):
    """Legacy jsonParsed-encoding sometimes returns plain-string keys."""
    with patch.object(
        requests,
        "post",
        return_value=_mock_resp(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "transaction": {
                        "message": {
                            "accountKeys": [
                                "11111111111111111111111111111111",
                                "MY_PROGRAM_ID",
                            ]
                        }
                    }
                },
            }
        ),
    ):
        v = oracle.verify(
            OracleClaim(
                domain="solana_program_invoked",
                payload={"signature": "sig123", "program_id": "MY_PROGRAM_ID"},
            )
        )
    assert v.verdict == "true"


def test_program_invoked_not_present_is_false(oracle):
    with patch.object(
        requests,
        "post",
        return_value=_mock_resp(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "transaction": {
                        "message": {
                            "accountKeys": [
                                {"pubkey": "OTHER_PROGRAM"},
                            ]
                        }
                    }
                },
            }
        ),
    ):
        v = oracle.verify(
            OracleClaim(
                domain="solana_program_invoked",
                payload={"signature": "sig", "program_id": "MY_PROGRAM_ID"},
            )
        )
    assert v.verdict == "false"


def test_program_invoked_signature_not_found_is_false(oracle):
    with patch.object(
        requests,
        "post",
        return_value=_mock_resp(
            {"jsonrpc": "2.0", "id": 1, "result": None}
        ),
    ):
        v = oracle.verify(
            OracleClaim(
                domain="solana_program_invoked",
                payload={"signature": "absent", "program_id": "P"},
            )
        )
    assert v.verdict == "false"
    assert v.evidence_ref == "absent@P"


def test_program_invoked_missing_arg_is_unknown(oracle):
    with patch.object(requests, "post") as mock_post:
        v = oracle.verify(
            OracleClaim(
                domain="solana_program_invoked",
                payload={"signature": "x"},  # missing program_id
            )
        )
    assert v.verdict == "unknown"
    assert v.evidence_ref == "missing_signature_or_program"
    mock_post.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────
# RPC failure modes
# ─────────────────────────────────────────────────────────────────────────


def test_rpc_connection_error_yields_unknown(oracle):
    with patch.object(
        requests, "post", side_effect=requests.ConnectionError("DNS")
    ):
        v = oracle.verify(
            OracleClaim(
                domain="solana_tx_confirmed",
                payload={"signature": "sig"},
            )
        )
    assert v.verdict == "unknown"
    assert v.evidence_ref == "rpc_unreachable"
    assert v.cost == DEFAULT_PER_CALL_COST_SOL


def test_rpc_timeout_yields_unknown(oracle):
    with patch.object(
        requests, "post", side_effect=requests.Timeout("slow")
    ):
        v = oracle.verify(
            OracleClaim(
                domain="solana_account_balance_gte",
                payload={"account": "x", "lamports": 1},
            )
        )
    assert v.verdict == "unknown"
    assert v.evidence_ref == "rpc_unreachable"


def test_rpc_http_500_yields_unknown(oracle):
    with patch.object(
        requests, "post", return_value=_mock_resp({}, status=500)
    ):
        v = oracle.verify(
            OracleClaim(
                domain="solana_tx_confirmed",
                payload={"signature": "sig"},
            )
        )
    assert v.verdict == "unknown"
    assert v.evidence_ref == "rpc_unreachable"


def test_jsonrpc_error_response_yields_unknown(oracle):
    """JSON-RPC application error → treated same as transport failure."""
    with patch.object(
        requests,
        "post",
        return_value=_mock_resp(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32602, "message": "Invalid params"},
            }
        ),
    ):
        v = oracle.verify(
            OracleClaim(
                domain="solana_tx_confirmed",
                payload={"signature": "sig"},
            )
        )
    assert v.verdict == "unknown"
    assert v.evidence_ref == "rpc_unreachable"


def test_fallback_url_used_after_primary_failure():
    """If the premium URL is down, the plug rolls over to the fallback list."""
    o = SolanaRpcOracle(
        rpc_url="https://primary.example.com",
        fallback_urls=["https://fallback.example.com"],
    )

    call_log = []

    def fake_post(url, **kwargs):
        call_log.append(url)
        if url == "https://primary.example.com":
            raise requests.ConnectionError("primary down")
        return _mock_resp(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "context": {"slot": 1},
                    "value": [
                        {"slot": 1, "err": None, "confirmationStatus": "finalized"}
                    ],
                },
            }
        )

    with patch.object(requests, "post", side_effect=fake_post):
        v = o.verify(
            OracleClaim(
                domain="solana_tx_confirmed",
                payload={"signature": "sig"},
            )
        )
    assert call_log == ["https://primary.example.com", "https://fallback.example.com"]
    assert v.verdict == "true"
