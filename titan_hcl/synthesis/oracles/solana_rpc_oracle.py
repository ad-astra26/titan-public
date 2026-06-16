"""Phase 6 — `solana_rpc` TruthOraclePlug (§P6.C; SPEC §25.3 + §25.5).

Wraps a Solana JSON-RPC endpoint (Helius today; sovereign light node =
``[TARGET]`` upgrade per SPEC §25.3) as a `TruthOraclePlug` for
on-chain truth claims.

Per SPEC §25.3 day-one set + arch §11.1: metered oracle (cost class
``"metered"``; oracle_id ``helius_rpc``). The INV-Syn-13 gate enforces
``daily_sol_budget["helius_rpc"]`` from
``titan_params.toml [synthesis.oracle.daily_sol_budget]``.

Claim domains served (SPEC §25.3 + arch §11.1):

- **solana_tx_confirmed** — "did this signature finalize?"
  Payload: ``{"signature": "<base58 sig>"}``.
  Verdict: ``"true"`` if ``getSignatureStatuses`` reports
  ``confirmed`` / ``finalized``; ``"false"`` if explicitly failed
  (``err`` non-null); ``"unknown"`` if RPC timeout / rate-limit /
  network unreachable.

- **solana_account_balance_gte** — "is account X holding ≥ N lamports?"
  Payload: ``{"account": "<base58 pubkey>", "lamports": <int>}``.
  Verdict: ``"true"`` if ``getBalance`` ≥ threshold; ``"false"`` if
  strictly below; ``"unknown"`` on RPC failure.

- **solana_program_invoked** — "does signature S invoke program P?"
  Payload: ``{"signature": "<sig>", "program_id": "<pubkey>"}``.
  Verdict: ``"true"`` if ``getTransaction`` surfaces the program in
  ``message.accountKeys``; ``"false"`` if the signature exists but
  the program does not appear; ``"unknown"`` on RPC failure.

The plug uses synchronous JSON-RPC over HTTPS (via ``requests``) so
``verify()`` matches the sync `TruthOraclePlug` protocol. It does NOT
go through ``HybridNetworkClient`` (which is async-first); it does
re-use the same URL set from ``[network]`` config so endpoint changes
land in one place.

``evidence_ref`` for accepted verdicts:
- solana_tx_confirmed → the signature
- solana_account_balance_gte → "<account>@<slot>" (slot from RPC response)
- solana_program_invoked → "<signature>@<program_id>"

``cost`` per call is nominal ``0.0001 SOL`` — a metering placeholder
since Helius free/premium tiers are flat-rate, not per-call. The
metabolic gate uses ``daily_sol_budget`` to cap call volume; cost
units stay consistent with other metered oracles.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import requests

from titan_hcl.synthesis.plugs import OracleClaim, OracleVerdict

logger = logging.getLogger(__name__)


SUPPORTED_DOMAINS = frozenset(
    {
        "solana_tx_confirmed",
        "solana_account_balance_gte",
        "solana_program_invoked",
    }
)

# Nominal per-call SOL cost — metering placeholder. Helius tiers are flat
# rate; this value just gives the INV-Syn-13 daily_sol_budget cap a unit
# to count against (~1000 calls/day at default budget 0.1 SOL).
DEFAULT_PER_CALL_COST_SOL: float = 0.0001

# Default per-request timeout for HTTPS JSON-RPC. Premium Solana RPC (Helius)
# typically responds in 1-3s; we cap at 5s so a degraded endpoint surfaces
# fast as "unknown" instead of stalling the synthesis_worker tick.
DEFAULT_TIMEOUT_S: float = 5.0


class SolanaRpcOracle:
    """TruthOraclePlug wrapping Solana JSON-RPC.

    Construction takes either an explicit ``rpc_url`` (preferred —
    typically Helius for premium reliability) OR a list of fallback
    URLs (e.g. mainnet-beta public RPCs). The plug tries them in
    order on each verify() until one returns a non-network-error
    response.

    Both args may be set via the network config block (``[network]``
    in ``config.toml``); the synthesis_worker passes the resolved
    URL list at construction.
    """

    oracle_id: str = "helius_rpc"     # matches the daily_sol_budget key in titan_params
    cost_class: str = "metered"       # INV-Syn-13: gated

    def __init__(
        self,
        *,
        rpc_url: Optional[str] = None,
        fallback_urls: Optional[list[str]] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        per_call_cost_sol: float = DEFAULT_PER_CALL_COST_SOL,
    ):
        # Build the ordered URL list; premium first if given, then fallbacks.
        # Empty list → the plug exists but every verify() returns
        # "unknown" — useful for tests + when the Titan starts offline.
        urls: list[str] = []
        if rpc_url:
            urls.append(rpc_url)
        if fallback_urls:
            urls.extend(u for u in fallback_urls if u and u != rpc_url)
        self._urls: tuple[str, ...] = tuple(urls)
        self._timeout_s = float(timeout_s)
        self._per_call_cost_sol = float(per_call_cost_sol)

    @property
    def urls(self) -> tuple[str, ...]:
        return self._urls

    def can_handle(self, domain: str) -> bool:
        return domain in SUPPORTED_DOMAINS

    # ── affective grounding helper (RFP_affective_grounding_loop §7.C) ───
    def latest_funding_feepayer(self, wallet_pubkey: str) -> Optional[str]:
        """Best-effort: the fee-payer pubkey of the most recent transaction
        touching `wallet_pubkey`. Used by the Affective Grounding Loop to tag a
        positive balance delta as Maker-originated (`maker_bond` signal) when the
        funding tx's fee-payer == the Maker pubkey.

        ONE getSignaturesForAddress(limit=1) + ONE getTransaction(jsonParsed);
        returns the fee-payer (the first signer account) pubkey string, or None on
        any RPC/parse failure or when no RPC is configured. Never raises — the
        caller treats None as "could not attribute" (no maker_bond this event).
        It is invoked only on a real +balance delta (rare), so the un-gated extra
        RPC pair is naturally bounded by funding frequency, not per-tick."""
        if not self._urls or not wallet_pubkey:
            return None
        try:
            sig_resp = self._rpc_post({
                "jsonrpc": "2.0", "id": 1,
                "method": "getSignaturesForAddress",
                "params": [str(wallet_pubkey), {"limit": 1}],
            })
            result = (sig_resp or {}).get("result") or []
            if not result:
                return None
            sig = result[0].get("signature")
            if not sig:
                return None
            tx_resp = self._rpc_post({
                "jsonrpc": "2.0", "id": 1,
                "method": "getTransaction",
                "params": [sig, {"encoding": "jsonParsed",
                                 "maxSupportedTransactionVersion": 0}],
            })
            tx = (tx_resp or {}).get("result") or {}
            account_keys = (
                ((tx.get("transaction") or {}).get("message") or {})
                .get("accountKeys") or [])
            # Fee-payer = the first signer account. jsonParsed → list of dicts
            # with {pubkey, signer, writable, source}; legacy → list of str
            # (index 0 is the fee-payer).
            for ak in account_keys:
                if isinstance(ak, dict):
                    if ak.get("signer"):
                        return str(ak.get("pubkey") or "") or None
                elif isinstance(ak, str):
                    return str(ak) or None
            return None
        except Exception:
            logger.debug(
                "[solana_rpc_oracle] latest_funding_feepayer failed",
                exc_info=True)
            return None

    # ── verify entry point ──────────────────────────────────────────────

    def verify(self, claim: OracleClaim) -> OracleVerdict:
        t0 = time.perf_counter()
        ts_now = time.time()

        if not self.can_handle(claim.domain):
            return self._verdict(
                t0=t0,
                ts_now=ts_now,
                verdict="unknown",
                evidence_ref="domain_unsupported",
            )

        if not self._urls:
            # No RPC configured — graceful degrade to "unknown" so the
            # router still anchors an auditable verdict.
            return self._verdict(
                t0=t0,
                ts_now=ts_now,
                verdict="unknown",
                evidence_ref="no_rpc_configured",
            )

        payload = claim.payload or {}

        try:
            if claim.domain == "solana_tx_confirmed":
                return self._verify_tx_confirmed(payload, t0, ts_now)
            if claim.domain == "solana_account_balance_gte":
                return self._verify_balance_gte(payload, t0, ts_now)
            if claim.domain == "solana_program_invoked":
                return self._verify_program_invoked(payload, t0, ts_now)
        except Exception:
            logger.exception("[solana_rpc_oracle] verify() raised")
            return self._verdict(
                t0=t0,
                ts_now=ts_now,
                verdict="unknown",
                evidence_ref="rpc_exception",
            )

        # Fall-through (unreachable because can_handle filters domains).
        return self._verdict(  # pragma: no cover
            t0=t0,
            ts_now=ts_now,
            verdict="unknown",
            evidence_ref="domain_unsupported",
        )

    # ── domain-specific verifiers ───────────────────────────────────────

    def _verify_tx_confirmed(
        self, payload: dict[str, Any], t0: float, ts_now: float
    ) -> OracleVerdict:
        sig = str(payload.get("signature", "")).strip()
        if not sig:
            return self._verdict(t0, ts_now, "unknown", "missing_signature")

        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignatureStatuses",
            "params": [[sig], {"searchTransactionHistory": True}],
        }
        resp = self._rpc_post(body)
        if resp is None:
            return self._verdict(t0, ts_now, "unknown", "rpc_unreachable", cost=self._per_call_cost_sol)

        result = (resp.get("result") or {}).get("value") or [None]
        status = result[0] if result else None
        if status is None:
            return self._verdict(t0, ts_now, "false", sig, cost=self._per_call_cost_sol)

        # err non-null → tx landed but failed → "false" (the signature
        # was confirmed on-chain but the transaction itself errored).
        if status.get("err") is not None:
            return self._verdict(t0, ts_now, "false", sig, cost=self._per_call_cost_sol)

        confirmation = (status.get("confirmationStatus") or "").lower()
        if confirmation in ("confirmed", "finalized"):
            return self._verdict(t0, ts_now, "true", sig, cost=self._per_call_cost_sol)
        # Processed but not yet confirmed/finalized — treat as "unknown"
        # so the caller can re-verify after a few slots.
        return self._verdict(t0, ts_now, "unknown", sig, cost=self._per_call_cost_sol)

    def _verify_balance_gte(
        self, payload: dict[str, Any], t0: float, ts_now: float
    ) -> OracleVerdict:
        account = str(payload.get("account", "")).strip()
        try:
            threshold = int(payload.get("lamports", 0))
        except (TypeError, ValueError):
            return self._verdict(t0, ts_now, "unknown", "bad_lamports_arg")
        if not account:
            return self._verdict(t0, ts_now, "unknown", "missing_account")

        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBalance",
            "params": [account, {"commitment": "confirmed"}],
        }
        resp = self._rpc_post(body)
        if resp is None:
            return self._verdict(t0, ts_now, "unknown", "rpc_unreachable", cost=self._per_call_cost_sol)

        result = resp.get("result")
        if not isinstance(result, dict):
            return self._verdict(t0, ts_now, "unknown", "malformed_rpc_response", cost=self._per_call_cost_sol)

        try:
            balance = int(result.get("value", 0))
        except (TypeError, ValueError):
            return self._verdict(t0, ts_now, "unknown", "non_integer_balance", cost=self._per_call_cost_sol)
        slot = result.get("context", {}).get("slot")
        verdict = "true" if balance >= threshold else "false"
        evidence = f"{account}@{slot}" if slot is not None else account
        return self._verdict(t0, ts_now, verdict, evidence, cost=self._per_call_cost_sol)

    def _verify_program_invoked(
        self, payload: dict[str, Any], t0: float, ts_now: float
    ) -> OracleVerdict:
        sig = str(payload.get("signature", "")).strip()
        program_id = str(payload.get("program_id", "")).strip()
        if not sig or not program_id:
            return self._verdict(t0, ts_now, "unknown", "missing_signature_or_program")

        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
        }
        resp = self._rpc_post(body)
        if resp is None:
            return self._verdict(t0, ts_now, "unknown", "rpc_unreachable", cost=self._per_call_cost_sol)

        tx = resp.get("result")
        if tx is None:
            # Signature not found at all.
            return self._verdict(t0, ts_now, "false", f"{sig}@{program_id}", cost=self._per_call_cost_sol)
        message = (tx.get("transaction") or {}).get("message") or {}
        keys = message.get("accountKeys") or []
        # Account keys may be strings (legacy) or {pubkey, ...} dicts (jsonParsed).
        for k in keys:
            if isinstance(k, dict) and k.get("pubkey") == program_id:
                return self._verdict(t0, ts_now, "true", f"{sig}@{program_id}", cost=self._per_call_cost_sol)
            if isinstance(k, str) and k == program_id:
                return self._verdict(t0, ts_now, "true", f"{sig}@{program_id}", cost=self._per_call_cost_sol)
        return self._verdict(t0, ts_now, "false", f"{sig}@{program_id}", cost=self._per_call_cost_sol)

    # ── plumbing ────────────────────────────────────────────────────────

    def _rpc_post(self, body: dict[str, Any]) -> Optional[dict[str, Any]]:
        """POST a JSON-RPC body to the first reachable URL.

        Returns the parsed JSON dict on success; ``None`` on every URL
        failing (so the caller emits an "unknown" verdict). Non-200
        responses count as failures and roll over to the next URL.
        """
        last_err: Optional[str] = None
        for url in self._urls:
            try:
                resp = requests.post(
                    url,
                    json=body,
                    headers={"Content-Type": "application/json"},
                    timeout=self._timeout_s,
                )
                if resp.status_code != 200:
                    last_err = f"{url} → HTTP {resp.status_code}"
                    continue
                parsed = resp.json()
                # JSON-RPC carries application-level errors in `error`.
                if isinstance(parsed, dict) and parsed.get("error"):
                    last_err = f"{url} → rpc error {parsed['error']}"
                    continue
                return parsed if isinstance(parsed, dict) else None
            except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
                last_err = f"{url} → {type(exc).__name__}: {exc}"
                continue
        if last_err:
            logger.warning("[solana_rpc_oracle] all URLs failed: %s", last_err)
        return None

    def _verdict(
        self,
        t0: float,
        ts_now: float,
        verdict: str,
        evidence_ref: str,
        *,
        cost: float = 0.0,
    ) -> OracleVerdict:
        return OracleVerdict(
            oracle_id=self.oracle_id,
            verdict=verdict,  # type: ignore[arg-type]  # narrowed by callers to {"true","false","unknown"}
            evidence_ref=evidence_ref,
            cost=cost,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            ts=ts_now,
        )


__all__ = ("SolanaRpcOracle", "SUPPORTED_DOMAINS", "DEFAULT_PER_CALL_COST_SOL")
