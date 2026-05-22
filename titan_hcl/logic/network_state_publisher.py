"""
network_state_publisher — NetworkStatePublisher writes network_state.bin SHM slot.

Producer for network_state slot per SPEC §7.1 (D-SPEC-71 v1.10.0). G21
single-writer contract: only the titan_HCL parent (TitanKernel monitor_tick
loop) publishes here.

Source: HybridNetworkClient instance (titan_hcl.core.network) — reads
Solana RPC client state (balance cache, pubkey, RPC endpoint).

Closes the network.balance / network.info / network.account.* bus-cache
state-lookups per Preamble G18.
"""
from __future__ import annotations

import time
from typing import Any

from titan_hcl.logic.base_state_publisher import BaseStatePublisher
from titan_hcl.logic.network_state_specs import (
    NETWORK_STATE_SLOT,
    NETWORK_STATE_SPEC,
)
from titan_hcl._phase_c_constants import NETWORK_STATE_SCHEMA_VERSION


class NetworkStatePublisher(BaseStatePublisher):
    slot_name = NETWORK_STATE_SLOT
    slot_spec = NETWORK_STATE_SPEC

    def _compute_payload(self, network: Any) -> dict[str, Any]:
        if network is None:
            return self._stub()
        # HybridNetworkClient public surface: .balance (float), .pubkey,
        # .rpc_urls, .premium_rpc, .rpc_endpoint, .is_available, ._account_cache
        balance = 0.0
        pubkey = ""
        rpc_urls: list[str] = []
        premium_rpc = None
        rpc_endpoint = ""
        is_available = False
        recent_account_data: dict[str, dict] = {}
        try:
            balance = float(getattr(network, "balance", 0.0) or 0.0)
        except Exception:
            pass
        try:
            pubkey = str(getattr(network, "pubkey", "") or "")
        except Exception:
            pass
        try:
            urls = getattr(network, "rpc_urls", []) or []
            if isinstance(urls, list):
                rpc_urls = [str(u) for u in urls]
        except Exception:
            pass
        try:
            premium_rpc = getattr(network, "premium_rpc", None)
            if premium_rpc is not None:
                premium_rpc = str(premium_rpc)
        except Exception:
            pass
        try:
            rpc_endpoint = str(getattr(network, "rpc_endpoint", "") or "")
        except Exception:
            pass
        try:
            is_available = bool(getattr(network, "is_available", False))
        except Exception:
            pass
        try:
            cache = getattr(network, "_account_cache", None)
            if isinstance(cache, dict):
                # Keep payload bounded — only first 8 cached PDAs, each
                # truncated to scalar fields.
                for pda, data in list(cache.items())[:8]:
                    if isinstance(data, dict):
                        recent_account_data[str(pda)] = {
                            k: v for k, v in data.items()
                            if isinstance(v, (int, float, str, bool))
                        }
        except Exception:
            pass
        return {
            "balance_sol": balance,
            "pubkey": pubkey,
            "premium_rpc": premium_rpc,
            "rpc_urls": rpc_urls,
            "rpc_endpoint": rpc_endpoint,
            "recent_account_data": recent_account_data,
            "last_balance_update_ts": 0.0,  # HybridNetworkClient doesn't expose this
            "last_info_update_ts": 0.0,
            "network_available": is_available,
            "schema_version": NETWORK_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }

    def _stub(self) -> dict[str, Any]:
        return {
            "balance_sol": 0.0,
            "pubkey": "",
            "premium_rpc": None,
            "rpc_urls": [],
            "rpc_endpoint": "",
            "recent_account_data": {},
            "last_balance_update_ts": 0.0,
            "last_info_update_ts": 0.0,
            "network_available": False,
            "schema_version": NETWORK_STATE_SCHEMA_VERSION,
            "ts": time.time(),
        }
