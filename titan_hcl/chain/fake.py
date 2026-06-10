"""FakeChainProvider — in-memory ChainProvider for hermetic tests.

Mirrors the test FakeArweave/FakeMemoStore but behind the real `ChainProvider`
interface, so every redesign test (backup, restore, orchestrator) shares ONE
double. Deterministic, no network, no daemon. Data plane is fully implemented;
trust + funding are implemented minimally so Phase B/C tests can use it too.
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

from titan_hcl.chain.provider import ChainProvider, HeadStatus, PutSource, _CHUNK


class FakeChainProvider(ChainProvider):
    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}          # tx_id → bytes
        self._memos: dict[str, str] = {}            # tx_sig → memo_text
        self._memo_order: list[str] = []            # sigs in commit order (list_memos)
        self.put_log: list[str] = []
        self.unverified: set[str] = set()           # tx_ids → force "unverified" on head
        self._balance_sol: float = 1.0
        self.fund_log: list[float] = []

    # ── DATA plane ──
    async def put(self, src: PutSource, *,
                  content_type: str = "application/octet-stream",
                  tags: Optional[dict] = None) -> str:
        data = self._read(src)
        tx_id = "fake_" + hashlib.sha256(data).hexdigest()[:40]
        self._store[tx_id] = data
        self.put_log.append(tx_id)
        return tx_id

    async def get_to_file(self, tx_id: str, dest_path: str) -> bool:
        blob = self._store.get(tx_id)
        if blob is None:
            return False
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        with open(dest_path, "wb") as fh:                # chunked, mirrors the real streamer
            for i in range(0, len(blob), _CHUNK):
                fh.write(blob[i:i + _CHUNK])
        return True

    async def get_bytes(self, tx_id: str) -> Optional[bytes]:
        return self._store.get(tx_id)

    async def head(self, tx_id: str) -> HeadStatus:
        if tx_id in self.unverified:
            return "unverified"
        return "present" if tx_id in self._store else "missing"

    # ── TRUST plane (minimal; for Phase B tests) ──
    async def commit_memo(self, memo_text: str, *, state_root: Optional[str] = None,
                          sovereignty_bp: Optional[int] = None) -> Optional[str]:
        sig = "fakesig_" + hashlib.sha256(
            (memo_text + (state_root or "")).encode()).hexdigest()[:40]
        self._memos[sig] = memo_text
        self._memo_order.append(sig)
        return sig

    async def read_memo(self, tx_sig: str) -> Optional[str]:
        return self._memos.get(tx_sig)

    async def list_memos(self, address: str, *, limit: Optional[int] = None) -> list[str]:
        return list(reversed(self._memo_order))[:limit]   # newest-first; None ⇒ all

    # ── FUNDING plane (minimal; for Phase C tests) ──
    async def balance(self) -> float:
        return self._balance_sol

    async def fund(self, amount_sol: float, *, daily_cap_sol: float = 0.0) -> Optional[str]:
        if daily_cap_sol > 0:
            amount_sol = min(amount_sol, daily_cap_sol)   # mimic the real cap trim
        if amount_sol <= 0:
            return None
        self._balance_sol += amount_sol
        self.fund_log.append(amount_sol)
        return "fakefund_" + hashlib.sha256(str(amount_sol).encode()).hexdigest()[:32]

    # ── helpers ──
    @staticmethod
    def _read(src: PutSource) -> bytes:
        if isinstance(src, (bytes, bytearray)):
            return bytes(src)
        if isinstance(src, str):
            with open(src, "rb") as f:
                return f.read()
        raise TypeError(f"put src must be bytes or path str, got {type(src).__name__}")
