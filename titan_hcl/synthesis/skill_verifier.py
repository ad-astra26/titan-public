"""SkillVerifier — INV-Syn-20 first-invocation `compiled_from` re-verification.

Per `ARCHITECTURE_synthesis_engine.md §8.4` + `PLAN_synthesis_engine_Phase8.md §P8.F`.

Synthesis-worker-side guard. Fires ONCE per skill (state: `verified_at IS NULL`
→ fire → `verified_at=now` on pass, `utility_score=-1.0` + `verified_at=now`
on fail). Read-only check (NOT replay — no side effects). Anchors
META_SKILL_VERIFIED / META_SKILL_REJECTED TXs on the meta fork.

Closes §8.4 "compilation hallucination mitigation":
  > `compiled_from` lineage requires re-verification at first invocation.

The agno-side `match_procedural_skill` tool returns `null` for any candidate
with `verified_at IS NULL`; the dream-time pass picks those candidates up,
verifies via SkillVerifier, and the next invocation finds them eligible.

`chain_reader` protocol: any object exposing
  `read_tx_by_content_hash(h: str) -> Optional[dict]`
returning a dict with at least `{content_hash}` field on hit, None on miss.

Production wires synthesis_worker's TimeChain index reader behind this
protocol. Tests inject in-memory fakes.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class _ChainReaderLike(Protocol):
    """Minimal contract for the chain reader: resolve content_hash → block dict."""
    def read_tx_by_content_hash(self, h: str) -> Optional[dict]: ...


class SkillVerifier:
    """INV-Syn-20 first-invocation chain-resolve + content-hash check."""

    def __init__(
        self,
        *,
        skill_store: Any,
        chain_reader: _ChainReaderLike,
        outer_memory_writer: Any,
        bus_emit: Optional[Any] = None,
    ):
        self._store = skill_store
        self._chain = chain_reader
        self._writer = outer_memory_writer
        self._bus_emit = bus_emit

    def verify_skill(self, skill_id: str) -> bool:
        """Return True on pass, False on reject.

        Idempotent — skipping a skill with verified_at IS NOT NULL is the
        whole point of the gate (verify once, trust forever for that
        compiled_from snapshot).
        """
        if not skill_id:
            return False
        skill = self._store.read_skill(skill_id)
        if skill is None:
            logger.debug("[SkillVerifier] skill %s not found — refuse", skill_id)
            return False
        # Already verified — short-circuit
        if skill.get("verified_at") is not None:
            return float(skill.get("utility_score") or 0.0) >= 0.0
        # Rejected skills carry utility_score=-1.0 + verified_at; the above
        # branch handles them. If somehow verified_at is None AND utility=-1,
        # treat as fresh + re-verify (defensive).
        compiled_from = skill.get("compiled_from") or []
        if not compiled_from:
            return self._reject(skill_id, reason="empty_compiled_from")

        # Walk every source tx_hash; any miss / hash mismatch → reject.
        for tx_hash in compiled_from:
            if not isinstance(tx_hash, str) or not tx_hash:
                return self._reject(skill_id, reason="malformed_tx_hash")
            try:
                block = self._chain.read_tx_by_content_hash(tx_hash)
            except Exception as e:
                logger.warning(
                    "[SkillVerifier] chain_reader raised for %s: %s — reject", tx_hash, e,
                )
                return self._reject(skill_id, reason=f"chain_reader_exception:{type(e).__name__}",
                                   compiled_from=compiled_from)
            if block is None:
                return self._reject(skill_id, reason=f"chain_resolve_miss:{tx_hash[:16]}",
                                   compiled_from=compiled_from)
            # Content-hash equality: the block's recorded content_hash MUST
            # match the tx_hash we recorded at compile time. The chain reader's
            # CONTRACT is to return None on miss and a block ONLY on a hash
            # match (ChainContentHashReader resolves BY content_hash), so a
            # block returned WITHOUT an explicit content_hash field is
            # authoritative (presence ⟺ match). We still reject an explicit
            # MISMATCH defensively. (AUDIT §5.3 G2 proposed fail-closing the
            # absent case, but that contradicts the documented reader contract
            # + the test_verify_skill_accepts_when_reader_returns_block_without_
            # hash_field invariant — declined as a false positive.)
            block_hash = block.get("content_hash") or block.get("tx_hash")
            if block_hash and block_hash != tx_hash:
                return self._reject(skill_id, reason=f"content_hash_mismatch:{tx_hash[:16]}",
                                   compiled_from=compiled_from)

        # All sources resolved + match — verify
        return self._accept(skill_id, compiled_from=compiled_from)

    def is_eligible(self, skill_id: str) -> bool:
        """Convenience: True iff the skill has been verified AND not rejected
        (utility_score >= 0)."""
        skill = self._store.read_skill(skill_id)
        if skill is None:
            return False
        if skill.get("verified_at") is None:
            return False
        return float(skill.get("utility_score") or 0.0) >= 0.0

    # ── Internals ─────────────────────────────────────────────────────

    def _accept(self, skill_id: str, *, compiled_from: list) -> bool:
        self._store.mark_verified(skill_id)
        try:
            self._writer.write_skill_lifecycle_tx(
                skill_id=skill_id, event_kind="verified",
                compiled_from=list(compiled_from or []),
            )
        except Exception as e:
            logger.warning("[SkillVerifier] write_skill_lifecycle_tx (verified) failed: %s", e)
        self._emit("META_SKILL_VERIFIED", {
            "skill_id": skill_id,
            "compiled_from_count": len(compiled_from or []),
        })
        return True

    def _reject(self, skill_id: str, *, reason: str, compiled_from: Optional[list] = None) -> bool:
        self._store.mark_rejected(skill_id, reason=reason)
        try:
            self._writer.write_skill_lifecycle_tx(
                skill_id=skill_id, event_kind="rejected",
                reason=reason,
                compiled_from=list(compiled_from or []),
            )
        except Exception as e:
            logger.warning("[SkillVerifier] write_skill_lifecycle_tx (rejected) failed: %s", e)
        self._emit("META_SKILL_REJECTED", {
            "skill_id": skill_id,
            "reason": reason,
        })
        return False

    def _emit(self, event: str, payload: dict) -> None:
        if self._bus_emit is None:
            return
        try:
            self._bus_emit(event, payload)
        except Exception as e:
            logger.warning("[SkillVerifier] bus_emit %s failed: %s", event, e)
