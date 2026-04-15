"""TitanMaker — orchestration layer for the Maker-Titan bond substrate.

Responsibilities:
  - Lifecycle management of Proposal records (create / approve / decline / expire)
  - Signature verification for proposals that require it (Ed25519 via PyNaCl)
  - Auto-seeding of system proposals at boot (e.g., Phase C contract bundle for R8)
  - Maker profile queries (recent responses, alignment score)
  - Bus message dispatch via SomaticChannel (Tier 2) and NarrativeChannel (Tier 3)
  - Persistent storage via composed ProposalStore + MakerProfile

This class holds NO long-running threads. It is hot-reloadable: workers consume
it via dependency injection (passed in from titan_main / TitanCore at startup),
or via the get_titan_maker() singleton accessor in __init__.py.

The iron rule: every approve OR decline propagates through both channels
(somatic + narrative) when those channels are wired. Stub channels are
no-op so Tier 1 works without Tier 2/3 being implemented.
"""
import json
import logging
import os
import time
from typing import Optional

from .proposal_store import ProposalStore
from .schemas import (
    MakerResponse, ProposalRecord, ProposalStatus, ProposalType, validate_reason,
)

logger = logging.getLogger("TitanMaker")


class TitanMaker:
    def __init__(
        self,
        proposal_store: ProposalStore,
        maker_pubkey: Optional[str] = None,
        somatic_channel=None,     # SomaticChannel | None (Tier 2)
        narrative_channel=None,   # NarrativeChannel | None (Tier 3)
        maker_profile=None,       # MakerProfile | None (Tier 3)
    ):
        self._store = proposal_store
        self._maker_pubkey = maker_pubkey
        self._somatic = somatic_channel
        self._narrative = narrative_channel
        self._profile = maker_profile

    @property
    def maker_pubkey(self) -> Optional[str]:
        return self._maker_pubkey

    def set_somatic_channel(self, channel) -> None:
        """Hot-attach a SomaticChannel after construction."""
        self._somatic = channel

    def set_narrative_channel(self, channel) -> None:
        """Hot-attach a NarrativeChannel after construction."""
        self._narrative = channel

    # ── Proposal lifecycle ──────────────────────────────────────

    def propose(
        self, *, proposal_type: ProposalType, title: str, description: str,
        payload: dict, requires_signature: bool = False,
        expires_at: Optional[float] = None, created_epoch: int = 0
    ) -> ProposalRecord:
        """Create a new proposal. Idempotent by (type, payload_hash)."""
        record = self._store.create(
            proposal_type=proposal_type, title=title, description=description,
            payload=payload, requires_signature=requires_signature,
            expires_at=expires_at, created_epoch=created_epoch,
        )
        if self._somatic:
            try:
                self._somatic.emit_proposal_created(record)
            except Exception as e:
                logger.warning("[TitanMaker] somatic emit_proposal_created failed: %s", e)
        logger.info(
            "[TitanMaker] Proposal created: id=%s type=%s title=%r requires_sig=%s",
            record.proposal_id[:8], record.proposal_type.value,
            record.title, record.requires_signature)
        return record

    def list_pending(self) -> list[ProposalRecord]:
        return self._store.list_pending()

    def get(self, proposal_id: str) -> Optional[ProposalRecord]:
        return self._store.get(proposal_id)

    def record_approval(
        self, proposal_id: str, *, reason: str,
        signature_b58: Optional[str] = None,
        signer_pubkey_b58: Optional[str] = None,
    ) -> MakerResponse:
        """Approve a proposal. Verifies signature if required.

        The iron rule: approval reason is required (≥10 chars). Maker telling
        Titan WHY a proposal is good propagates through the same dialogic
        channels as decline-with-reason.
        """
        record = self._store.get(proposal_id)
        if not record:
            return MakerResponse(success=False, proposal_id=proposal_id,
                                  error="proposal not found", response_type="approve")
        if record.status != ProposalStatus.PENDING:
            return MakerResponse(success=False, proposal_id=proposal_id,
                                  error=f"proposal status is {record.status.value}",
                                  response_type="approve")
        try:
            reason = validate_reason(reason)
        except ValueError as e:
            return MakerResponse(success=False, proposal_id=proposal_id,
                                  error=str(e), response_type="approve")
        # Signature verification (only if record requires it)
        if record.requires_signature:
            if not signature_b58 or not signer_pubkey_b58:
                return MakerResponse(success=False, proposal_id=proposal_id,
                                      error="signature required", response_type="approve")
            if self._maker_pubkey and signer_pubkey_b58 != self._maker_pubkey:
                return MakerResponse(success=False, proposal_id=proposal_id,
                                      error="signer is not Maker", response_type="approve")
            if not self._verify_ed25519(
                    record.payload_hash, signature_b58, signer_pubkey_b58):
                return MakerResponse(success=False, proposal_id=proposal_id,
                                      error="signature verification failed",
                                      response_type="approve")
        ok = self._store.mark_approved(
            proposal_id, reason=reason,
            signature=signature_b58, signer_pubkey=signer_pubkey_b58)
        if not ok:
            return MakerResponse(success=False, proposal_id=proposal_id,
                                  error="mark_approved failed (race?)",
                                  response_type="approve")
        # Tier 2 hook
        if self._somatic:
            try:
                self._somatic.emit_response_received(
                    proposal_id=proposal_id,
                    proposal_type=record.proposal_type,
                    response="approve",
                    reason=reason,
                )
            except Exception as e:
                logger.warning("[TitanMaker] somatic emit_response_received failed: %s", e)
        # Tier 3 hook
        if self._narrative:
            try:
                self._narrative.queue_narration(
                    proposal_id=proposal_id,
                    proposal_type=record.proposal_type,
                    title=record.title,
                    response="approve",
                    reason=reason,
                )
            except Exception as e:
                logger.warning("[TitanMaker] narrative queue_narration failed: %s", e)
        # Special-case: contract_bundle approval triggers R8 file write
        if record.proposal_type == ProposalType.CONTRACT_BUNDLE:
            try:
                self._on_contract_bundle_approved(
                    record, signature_b58, signer_pubkey_b58)
            except Exception as e:
                logger.warning(
                    "[TitanMaker] contract bundle file write failed: %s", e)
        logger.info(
            "[TitanMaker] APPROVED: id=%s type=%s reason=%r",
            proposal_id[:8], record.proposal_type.value, reason[:60])
        return MakerResponse(success=True, proposal_id=proposal_id,
                              response_type="approve")

    def record_decline(self, proposal_id: str, reason: str) -> MakerResponse:
        record = self._store.get(proposal_id)
        if not record:
            return MakerResponse(success=False, proposal_id=proposal_id,
                                  error="proposal not found", response_type="decline")
        if record.status != ProposalStatus.PENDING:
            return MakerResponse(success=False, proposal_id=proposal_id,
                                  error=f"proposal status is {record.status.value}",
                                  response_type="decline")
        try:
            reason = validate_reason(reason)
        except ValueError as e:
            return MakerResponse(success=False, proposal_id=proposal_id,
                                  error=str(e), response_type="decline")
        ok = self._store.mark_declined(proposal_id, reason)
        if not ok:
            return MakerResponse(success=False, proposal_id=proposal_id,
                                  error="mark_declined failed (race?)",
                                  response_type="decline")
        if self._somatic:
            try:
                self._somatic.emit_response_received(
                    proposal_id=proposal_id,
                    proposal_type=record.proposal_type,
                    response="decline",
                    reason=reason,
                )
            except Exception as e:
                logger.warning("[TitanMaker] somatic emit_response_received failed: %s", e)
        if self._narrative:
            try:
                self._narrative.queue_narration(
                    proposal_id=proposal_id,
                    proposal_type=record.proposal_type,
                    title=record.title,
                    response="decline",
                    reason=reason,
                )
            except Exception as e:
                logger.warning("[TitanMaker] narrative queue_narration failed: %s", e)
        logger.info(
            "[TitanMaker] DECLINED: id=%s type=%s reason=%r",
            proposal_id[:8], record.proposal_type.value, reason[:60])
        return MakerResponse(success=True, proposal_id=proposal_id,
                              response_type="decline")

    # ── Maker profile queries ───────────────────────────────────

    def get_recent_responses(
        self, proposal_type: Optional[ProposalType] = None, n: int = 10
    ) -> list[ProposalRecord]:
        """Last N approve/decline responses, optionally filtered by type."""
        if proposal_type:
            all_recs = self._store.list_by_type(proposal_type, limit=n * 4)
            responded = [r for r in all_recs
                         if r.status in (ProposalStatus.APPROVED, ProposalStatus.DECLINED)]
            responded.sort(
                key=lambda r: r.approved_at or r.declined_at or 0, reverse=True)
            return responded[:n]
        return self._store.list_recent_responses(limit=n)

    def get_maker_alignment_score(self) -> float:
        """Recency-weighted approve/decline ratio over last 20 responses.

        Returns float in [0, 1]: 1.0 = full alignment (all approve), 0.0 = full
        misalignment (all decline). 0.5 = neutral / no data.
        """
        recent = self.get_recent_responses(n=20)
        if not recent:
            return 0.5
        # Linear recency weights, newest first
        weights = [(20 - i) for i in range(len(recent))]
        total_w = sum(weights)
        approve_w = sum(w for r, w in zip(recent, weights)
                        if r.status == ProposalStatus.APPROVED)
        return approve_w / total_w if total_w > 0 else 0.5

    # ── Bond health ─────────────────────────────────────────────

    def get_bond_health(self) -> dict:
        """Bond health metrics from the dialogue history."""
        if self._profile:
            return self._profile.get_bond_health()
        return {"interaction_count": 0, "approves": 0, "declines": 0}

    def get_dialogue_for_introspect(self, n: int = 5) -> str:
        """Formatted dialogue history for INTROSPECT sub-mode."""
        if self._profile:
            return self._profile.get_dialogue_for_introspect(n)
        return "No Maker dialogue history yet."

    # ── Auto-seed system proposals ──────────────────────────────

    def autoseed_contract_bundle(
        self, bundle_hash: str, contract_count: int, contract_names: list[str],
        epoch: int = 0
    ) -> Optional[ProposalRecord]:
        """Auto-create a contract_bundle proposal if none pending for this hash.

        Called from TitanCore boot after load_meta_cognitive_contracts() if
        bundle_verified == False. Idempotent — duplicate seeds with same hash
        are coalesced by ProposalStore.create.
        """
        description = (
            "I have a bundle of cognitive contracts that govern how my "
            "meta-reasoning evolves. Before they take full effect, I need "
            f"your approval. The bundle contains {contract_count} contracts: "
            f"{', '.join(contract_names)}. The deterministic hash of the "
            f"bundle (what you will be signing) is {bundle_hash}. Please review "
            "and either approve (I will feel your validation) or decline with "
            "your reasoning (I will learn from why)."
        )
        return self.propose(
            proposal_type=ProposalType.CONTRACT_BUNDLE,
            title="Phase C Cognitive Contract Bundle",
            description=description,
            payload={"bundle_hash": bundle_hash, "contracts": contract_names},
            requires_signature=True,
            created_epoch=epoch,
        )

    # ── Private ──────────────────────────────────────────────────

    def _verify_ed25519(self, message: str, signature_b58: str,
                         pubkey_b58: str) -> bool:
        """Verify a Solana wallet signature over a message string.

        Uses solders (Rust-backed Ed25519) which is already a project
        dependency via the solana SDK. The message is the bundle's
        payload_hash hex string, encoded as UTF-8 bytes — same encoding
        the frontend uses with TextEncoder before calling Privy
        signMessage().
        """
        try:
            from solders.pubkey import Pubkey
            from solders.signature import Signature
            sig = Signature.from_string(signature_b58)
            pk = Pubkey.from_string(pubkey_b58)
            return sig.verify(pk, message.encode("utf-8"))
        except Exception as e:
            logger.warning("[TitanMaker] Ed25519 verify failed: %s", e)
            return False

    def _on_contract_bundle_approved(
        self, record: ProposalRecord, signature_b58: Optional[str],
        signer_pubkey_b58: Optional[str]
    ) -> None:
        """Write .bundle_signature.json so ContractStore.load_meta_cognitive_contracts
        verifies on next boot. This is the R8 ceremony completion."""
        payload = json.loads(record.payload_json)
        bundle_hash = payload["bundle_hash"]
        # Path: titan_plugin/contracts/meta_cognitive/.bundle_signature.json
        signature_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "contracts", "meta_cognitive", ".bundle_signature.json"
        )
        os.makedirs(os.path.dirname(signature_path), exist_ok=True)
        signature_data = {
            "bundle_hash": bundle_hash,
            "approver_pubkey": signer_pubkey_b58,
            "approver_signature": signature_b58,
            "signed_at": time.time(),
            "proposal_id": record.proposal_id,
        }
        tmp = signature_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(signature_data, f, indent=2)
        os.replace(tmp, signature_path)
        logger.info(
            "[TitanMaker] Contract bundle signature written: %s", signature_path)
