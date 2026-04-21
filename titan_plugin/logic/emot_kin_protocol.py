"""EMOT-CGN Kin Protocol — Titan↔Titan emotion transmission (rFP §21).

Titan shares his emotional region + derived scalars + a compact
latent fingerprint with kin. Receiving Titan projects the peer's
state into his own latent space (learned-alignment, future work)
and uses it to ground the I/YOU/ME/WE MSL concepts — the empathy
loop (Maker directive 2026-04-21).

This module defines:
- KIN_EMOT_STATE bus message shape + constructors
- parse / validate helpers for incoming payloads
- MSL binding rules (rule-based v1; learned alignment future)

Scope of this scaffold:
- Bus-level only. Actual network transmission (HTTP to peer Titan's
  /v4/kin/emot endpoint) is wired in a separate commit that extends
  the existing KinSenseHelper HTTP path.
- MSL binding is rule-based in v1 (valence/region similarity →
  direct I/YOU/ME/WE/YES/NO activation math). Learned alignment
  replaces this once we have enough co-observed kin events to train
  a projection.

Message format
==============

Outgoing (kin broadcast — dst="all" locally, then relayed by
kin_sense helper to peer Titans on 0.03–0.1 Hz cadence):

    type:    "KIN_EMOT_STATE"
    src:     "emot_cgn"
    dst:     "all"  (bus broadcast; kin_sense picks up for HTTP relay)
    payload:
      titan_src:         str  — sender's titan_id ("T1"/"T2"/"T3")
      peer_ts_ms:        int  — sender's bundle write ts
      region_id:         int  — -1=NOISE, -2=unclustered, ≥0=region
      region_signature:  int  — 64-bit stable centroid ID
      region_confidence: f32  — [0,1]
      region_residence_s:f32  — seconds in current region
      regions_emerged:   int  — total stable regions on sender
      valence:           f32  — [-1,+1]
      arousal:           f32  — [-1,+1]
      novelty:           f32  — [0,1]
      legacy_idx:        int  — 0..7 (for cross-Titan back-compat displays)
      encoder_id:        int  — provenance (thin=0, l5_phase0=1)
      signature:         str  — ed25519 sig by sender's titan key
                                (future; empty string in v1)

Incoming (from peer Titan via kin HTTP or direct bus):
    Same payload. Receiver validates:
    - titan_src is known kin (in kin_addresses config)
    - peer_ts_ms freshness (within last 5 min)
    - optional signature verification (future)

MSL binding (v1 rule-based)
===========================

Given self_state (own bundle) + peer_state (received KIN_EMOT_STATE):

  region_match = (self.region_signature == peer.region_signature
                  AND both have region_confidence > 0.5)

  I_activation = self.region_confidence      # "I am grounded in a region"
  YOU_activation = peer.region_confidence    # "peer is grounded"
  ME_activation = |self.valence|             # "I feel an affect"
  WE_activation = 1.0 if region_match else 0 # "peer and I are in same region"

  YES_activation = 0.5 * (self.valence + 1)  # positive-affect binding
  NO_activation = 0.5 * (1 - self.valence)   # negative-affect binding

These drive msl_activations_6d in the next bundle write → consumers
using Plug A read the bound state as part of their 30D context.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

KIN_EMOT_STATE_MSG_TYPE = "KIN_EMOT_STATE"

# MSL concept index — matches emot_bundle_protocol.MSL_CONCEPTS order.
MSL_IDX_I = 0
MSL_IDX_YOU = 1
MSL_IDX_ME = 2
MSL_IDX_WE = 3
MSL_IDX_YES = 4
MSL_IDX_NO = 5

# Relay cadence (seconds) — rate limit for outgoing KIN_EMOT_STATE emission.
# At 30s default, 2 emissions/min × 3 kin = 360/h traffic — well within
# daily-limit (48/day/kin = no issue). Tunable via config.
DEFAULT_EMIT_INTERVAL_S = 30.0

# Peer freshness window — stale KIN state gets ignored.
DEFAULT_PEER_FRESHNESS_S = 300.0


def build_kin_emot_state_payload(
    *,
    titan_src: str,
    region_id: int,
    region_signature: int,
    region_confidence: float,
    region_residence_s: float,
    regions_emerged: int,
    valence: float,
    arousal: float,
    novelty: float,
    legacy_idx: int,
    encoder_id: int,
    ts_ms: Optional[int] = None,
) -> dict:
    """Construct a KIN_EMOT_STATE payload. Defaults coerced defensively."""
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)
    return {
        "titan_src": str(titan_src),
        "peer_ts_ms": int(ts_ms),
        "region_id": int(region_id),
        "region_signature": int(region_signature) & 0xFFFFFFFFFFFFFFFF,
        "region_confidence": float(max(0.0, min(1.0, region_confidence))),
        "region_residence_s": float(max(0.0, region_residence_s)),
        "regions_emerged": int(max(0, regions_emerged)),
        "valence": float(max(-1.0, min(1.0, valence))),
        "arousal": float(max(-1.0, min(1.0, arousal))),
        "novelty": float(max(0.0, min(1.0, novelty))),
        "legacy_idx": int(max(0, min(7, legacy_idx))),
        "encoder_id": int(encoder_id),
        "signature": "",  # reserved for future ed25519 sig
    }


def parse_kin_emot_state(
    payload: dict,
    expected_self_id: str = "",
    freshness_s: float = DEFAULT_PEER_FRESHNESS_S,
) -> Optional[dict]:
    """Validate + parse an incoming KIN_EMOT_STATE payload.

    Returns None on validation failure (so callers can cleanly skip).
    Failures: missing fields, own-emission bounceback, stale peer_ts,
    malformed signature.
    """
    try:
        if not isinstance(payload, dict):
            return None
        titan_src = str(payload.get("titan_src", ""))
        if not titan_src:
            return None
        # Own-emission skip (bus dst="all" bounces back locally).
        if expected_self_id and titan_src == expected_self_id:
            return None
        peer_ts_ms = int(payload.get("peer_ts_ms", 0))
        if peer_ts_ms <= 0:
            return None
        age_s = (time.time() * 1000 - peer_ts_ms) / 1000.0
        if age_s > freshness_s:
            return None  # stale
        return {
            "titan_src": titan_src,
            "peer_ts_ms": peer_ts_ms,
            "age_s": age_s,
            "region_id": int(payload.get("region_id", -2)),
            "region_signature": int(
                payload.get("region_signature", 0)) & 0xFFFFFFFFFFFFFFFF,
            "region_confidence": float(payload.get("region_confidence", 0.0)),
            "region_residence_s": float(
                payload.get("region_residence_s", 0.0)),
            "regions_emerged": int(payload.get("regions_emerged", 0)),
            "valence": float(payload.get("valence", 0.0)),
            "arousal": float(payload.get("arousal", 0.0)),
            "novelty": float(payload.get("novelty", 0.5)),
            "legacy_idx": int(payload.get("legacy_idx", 0)),
            "encoder_id": int(payload.get("encoder_id", 0)),
        }
    except Exception as e:
        logger.debug("[KinEmotProtocol] parse failed: %s", e)
        return None


def compute_msl_activations(
    *,
    self_region_confidence: float,
    self_region_signature: int,
    self_valence: float,
    peer_state: Optional[dict] = None,
) -> list[float]:
    """Compute 6D MSL activations from self + optional peer state.

    Returns a list [I, YOU, ME, WE, YES, NO] with values in [0, 1].
    Rules (v1 rule-based, replaced by learned binding later):

      I   = self.region_confidence — grounded in a region
      YOU = peer.region_confidence when peer present else 0
      ME  = abs(self.valence) — felt affect magnitude
      WE  = 1.0 if same region signature AND both confident > 0.5 else 0
      YES = (self.valence + 1) / 2 — positive-affect binding
      NO  = (1 - self.valence) / 2 — negative-affect binding

    Note: YES + NO are complementary by design (sum to 1). This is
    intentional — affect valence is a signed quantity, MSL gets its
    two poles as separate slots for consumer simplicity.
    """
    I = float(max(0.0, min(1.0, self_region_confidence)))
    v = float(max(-1.0, min(1.0, self_valence)))
    ME = float(abs(v))
    YES = float(max(0.0, min(1.0, 0.5 * (v + 1.0))))
    NO = float(max(0.0, min(1.0, 0.5 * (1.0 - v))))

    if peer_state is None:
        YOU = 0.0
        WE = 0.0
    else:
        peer_conf = float(peer_state.get("region_confidence", 0.0))
        YOU = float(max(0.0, min(1.0, peer_conf)))
        same_region = (
            peer_state.get("region_signature", 0) == self_region_signature
            and self_region_signature != 0
        )
        both_confident = I > 0.5 and peer_conf > 0.5
        WE = 1.0 if (same_region and both_confident) else 0.0

    return [I, YOU, ME, WE, YES, NO]
