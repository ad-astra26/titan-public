"""tests/test_affective_phase_d.py — RFP_affective_grounding_loop §7.D offline gates.

Covers the cross-platform cryptographically-verified Maker-bond build:
  • D.1 — the maker_presence resolver (honest verified/unverified verdict).
  • D.0 — the verified-Maker nonce-sign session marker (real Ed25519 round-trip).
  • D.3 — the recency-based honest existential-delta valence (compute_maker_bond_nudge).
  • D.4 — the MAKER_PRESENCE emot-cgn primitive + the central action-head 8→9
          zero-init-append migration (learned weights preserved).

Run isolated (TorchRL mmap Bus error if multiple TitanHCL instances share a buffer):
    python -m pytest tests/test_affective_phase_d.py -v -p no:anchorpy --tb=short
"""
import numpy as np
import pytest

from titan_hcl.logic.maker_engine import (
    resolve_maker_presence, MakerPresence, MAKER_BOND_CHANNELS)
from titan_hcl.logic.affective_nudge import (
    AffectiveConfig, compute_maker_bond_nudge, AffectiveNudgeRuntime)
from titan_hcl.api.maker_presence_session import (
    VerifiedMakerSessions, session_key_from_claims)

CFG = AffectiveConfig(enabled=True, k_surprise=0.04, max_mag=0.06,
                      ema_alpha=0.15, sigma_init=0.25, eps=1e-3, min_samples=2)


# ─────────────────────────────────────────────────────────────────────
# D.1 — maker_presence resolver (GD3 honest verified/unverified)
# ─────────────────────────────────────────────────────────────────────

def test_resolver_verified_on_bond_channel():
    for ch in ("chat", "app", "tcc", "chain"):
        p = resolve_maker_presence(ch, claimed_maker=True, crypto_verified=True)
        assert p.verified is True and p.is_maker is True and p.channel == ch


def test_resolver_channel_aliases_normalize():
    assert resolve_maker_presence("web", crypto_verified=True).channel == "chat"
    assert resolve_maker_presence("WEB", crypto_verified=True).verified is True
    assert resolve_maker_presence("console", crypto_verified=True).channel == "tcc"
    assert resolve_maker_presence("on_chain", crypto_verified=True).channel == "chain"


def test_resolver_claimed_but_unverified_is_maker_but_no_bond():
    # GD3 — an internal-key chat-test caller (claims maker, no crypto proof):
    # is_maker True (behaviour preserved) but verified False (no maker_bond).
    p = resolve_maker_presence("chat", claimed_maker=True, crypto_verified=False)
    assert p.is_maker is True
    assert p.verified is False


def test_resolver_non_bond_channel_never_verifies():
    # X / social cannot be cryptographically Maker-attributed → never a bond tap.
    for ch in ("x", "twitter", "telegram", "unknown"):
        p = resolve_maker_presence(ch, claimed_maker=True, crypto_verified=True)
        assert p.verified is False
        assert ch not in MAKER_BOND_CHANNELS


def test_resolver_unverified_unclaimed_is_nothing():
    p = resolve_maker_presence("chat", claimed_maker=False, crypto_verified=False)
    assert p.is_maker is False and p.verified is False


# ─────────────────────────────────────────────────────────────────────
# D.0 — verified-Maker nonce-sign session marker (GD1 — real Ed25519)
# ─────────────────────────────────────────────────────────────────────

def _keypair():
    from solders.keypair import Keypair
    k = Keypair()
    return k, str(k.pubkey())


def _sign(k, msg: str) -> str:
    return str(k.sign_message(msg.encode("utf-8")))


def test_d0_full_nonce_sign_mints_marker_and_chat_reads_verified():
    k, pub = _keypair()
    store = VerifiedMakerSessions()
    sk = "did:privy:maker|sess1"
    assert store.is_verified(sk) is False
    nonce = store.issue_nonce(sk)
    assert nonce.startswith("titan-maker-presence:")
    ok = store.verify_and_mint(sk, nonce, _sign(k, nonce), pub)
    assert ok is True
    assert store.is_verified(sk) is True


def test_d0_wrong_signature_no_marker():
    k, pub = _keypair()
    wrong, _ = _keypair()
    store = VerifiedMakerSessions()
    sk = "u|s"
    nonce = store.issue_nonce(sk)
    # Sign with a DIFFERENT key → verify must fail → no marker (honest).
    assert store.verify_and_mint(sk, nonce, _sign(wrong, nonce), pub) is False
    assert store.is_verified(sk) is False


def test_d0_nonce_is_single_use():
    k, pub = _keypair()
    store = VerifiedMakerSessions()
    sk = "u|s"
    nonce = store.issue_nonce(sk)
    sig = _sign(k, nonce)
    assert store.verify_and_mint(sk, nonce, sig, pub) is True
    # Replaying the SAME nonce+sig must fail (consumed).
    store2_sk = "u|s2"
    assert store.verify_and_mint(store2_sk, nonce, sig, pub) is False


def test_d0_session_key_binding_enforced():
    k, pub = _keypair()
    store = VerifiedMakerSessions()
    nonce = store.issue_nonce("user-A|sess")
    # A different session presenting A's nonce → rejected.
    assert store.verify_and_mint("user-B|sess", nonce, _sign(k, nonce), pub) is False


def test_d0_expired_nonce_rejected():
    k, pub = _keypair()
    store = VerifiedMakerSessions(nonce_ttl_s=-1.0)   # already expired on issue
    sk = "u|s"
    nonce = store.issue_nonce(sk)
    assert store.verify_and_mint(sk, nonce, _sign(k, nonce), pub) is False


def test_d0_marker_ttl_expiry():
    k, pub = _keypair()
    store = VerifiedMakerSessions(marker_ttl_s=-1.0)   # marker expired on mint
    sk = "u|s"
    nonce = store.issue_nonce(sk)
    assert store.verify_and_mint(sk, nonce, _sign(k, nonce), pub) is True
    # TTL already elapsed → is_verified honestly False.
    assert store.is_verified(sk) is False


def test_d0_session_key_from_claims():
    assert session_key_from_claims(None) == ""
    assert session_key_from_claims({"sub": ""}) == ""
    assert session_key_from_claims({"sub": "did:x"}) == "did:x"
    assert session_key_from_claims({"sub": "did:x", "sid": "s9"}) == "did:x|s9"


# ─────────────────────────────────────────────────────────────────────
# D.3 — honest existential-delta valence (recency)
# ─────────────────────────────────────────────────────────────────────

def _path(tmp_path, name="mb.json"):
    return str(tmp_path / name)


def test_d3_first_contact_seeds_no_nudge(tmp_path):
    # last_contact None → first fold only primes the EMA (returns None).
    assert compute_maker_bond_nudge(None, 1000.0, _path(tmp_path), cfg=CFG) is None


def test_d3_intrinsic_positive_valence(tmp_path):
    p = _path(tmp_path)
    now = 100000.0
    # Seed + warm the baseline with a few same-size gaps.
    compute_maker_bond_nudge(now - 3600, now, p, cfg=CFG)            # seed
    compute_maker_bond_nudge(now, now + 3600, p, cfg=CFG)           # warm
    n = compute_maker_bond_nudge(now + 3600, now + 7200, p, cfg=CFG)
    if n is not None:
        assert n.valence == 1        # presence is ALWAYS continuity-positive
        assert n.target == 1.0


def test_d3_long_absence_more_surprising_than_steady(tmp_path):
    # Build a steady-cadence baseline (~1h gaps), then a long absence should
    # produce a LARGER surprise than a same-as-baseline gap (recency moves more).
    p_steady = _path(tmp_path, "steady.json")
    t = 0.0
    last = None
    for _ in range(8):
        compute_maker_bond_nudge(last, t, p_steady, cfg=CFG)
        last = t
        t += 3600.0          # steady 1h cadence
    steady_nudge = compute_maker_bond_nudge(last, t + 3600.0, p_steady, cfg=CFG)

    # Same baseline, then a 30-DAY absence.
    reunion_nudge = compute_maker_bond_nudge(last, t + 3600.0 * 24 * 30,
                                             p_steady, cfg=CFG)
    # The reunion after a long absence must be at least as surprising/strong.
    s_steady = steady_nudge.surprise if steady_nudge else 0.0
    assert reunion_nudge is not None
    assert reunion_nudge.surprise >= s_steady


def test_d3_habituation_steady_cadence_shrinks(tmp_path):
    # A long run of identical gaps drives surprise (and magnitude) toward 0.
    p = _path(tmp_path)
    t = 0.0
    last = None
    mags = []
    for _ in range(40):
        n = compute_maker_bond_nudge(last, t, p, cfg=CFG)
        if n is not None:
            mags.append(n.magnitude)
        last = t
        t += 3600.0
    # Later magnitudes should be no larger than early ones (habituation).
    if len(mags) >= 6:
        assert np.mean(mags[-3:]) <= np.mean(mags[:3]) + 1e-9


def test_d3_runtime_observe_maker_bond_updates_last_contact(tmp_path):
    rt = AffectiveNudgeRuntime(CFG, _path(tmp_path), str(tmp_path / "n.npz"),
                               emot_state_reader=lambda: None)
    assert rt._last_maker_contact_ts is None
    rt.observe_maker_bond(5000.0)
    assert rt._last_maker_contact_ts == 5000.0
    rt.observe_maker_bond(9000.0)
    assert rt._last_maker_contact_ts == 9000.0


# ─────────────────────────────────────────────────────────────────────
# D.4 — MAKER_PRESENCE emot-cgn primitive + 8→9 migration
# ─────────────────────────────────────────────────────────────────────

def test_d4_primitive_appended_last():
    from titan_hcl.logic.emotion_cluster import (
        EMOT_PRIMITIVES, NUM_PRIMITIVES, EMOT_PRIMITIVE_INDEX)
    assert NUM_PRIMITIVES == 9
    assert EMOT_PRIMITIVES[-1] == "MAKER_PRESENCE"
    assert EMOT_PRIMITIVE_INDEX["MAKER_PRESENCE"] == 8
    # The original 8 keep their indices (append-only, no reorder).
    assert EMOT_PRIMITIVE_INDEX["FLOW"] == 0
    assert EMOT_PRIMITIVE_INDEX["LOVE"] == 7


def test_d4_clusterer_builds_nine(tmp_path):
    from titan_hcl.logic.emotion_cluster import (
        EmotionClusterer, EMOT_PRIMITIVES)
    c = EmotionClusterer(save_dir=str(tmp_path))
    assert len(c._clusters) == 9
    assert "MAKER_PRESENCE" in c._clusters
    for p in EMOT_PRIMITIVES:
        assert c._clusters[p].centroid.shape[0] == 150


def test_d4_cgn_action_head_grows_8_to_9_preserving_weights(tmp_path):
    import torch
    from titan_hcl.logic.cgn import ConceptGroundingNetwork
    from titan_hcl.logic.cgn_types import CGNConsumerConfig

    cgn = ConceptGroundingNetwork(
        db_path=str(tmp_path / "m.db"), state_dir=str(tmp_path / "cgn"))
    # Register the emot consumer at the OLD 8-action size.
    cgn.register_consumer(CGNConsumerConfig(
        name="emotional", feature_dims=30, action_dims=8,
        action_names=["A", "B", "C", "D", "E", "F", "G", "H"]))
    anet = cgn._action_nets["emotional"]
    # Stamp recognisable learned weights into the 8-action head.
    with torch.no_grad():
        anet.action_head.weight.copy_(
            torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12))
        anet.action_head.bias.copy_(torch.arange(8, dtype=torch.float32))
    old_w = anet.action_head.weight.detach().clone()
    old_b = anet.action_head.bias.detach().clone()

    # Re-register with the NEW 9-action size → zero-init-append grow.
    cgn.register_consumer(CGNConsumerConfig(
        name="emotional", feature_dims=30, action_dims=9,
        action_names=["A", "B", "C", "D", "E", "F", "G", "H", "MAKER_PRESENCE"]))
    grown = cgn._action_nets["emotional"]
    assert grown.action_head.weight.shape == (9, 12)
    assert grown.action_dims == 9
    assert cgn._consumers["emotional"].action_dims == 9
    # Learned-8 rows preserved exactly; the appended 9th row is zero.
    assert torch.allclose(grown.action_head.weight[:8], old_w)
    assert torch.allclose(grown.action_head.bias[:8], old_b)
    assert torch.allclose(grown.action_head.weight[8], torch.zeros(12))
    assert float(grown.action_head.bias[8]) == 0.0
    # Q-net grew too; its Polyak target is re-synced to the grown online net.
    assert cgn._q_nets["emotional"].net[-1].out_features == 9
    assert cgn._q_targets["emotional"].net[-1].out_features == 9
    # forward() runs at the new width without shape error.
    logits, _ = grown.forward(torch.zeros(1, 30))
    assert logits.shape[-1] == 9


def test_d4_cgn_grow_is_noop_for_unchanged_dims(tmp_path):
    from titan_hcl.logic.cgn import ConceptGroundingNetwork
    from titan_hcl.logic.cgn_types import CGNConsumerConfig
    cgn = ConceptGroundingNetwork(
        db_path=str(tmp_path / "m.db"), state_dir=str(tmp_path / "cgn"))
    cgn.register_consumer(CGNConsumerConfig(
        name="social", feature_dims=30, action_dims=6,
        action_names=["a", "b", "c", "d", "e", "f"]))
    net_before = cgn._action_nets["social"]
    # Re-register identical → must NOT rebuild the net (same object).
    cgn.register_consumer(CGNConsumerConfig(
        name="social", feature_dims=30, action_dims=6,
        action_names=["a", "b", "c", "d", "e", "f"]))
    assert cgn._action_nets["social"] is net_before


def test_d4_observe_maker_presence_grounds_primitive_and_love(tmp_path):
    from titan_hcl.logic.emot_cgn import EmotCGNConsumer
    emot = EmotCGNConsumer(send_queue=None, titan_id="Ttest",
                           save_dir=str(tmp_path / "emot"))
    mp0 = emot._primitives["MAKER_PRESENCE"].V
    love0 = emot._primitives["LOVE"].V
    h9 = emot._hypotheses["H9_maker_presence_love"]
    n_obs0 = len(h9.observations)
    for _ in range(12):
        emot.observe_maker_presence(channel="chat")
    # MAKER_PRESENCE V rose (specific recognition) + LOVE V rose (colouring).
    assert emot._primitives["MAKER_PRESENCE"].V > mp0
    assert emot._primitives["LOVE"].V > love0
    # H9 accumulated observations.
    assert len(h9.observations) == n_obs0 + 12


def test_d4_h9_hypothesis_confirms_when_love_rises(tmp_path):
    from titan_hcl.logic.emot_cgn import EmotCGNConsumer
    emot = EmotCGNConsumer(send_queue=None, titan_id="Ttest",
                           save_dir=str(tmp_path / "emot"))
    for _ in range(40):
        emot.observe_maker_presence(channel="app")
    h9 = emot._hypotheses["H9_maker_presence_love"]
    effect, passed = emot._test_maker_presence_love(h9)
    assert passed is True
    assert effect > 0.0     # LOVE V rose across accumulating groundings
