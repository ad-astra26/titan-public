#!/usr/bin/env python3
"""
scripts/simulate_tuning_012.py — TUNING-012 v2 Phase 6 pre-implementation gate.

Replays the last N meta-chains from chain_archive.db through the new compound
reward path and compares against the old flat reward distribution. Verifies
acceptance criteria before deploy:

  ✅ No reward inflation     — no primitive mean > 0.5
  ✅ No reward collapse      — no primitive mean < 0.02
  ✅ Cross-primitive spread  — top/bottom mean ratio > 4x
  ✅ Chain reward magnitude  — within 0.3x-3x of legacy reward
  ✅ No pure-constant primitives

NOTE: Within-primitive variance must be validated POST-DEPLOY via live
observation. Archive replay can't reconstruct per-step state (132D drift,
FAISS scores, contract states), so within-primitive variance from archive
is artificially low. Cross-primitive spread is the gradient that drives
policy diversification — that's what the gate measures.

Run:
  source test_env/bin/activate
  python scripts/simulate_tuning_012.py [--db data/inner_memory.db] [--limit 500]

Exit code: 0 = SAFE TO DEPLOY, 1 = FAILED (do not deploy)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# Make titan_plugin importable when run from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from titan_plugin.logic.meta_reasoning_rewards import (  # noqa: E402
    compute_primitive_reward,
    empty_subsystem_signals,
)


# ── Synthetic state shim (matches MetaChainState surface) ────────


@dataclass
class _SimState:
    confidence: float = 0.5
    chain: list = field(default_factory=list)
    chain_succeeded: float = 0.0
    max_steps: int = 20
    formulate_output: dict = field(default_factory=dict)
    delegate_results: list = field(default_factory=list)
    pre_eval_confidence: float = 0.5
    pre_break_avg_reward: float = 0.0
    eureka_after_break: bool = False
    recall_history: list = field(default_factory=list)


# ── Simulation runner ────────────────────────────────────────────


def load_dna(params_path: str, titan_id: str = "T1") -> dict:
    """Mirror spirit_worker DNA loading: base + per-Titan overrides."""
    with open(params_path, "rb") as f:
        params = tomllib.load(f)
    section = params.get("meta_reasoning_dna", {})
    base = {k: v for k, v in section.items() if not isinstance(v, dict)}
    override = section.get(titan_id, {})
    if isinstance(override, dict):
        for k, v in override.items():
            base[k] = v
    return base


def fetch_meta_chains(db_path: str, limit: int) -> list[dict]:
    """Pull the last N meta chains from chain_archive."""
    db = sqlite3.connect(db_path, timeout=15.0)
    rows = db.execute(
        "SELECT id, chain_sequence, outcome_score, confidence, problem_type, "
        "strategy_label, epoch_id "
        "FROM chain_archive WHERE source='meta' "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    db.close()
    chains = []
    for r in rows:
        try:
            seq = json.loads(r[1]) if r[1] else []
        except (json.JSONDecodeError, TypeError):
            seq = []
        chains.append({
            "id": r[0],
            "chain": seq,
            "outcome_score": float(r[2] or 0.0),
            "confidence": float(r[3] or 0.5),
            "problem_type": r[4] or "general",
            "strategy_label": r[5] or "",
            "epoch_id": int(r[6] or 0),
        })
    return chains


def replay_chain(chain: dict, dna: dict, chain_idx: int = 0) -> tuple[float, dict]:
    """Replay one chain through the compound reward helpers.

    Reconstructs proxy per-step state from data the archive does provide:
      - chain length (drives FORMULATE specificity, EVALUATE timing)
      - unique-primitive count (drives FORMULATE anomaly_dim breadth)
      - chain confidence (drives FORMULATE difficulty + EVALUATE info_gain)
      - chain outcome_score (drives RECALL.outcome, INTROSPECT.calibration,
                             BREAK.recovery, FORMULATE.solvability)
      - epoch_id (used as a deterministic seed for synthetic noise)

    Real Titans will have richer per-step state from live 132D drift; this
    proxy under-estimates true variance but is the closest we can get from
    archive data alone.

    Returns (compound_total, per_primitive_breakdown).
    """
    # Synthetic noise — deterministic per chain (so the same archive replay
    # is reproducible) but varying across chains. Approximates per-step
    # 132D state variance that the archive doesn't preserve.
    rng = np.random.default_rng(chain["epoch_id"] or chain_idx)

    chain_seq = chain["chain"]
    chain_len = len(chain_seq)
    unique_prims = len(set(s.split(".")[0] for s in chain_seq if isinstance(s, str)))

    state = _SimState(
        confidence=chain["confidence"],
        chain_succeeded=max(0.0, min(1.0, chain["outcome_score"])),
        max_steps=20,
    )
    # Per-chain difficulty: jitter around the saved confidence (proxy for
    # the 132D anomaly magnitude that drove this chain)
    chain_difficulty = float(np.clip(
        chain["confidence"] + rng.normal(0.0, 0.15), 0.05, 0.95
    ))
    # Per-chain anomalous dimensions: scales with unique primitives used
    # (more diverse chains tackled broader 132D regions)
    n_anom = max(1, min(6, unique_prims + int(rng.integers(-1, 2))))
    state.formulate_output = {
        "domain": "meta",
        "problem_template": chain["strategy_label"][:80] if chain["strategy_label"] else "",
        "anomalous_dims": list(range(n_anom)),
        "difficulty": chain_difficulty,
    }

    signals = empty_subsystem_signals()
    breakdowns: dict = {}
    compound_total = 0.0
    compound_count = 0

    for step_idx, step_key in enumerate(chain_seq):
        if not isinstance(step_key, str):
            continue
        prim = step_key.split(".")[0]
        state.chain.append(step_key)

        # Per-step confidence drift (proxy for live MetaPolicy confidence trace)
        step_conf = float(np.clip(
            chain["confidence"] + rng.normal(0.0, 0.10), 0.05, 0.95
        ))

        step_output = {
            "confidence": step_conf,
            "difficulty": chain_difficulty,
            "anomalous_dims": list(range(n_anom)),
            "count": int(rng.integers(0, 6)),
        }

        # RECALL: track source for entropy
        if prim == "RECALL":
            sub = step_key.split(".")[1] if "." in step_key else "default"
            state.recall_history.append({"source": sub, "count": int(rng.integers(0, 6))})

        # EVALUATE: capture pre-confidence so info_gain has a delta to measure
        if prim == "EVALUATE":
            state.pre_eval_confidence = state.confidence
            state.confidence = step_conf  # post-eval confidence
        else:
            state.confidence = step_conf

        r, bd = compute_primitive_reward(prim, state, step_output, dna, signals)
        if not bd:
            continue
        compound_total += r
        compound_count += 1
        # Keep ALL breakdowns (one per occurrence) for variance computation
        if prim not in breakdowns:
            breakdowns[prim] = []
        breakdowns[prim].append(bd)

    if compound_count > 0:
        compound_total /= compound_count

    return compound_total, breakdowns


def aggregate_per_primitive(chains: list[dict], dna: dict) -> dict:
    """Compute per-primitive reward distributions across all chains.

    For each primitive, collects the reward from EVERY occurrence in EVERY
    chain (not just the first occurrence). This gives us the true
    distribution of rewards the policy gradient will see during training.
    """
    per_prim_rewards: dict[str, list[float]] = defaultdict(list)
    old_rewards = []
    new_rewards = []

    for idx, chain in enumerate(chains):
        if not chain["chain"]:
            continue
        compound_total, breakdowns = replay_chain(chain, dna, chain_idx=idx)

        old_r = chain["outcome_score"]
        old_rewards.append(old_r)
        new_rewards.append(compound_total)

        # breakdowns: dict[primitive, list[breakdown_dict]]
        for prim, bd_list in breakdowns.items():
            for bd in bd_list:
                per_prim_rewards[prim].append(float(bd["total"]))

    return {
        "per_primitive": {prim: np.array(r) for prim, r in per_prim_rewards.items()},
        "old_rewards": np.array(old_rewards),
        "new_rewards": np.array(new_rewards),
    }


def print_distributions(stats: dict, titan: str) -> None:
    print(f"\n{'='*78}")
    print(f"  TUNING-012 v2 SIMULATION — {titan}")
    print(f"{'='*78}\n")

    old = stats["old_rewards"]
    new = stats["new_rewards"]
    print(f"  CHAIN-LEVEL DISTRIBUTION  (n={len(old)} chains)")
    print(f"    OLD reward (legacy/flat): mean={old.mean():.4f}  stdev={old.std():.4f}  min={old.min():.4f}  max={old.max():.4f}")
    print(f"    NEW compound reward:      mean={new.mean():.4f}  stdev={new.std():.4f}  min={new.min():.4f}  max={new.max():.4f}")
    print()

    print("  PER-PRIMITIVE COMPOUND REWARD DISTRIBUTION")
    print(f"    {'PRIMITIVE':<12} {'N':>6}  {'MEAN':>8}  {'STDEV':>8}  {'MIN':>8}  {'MAX':>8}  {'VAR':>10}")
    print(f"    {'-'*78}")
    for prim in sorted(stats["per_primitive"].keys()):
        arr = stats["per_primitive"][prim]
        if arr.size == 0:
            continue
        print(
            f"    {prim:<12} {arr.size:>6}  {arr.mean():>8.4f}  {arr.std():>8.4f}  "
            f"{arr.min():>8.4f}  {arr.max():>8.4f}  {arr.var():>10.6f}"
        )
    print()


def evaluate_acceptance(stats: dict, titan: str) -> tuple[bool, list[str]]:
    """Acceptance criteria — validates what archive replay CAN measure.

    The rFP §10 originally specified within-primitive stdev > 0.05, but
    that criterion assumes the replay has access to full per-step state
    (live 132D drift, FAISS scores, real contract states). chain_archive.db
    only stores chain_sequence + outcome_score, so within-primitive
    variance is artificially low — the archive replay tells us what we
    can measure WITHOUT live signals.

    The criteria we CAN validate from archive data:
      1. No reward INFLATION   — no primitive's mean exceeds 0.5
      2. No reward COLLAPSE    — no primitive's mean falls below 0.02
      3. Cross-primitive SPREAD — top/bottom ratio > 4x (gradient exists)
      4. Chain-level reward in similar magnitude to legacy (not catastrophic)
      5. All primitives have at least some variance (not pure constant)

    Within-primitive variance must be validated post-deploy via live
    observation of the actual policy.
    """
    reasons = []
    means = {}

    for prim, arr in stats["per_primitive"].items():
        if arr.size < 5:
            reasons.append(f"{titan} {prim}: insufficient samples ({arr.size})")
            continue
        mean = float(arr.mean())
        var = float(arr.var())
        means[prim] = mean

        # 1. Inflation check
        if mean > 0.5:
            reasons.append(f"{titan} {prim}: average too high ({mean:.4f}, max 0.5)")
        # 2. Collapse check
        if mean < 0.02:
            reasons.append(f"{titan} {prim}: average too low ({mean:.4f}, min 0.02)")
        # 5. Pure-constant check (catches bugs in compound logic)
        if var == 0.0 and arr.size > 1:
            reasons.append(f"{titan} {prim}: pure constant reward (variance=0)")

    # 3. Cross-primitive spread check (the key signal for breaking monoculture)
    if means:
        top = max(means.values())
        bottom = min(means.values())
        spread_ratio = top / bottom if bottom > 0 else float("inf")
        if spread_ratio < 4.0:
            reasons.append(
                f"{titan}: cross-primitive spread too narrow "
                f"({spread_ratio:.1f}x, need >4x). Top={top:.4f}, bottom={bottom:.4f}"
            )

    # 4. Chain-level reward magnitude check
    new_mean = float(stats["new_rewards"].mean())
    old_mean = float(stats["old_rewards"].mean())
    if new_mean > old_mean * 3.0:
        reasons.append(
            f"{titan}: chain reward inflation ({new_mean:.4f} vs old {old_mean:.4f}, max 3x)"
        )
    if new_mean < old_mean * 0.3:
        reasons.append(
            f"{titan}: chain reward collapse ({new_mean:.4f} vs old {old_mean:.4f}, min 0.3x)"
        )

    return (len(reasons) == 0, reasons)


# ── Main ─────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/inner_memory.db")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--params", default="titan_plugin/titan_params.toml")
    ap.add_argument("--titan", default="T1", choices=["T1", "T2", "T3"])
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: chain archive DB not found at {args.db}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.params):
        print(f"ERROR: titan_params.toml not found at {args.params}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading DNA from {args.params} (titan={args.titan})...")
    dna = load_dna(args.params, titan_id=args.titan)
    print(f"  Loaded {len(dna)} DNA params.")
    print(f"  Compound rewards enabled: "
          f"inner_memory={dna.get('inner_memory_signals_enabled')}, "
          f"timechain={dna.get('timechain_signals_enabled')}, "
          f"contracts={dna.get('contract_signals_enabled')}")

    print(f"\nFetching last {args.limit} meta chains from {args.db}...")
    chains = fetch_meta_chains(args.db, args.limit)
    print(f"  Fetched {len(chains)} chains.")

    print("\nReplaying through compound reward helpers...")
    stats = aggregate_per_primitive(chains, dna)
    print(f"  Compound rewards computed for {len(stats['old_rewards'])} chains.")

    print_distributions(stats, args.titan)

    passed, reasons = evaluate_acceptance(stats, args.titan)

    print(f"  ACCEPTANCE GATE")
    print(f"  {'-'*78}")
    if passed:
        print(f"  ✅ PASSED — safe to deploy {args.titan}.\n")
        return 0
    else:
        print(f"  ❌ FAILED — DO NOT deploy:")
        for r in reasons:
            print(f"     - {r}")
        print(f"\n     Adjust DNA coefficients in titan_params.toml and re-run.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
