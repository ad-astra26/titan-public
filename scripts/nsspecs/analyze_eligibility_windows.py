#!/usr/bin/env python3
"""
rFP β Stage 0.5 — Empirical eligibility-window analysis.

Derives principled per-program (K, decay, freshness_window_s) for the
Stage 2 neuromod reward observer, using REAL telemetry:
  - inner_memory.db program_fires (105k+ rows, 11 programs, timestamped)
  - inner_memory.db chain_archive (58k reasoning events with outcome_score)
  - inner_memory.db creative_works (3.4k ART/MUSIC events w/ trigger program)
  - inner_memory.db self_insights (2.7k self-reasoning events)
  - data/meta_cgn_emissions.jsonl (1.2k concept_grounded + balance_held events)

For each (program × proposed_reward_source) pair we compute:
  - Fire rate (events/hour) for the program
  - Δt distribution from nearest recent fire to reward event
  - Recommended (K=80th-pct fires covered, decay=exp half-life at K/2,
    freshness_window_s=90th-pct Δt)

Output: data/neural_nervous_system/eligibility_params.json
Loaded at boot by the Stage 2 NeuromodRewardObserver.

Usage:
  python scripts/nsspecs/analyze_eligibility_windows.py [--verbose]
"""
import json
import math
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INNER_MEMORY = PROJECT_ROOT / "data" / "inner_memory.db"
META_CGN_LOG = PROJECT_ROOT / "data" / "meta_cgn_emissions.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "neural_nervous_system" / "eligibility_params.json"

NS_PROGRAMS = [
    "REFLEX", "FOCUS", "INTUITION", "IMPULSE", "INSPIRATION",
    "CREATIVITY", "CURIOSITY", "EMPATHY", "REFLECTION",
    "METABOLISM", "VIGILANCE",
]

# Program scale class — informs fallback freshness window when no reward-source
# data exists. Based on rFP β design discussion 2026-04-16 (Q1 answer).
PROGRAM_SCALE = {
    # autonomic (tight window, fast credit)
    "REFLEX": "autonomic", "METABOLISM": "autonomic",
    "INTUITION": "autonomic", "IMPULSE": "autonomic",
    "VIGILANCE": "autonomic",
    # cognitive (medium window)
    "FOCUS": "cognitive", "INSPIRATION": "cognitive",
    "CURIOSITY": "cognitive",
    # personality (wider window, sparse)
    "CREATIVITY": "personality", "EMPATHY": "personality",
    "REFLECTION": "personality",
}

SCALE_DEFAULT_FRESHNESS_S = {
    "autonomic": 5.0,     # 5s — reflex timescale
    "cognitive": 60.0,    # 1min — reasoning chain timescale
    "personality": 600.0, # 10min — expression / social timescale
}


# ── Stats helpers ──────────────────────────────────────────────────


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(p / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[k]


def summarize(vals, label=""):
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {
        "n": len(s),
        "min": round(s[0], 3),
        "p25": round(percentile(s, 25), 3),
        "p50": round(percentile(s, 50), 3),
        "p75": round(percentile(s, 75), 3),
        "p90": round(percentile(s, 90), 3),
        "p95": round(percentile(s, 95), 3),
        "max": round(s[-1], 3),
        "mean": round(sum(s) / len(s), 3),
    }


# ── Data loaders ───────────────────────────────────────────────────


def load_program_fires(db_path: Path) -> dict:
    """Return {program: sorted_timestamps_list}."""
    c = sqlite3.connect(str(db_path))
    try:
        fires = {}
        for prog in NS_PROGRAMS:
            rows = c.execute(
                "SELECT timestamp FROM program_fires WHERE program=? ORDER BY timestamp",
                (prog,)).fetchall()
            fires[prog] = [r[0] for r in rows]
        return fires
    finally:
        c.close()


def load_reasoning_commits(db_path: Path, min_outcome: float = 0.5) -> list:
    """Return sorted list of (timestamp, outcome_score) for high-quality commits.

    chain_archive.created_at is ISO-ish string; we need to parse.
    """
    import datetime as dt
    c = sqlite3.connect(str(db_path))
    try:
        rows = c.execute(
            "SELECT created_at, outcome_score FROM chain_archive "
            "WHERE outcome_score >= ? ORDER BY created_at",
            (min_outcome,)).fetchall()
        out = []
        for ca, score in rows:
            try:
                # Try unix ts first, else ISO parse
                ts = float(ca)
            except (ValueError, TypeError):
                try:
                    ts = dt.datetime.fromisoformat(str(ca).replace("Z", "+00:00")).timestamp()
                except Exception:
                    continue
            out.append((ts, score))
        out.sort()
        return out
    finally:
        c.close()


def load_creative_works(db_path: Path, min_score: float = 0.0) -> dict:
    """Return {triggering_program: [timestamps]}."""
    c = sqlite3.connect(str(db_path))
    try:
        rows = c.execute(
            "SELECT timestamp, triggering_program, assessment_score "
            "FROM creative_works WHERE assessment_score >= ? "
            "ORDER BY timestamp", (min_score,)).fetchall()
        by_prog = {}
        for ts, prog, score in rows:
            if not ts or not prog:
                continue
            by_prog.setdefault(prog, []).append(float(ts))
        return by_prog
    finally:
        c.close()


def load_self_insights(db_path: Path) -> list:
    c = sqlite3.connect(str(db_path))
    try:
        rows = c.execute(
            "SELECT timestamp FROM self_insights WHERE timestamp IS NOT NULL "
            "ORDER BY timestamp").fetchall()
        return sorted(float(r[0]) for r in rows if r[0])
    finally:
        c.close()


def load_meta_cgn_events(log_path: Path) -> dict:
    """Return {(consumer, event_type): [timestamps]}."""
    if not log_path.exists():
        return {}
    out = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                key = (o.get("consumer", "?"), o.get("event_type", "?"))
                out.setdefault(key, []).append(float(o["ts"]))
            except Exception:
                continue
    for k in out:
        out[k].sort()
    return out


# ── Fire-interval analysis ────────────────────────────────────────


def analyze_fire_intervals(fires_by_program: dict) -> dict:
    """Per-program inter-fire interval stats. Gives baseline rate + typical gap."""
    result = {}
    for prog, ts_list in fires_by_program.items():
        if len(ts_list) < 2:
            result[prog] = {"status": "insufficient_fires", "n": len(ts_list)}
            continue
        intervals_s = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
        span_s = ts_list[-1] - ts_list[0]
        rate_per_hour = 3600.0 * len(ts_list) / max(span_s, 1.0)
        result[prog] = {
            "n_fires": len(ts_list),
            "span_hours": round(span_s / 3600.0, 2),
            "rate_per_hour": round(rate_per_hour, 3),
            "interval_s": summarize(intervals_s, "interval"),
        }
    return result


# ── Fire-to-reward Δt analysis ────────────────────────────────────


def fire_to_reward_deltas(fire_ts: list, reward_ts: list, max_window_s: float = 3600.0) -> list:
    """For each reward, find the most recent fire within max_window_s, return Δt.
    Returns [] if no matching fires exist; used to compute the distribution."""
    if not fire_ts or not reward_ts:
        return []
    import bisect
    deltas = []
    for r in reward_ts:
        # Find the most recent fire at or before r
        idx = bisect.bisect_right(fire_ts, r) - 1
        if idx < 0:
            continue
        dt = r - fire_ts[idx]
        if 0 <= dt <= max_window_s:
            deltas.append(dt)
    return deltas


def recommend_params(program: str, fire_stats: dict, delta_stats: dict | None) -> dict:
    """Combine fire-interval + Δt stats into final (K, decay, freshness_window_s).

    Logic:
    - freshness_window_s = 90th-pct Δt if we have it; else class default
    - K = how many fires typically fall within that window (derived from rate)
    - decay = 0.5 at K/2 (half-life scales with K)
    """
    scale = PROGRAM_SCALE.get(program, "cognitive")
    if delta_stats and delta_stats.get("n", 0) >= 20:
        # Use empirical Δt distribution
        window_s = float(delta_stats.get("p90", SCALE_DEFAULT_FRESHNESS_S[scale]))
        window_s = max(1.0, min(window_s, 3600.0))  # clamp [1s, 1h]
        source = "empirical_fire_to_reward"
    else:
        window_s = SCALE_DEFAULT_FRESHNESS_S[scale]
        source = "class_default"

    rate_per_hour = fire_stats.get("rate_per_hour", 1.0)
    fires_in_window = rate_per_hour * window_s / 3600.0
    K = max(1, min(20, math.ceil(fires_in_window * 1.25)))  # 80th pct coverage

    # Decay: want weight(0)=1.0, weight(K-1)=0.2 → decay^K ≈ 0.2 → decay = 0.2^(1/K)
    decay = 0.2 ** (1.0 / max(K, 1))
    decay = max(0.5, min(0.95, decay))

    return {
        "K": K,
        "decay": round(decay, 3),
        "freshness_window_s": round(window_s, 1),
        "source": source,
        "scale_class": scale,
        "rate_per_hour": round(rate_per_hour, 3),
    }


# ── Main ──────────────────────────────────────────────────────────


def main(verbose: bool = False):
    print("=" * 78)
    print("rFP β Stage 0.5 — Empirical Eligibility-Window Analysis")
    print("=" * 78)

    print(f"\nLoading program fires from {INNER_MEMORY.name}...")
    fires = load_program_fires(INNER_MEMORY)
    total_fires = sum(len(v) for v in fires.values())
    print(f"  Loaded {total_fires:,} fire events across {len(fires)} programs")

    print("\nLoading reward-source events...")
    commits = load_reasoning_commits(INNER_MEMORY, min_outcome=0.5)
    print(f"  chain_archive: {len(commits):,} commits with outcome_score >= 0.5")
    commits_high = [c for c in commits if c[1] >= 0.7]
    print(f"  chain_archive: {len(commits_high):,} commits with outcome_score >= 0.7 (eureka)")

    creative = load_creative_works(INNER_MEMORY, min_score=0.0)
    print(f"  creative_works: {sum(len(v) for v in creative.values()):,} events, "
          f"triggering programs: {list(creative.keys())}")

    insights = load_self_insights(INNER_MEMORY)
    print(f"  self_insights: {len(insights):,} events")

    meta_events = load_meta_cgn_events(META_CGN_LOG)
    concept_grounded_ts = []
    for (consumer, event_type), ts_list in meta_events.items():
        if event_type == "concept_grounded":
            concept_grounded_ts.extend(ts_list)
    concept_grounded_ts.sort()
    print(f"  meta_cgn_emissions: {len(concept_grounded_ts):,} concept_grounded events "
          f"(across {sum(1 for k in meta_events if k[1]=='concept_grounded')} consumers)")

    # ── Fire interval analysis ──
    print("\n\n1. PER-PROGRAM FIRE INTERVAL DISTRIBUTION")
    print("-" * 78)
    fire_stats = analyze_fire_intervals(fires)
    print(f"  {'PROGRAM':<14} {'N_FIRES':>8} {'RATE/H':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p90':>8}")
    for prog in NS_PROGRAMS:
        s = fire_stats.get(prog, {})
        if s.get("n_fires"):
            iv = s["interval_s"]
            print(f"  {prog:<14} {s['n_fires']:>8} {s['rate_per_hour']:>8.2f} "
                  f"{iv['p25']:>8.1f} {iv['p50']:>8.1f} {iv['p75']:>8.1f} {iv['p90']:>8.1f}")
        else:
            print(f"  {prog:<14} [insufficient data]")

    # ── Fire-to-reward Δt analysis per (program, source) ──
    print("\n\n2. FIRE → REWARD Δt DISTRIBUTIONS (per proposed pair)")
    print("-" * 78)

    # Proposed (program → source_ts_list) mappings from rFP β § 4a Option 4
    reward_sources = {
        "FOCUS":       ("reasoning.commit(>=0.5)", [c[0] for c in commits]),
        "INSPIRATION": ("reasoning.eureka(>=0.7)", [c[0] for c in commits_high]),
        "CURIOSITY":   ("cgn.concept_grounded",   concept_grounded_ts),
        "REFLECTION":  ("self_insights",          insights),
    }
    # Creative works: split by triggering_program
    for creative_prog, ts_list in creative.items():
        creative_prog_upper = creative_prog.upper() if creative_prog else ""
        if creative_prog_upper in {"CREATIVITY", "INSPIRATION", "EMPATHY"}:
            if creative_prog_upper not in reward_sources:
                reward_sources[creative_prog_upper] = (f"creative_works[{creative_prog}]", ts_list)

    # CREATIVITY baseline from any creative work (broader)
    all_creative_ts = sorted(ts for lst in creative.values() for ts in lst)
    if all_creative_ts and "CREATIVITY" not in reward_sources:
        reward_sources["CREATIVITY"] = ("creative_works(all)", all_creative_ts)

    deltas_by_program = {}
    for prog in NS_PROGRAMS:
        mapping = reward_sources.get(prog)
        if not mapping:
            deltas_by_program[prog] = None
            continue
        label, rw_ts = mapping
        deltas = fire_to_reward_deltas(fires.get(prog, []), rw_ts, max_window_s=3600.0)
        stats = summarize(deltas)
        deltas_by_program[prog] = {"source_label": label, "stats": stats}
        print(f"\n  {prog} ← {label}")
        print(f"    Δt samples: {stats.get('n', 0):,}  |  p25={stats.get('p25', 0):.1f}s  "
              f"p50={stats.get('p50', 0):.1f}s  p75={stats.get('p75', 0):.1f}s  "
              f"p90={stats.get('p90', 0):.1f}s  p95={stats.get('p95', 0):.1f}s")

    # ── Derive recommendations ──
    print("\n\n3. RECOMMENDED ELIGIBILITY PARAMETERS")
    print("-" * 78)
    params = {}
    print(f"  {'PROGRAM':<14} {'K':>4} {'DECAY':>7} {'WINDOW_S':>10} {'RATE/H':>8}  SOURCE")
    for prog in NS_PROGRAMS:
        fs = fire_stats.get(prog, {})
        ds = deltas_by_program.get(prog)
        delta_stats = ds["stats"] if ds else None
        rec = recommend_params(prog, fs, delta_stats)
        if ds:
            rec["reward_source_label"] = ds["source_label"]
        else:
            rec["reward_source_label"] = "TBD_no_data_yet"
        params[prog] = rec
        print(f"  {prog:<14} {rec['K']:>4} {rec['decay']:>7.3f} "
              f"{rec['freshness_window_s']:>10.1f} {rec['rate_per_hour']:>8.2f}  {rec['source']}")

    # ── Write output ──
    output = {
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "stage": "rFP β Stage 0.5 — empirical eligibility analysis",
        "source_files": {
            "inner_memory": str(INNER_MEMORY.relative_to(PROJECT_ROOT)),
            "meta_cgn_log": str(META_CGN_LOG.relative_to(PROJECT_ROOT)),
        },
        "summary_counts": {
            "total_fires": total_fires,
            "reasoning_commits_gte_0.5": len(commits),
            "reasoning_eurekas_gte_0.7": len(commits_high),
            "creative_works": sum(len(v) for v in creative.values()),
            "self_insights": len(insights),
            "concept_grounded_events": len(concept_grounded_ts),
        },
        "fire_stats": fire_stats,
        "reward_delta_stats": {
            prog: (deltas_by_program[prog] if deltas_by_program.get(prog) else None)
            for prog in NS_PROGRAMS
        },
        "params": params,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n\nWritten: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Loaded by Stage 2 NeuromodRewardObserver at boot.")
    print("=" * 78)


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    main(verbose=verbose)
