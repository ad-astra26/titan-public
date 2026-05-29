#!/usr/bin/env python3
"""Analyze a synthesis soak run dir against the run-book's 7 questions.

Usage: python scripts/synthesis_soak_analyze.py titan-docs/sessions/synthesis_soak_<id>
"""
from __future__ import annotations

import json
import sys
import statistics
from collections import Counter, defaultdict


def _load(path):
    out = []
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    except FileNotFoundError:
        pass
    return out


def _ep(rec, path):
    """endpoint data dict (or {} if errored/missing)."""
    e = rec.get("endpoints", {}).get(path, {})
    if e.get("status") == 200 and isinstance(e.get("data"), dict):
        return e["data"]
    return {}


def _first_last(series):
    series = [s for s in series if s is not None]
    if not series:
        return None, None
    return series[0], series[-1]


def analyze(run_dir):
    print("=" * 78)
    print(f"  SYNTHESIS SOAK ANALYSIS — {run_dir}")
    print("=" * 78)

    chat = {t: _load(f"{run_dir}/chat_{t}.jsonl") for t in ("T2", "T3")}
    telem = {t: _load(f"{run_dir}/telemetry_{t}.jsonl") for t in ("T2", "T3")}
    rss = _load(f"{run_dir}/worker_rss.jsonl")

    # ── span ──
    all_chat = chat["T2"] + chat["T3"]
    print(f"\nTurns: T2={len(chat['T2'])} T3={len(chat['T3'])} total={len(all_chat)}")
    if telem["T2"]:
        print(f"Telemetry polls: T2={len(telem['T2'])} T3={len(telem['T3'])}; RSS samples={len(rss)}")

    # ── Q: chat outcomes (status/mode/latency/track/failures) ──
    print("\n── Chat outcomes ──────────────────────────────────────────")
    for t in ("T2", "T3"):
        rows = chat[t]
        if not rows:
            continue
        codes = Counter(r.get("status_code") for r in rows)
        modes = Counter(r.get("mode") for r in rows)
        lat = [r["latency_s"] for r in rows if isinstance(r.get("latency_s"), (int, float)) and r.get("status_code") == 200]
        tracks = Counter(r.get("track") for r in rows)
        ok = sum(1 for r in rows if r.get("status_code") == 200)
        print(f"  {t}: 200={ok}/{len(rows)} ({100*ok//max(len(rows),1)}%)  codes={dict(codes)}")
        print(f"      modes={dict(modes)}")
        if lat:
            lat.sort()
            print(f"      chat latency s: p50={lat[len(lat)//2]:.1f} p95={lat[int(len(lat)*0.95)]:.1f} max={max(lat):.1f} mean={statistics.mean(lat):.1f}")
        print(f"      tracks={dict(tracks)}")

    # ── Q1: sovereignty ratio ──
    print("\n── Q1: Sovereignty ratio (recall vs LLM re-derivation) ─────")
    for t in ("T2", "T3"):
        km, rs, cited, skilld, ratio = [], [], [], [], []
        for rec in telem[t]:
            sv = _ep(rec, "/v6/synthesis/metrics/sovereignty").get("sovereignty", {})
            w = (sv.get("windows") or {}).get("all", {})
            if w:
                km.append(w.get("knowledge_moments")); rs.append(w.get("recall_satisfied"))
                cited.append(w.get("cited_recalls")); skilld.append(w.get("skill_delegations"))
                ratio.append(w.get("ratio"))
        f, l = _first_last(km)
        print(f"  {t} (all-window): knowledge_moments {f}→{l}  recall_satisfied {_first_last(rs)[1]}  "
              f"cited {_first_last(cited)[1]}  skill_deleg {_first_last(skilld)[1]}  ratio {_first_last(ratio)[0]}→{_first_last(ratio)[1]}")

    # ── Q2: skill library ──
    print("\n── Q2: Skill library (compile / verify / delegate) ─────────")
    for t in ("T2", "T3"):
        size, ver, util = [], [], []
        for rec in telem[t]:
            sk = _ep(rec, "/v6/synthesis/skills")
            skills = sk.get("skills") if isinstance(sk.get("skills"), list) else None
            if skills is not None:
                size.append(len(skills))
                ver.append(sum(1 for s in skills if s.get("verified_at")))
                us = [s.get("utility_score", 0) for s in skills]
                util.append(round(statistics.mean(us), 3) if us else 0)
        print(f"  {t}: skills {_first_last(size)[0]}→{_first_last(size)[1]}  verified→{_first_last(ver)[1]}  mean_utility→{_first_last(util)[1]}")

    # ── Q3: concepts + forks ──
    print("\n── Q3: Concept spine + hypothesis forks ────────────────────")
    for t in ("T2", "T3"):
        ccount, fopen, fgrad, faband = [], [], [], []
        for rec in telem[t]:
            c = _ep(rec, "/v6/synthesis/concepts")
            cl = c.get("concepts") if isinstance(c.get("concepts"), list) else None
            if cl is not None:
                ccount.append(len(cl))
            f = _ep(rec, "/v6/synthesis/forks")
            summ = f.get("summary") if isinstance(f.get("summary"), dict) else f
            if isinstance(summ, dict) and ("open" in summ or "graduated" in summ):
                fopen.append(summ.get("open")); fgrad.append(summ.get("graduated")); faband.append(summ.get("abandoned"))
        print(f"  {t}: concepts {_first_last(ccount)[0]}→{_first_last(ccount)[1]}  "
              f"forks open→{_first_last(fopen)[1]} graduated→{_first_last(fgrad)[1]} abandoned→{_first_last(faband)[1]}")

    # ── Q4: oracle coverage ──
    print("\n── Q4: Oracle / scored_by coverage ─────────────────────────")
    for t in ("T2", "T3"):
        cov = []
        for rec in telem[t]:
            c = _ep(rec, "/v6/synthesis/oracles/coverage") or _ep(rec, "/v6/synthesis/skills/coverage")
            for k in ("coverage", "coverage_pct", "scored_pct", "pct"):
                if isinstance(c.get(k), (int, float)):
                    cov.append(c[k]); break
        print(f"  {t}: coverage {_first_last(cov)[0]}→{_first_last(cov)[1]}  (samples={len(cov)})")

    # ── Q5: retrieval p99 + chi ──
    print("\n── Q5: Retrieval latency (B.4) + chi (B.5) ─────────────────")
    for t in ("T2", "T3"):
        p99, chi = [], []
        for rec in telem[t]:
            r = _ep(rec, "/v6/synthesis/metrics/retrieval")
            ov = (r.get("retrieval") or {}).get("overall") if isinstance(r.get("retrieval"), dict) else None
            if isinstance(ov, dict) and ov.get("p99") is not None:
                p99.append(ov["p99"])
            ch = (r.get("chi") or {}) if isinstance(r.get("chi"), dict) else {}
            if ch.get("spent") is not None:
                chi.append(ch["spent"])
        print(f"  {t}: retrieval p99 {_first_last(p99)[0]}→{_first_last(p99)[1]} (max={max(p99) if p99 else None})  chi_spent→{_first_last(chi)[1]}")

    # ── Q6: chain growth ──
    print("\n── Q6: Chain growth (B.7 bounded?) ─────────────────────────")
    for t in ("T2", "T3"):
        tb = []
        for rec in telem[t]:
            c = _ep(rec, "/v6/synthesis/metrics/chain-growth").get("chain_growth", {})
            if isinstance(c, dict) and c.get("total_bytes") is not None:
                tb.append(c["total_bytes"])
        f, l = _first_last(tb)
        if f is not None:
            print(f"  {t}: chain bytes {f/1e6:.1f}MB→{l/1e6:.1f}MB  Δ={(l-f)/1e6:+.2f}MB over run")
        else:
            print(f"  {t}: chain-growth not available")

    # ── Q7: THE LIMIT — agno RSS + restarts + VPS load ──
    print("\n── Q7: agno RSS → restart cycle + VPS limit ────────────────")
    restarts = defaultdict(list)
    agno = defaultdict(list)
    for rec in rss:
        for ev in rec.get("restart_events", []):
            restarts[ev.get("target")].append(ev)
        for t in ("T2", "T3"):
            ti = rec.get("rss", {}).get(t, {})
            if isinstance(ti, dict) and ti.get("heaviest_rss_mb") is not None and ti.get("heaviest_is_agno"):
                agno[t].append(ti["heaviest_rss_mb"])
    for t in ("T2", "T3"):
        a = agno[t]
        if a:
            print(f"  {t} agno RSS MB: min={min(a):.0f} max={max(a):.0f} mean={statistics.mean(a):.0f}  "
                  f"restarts={len(restarts[t])}  (samples confirmed-agno={len(a)})")
        else:
            print(f"  {t}: no confirmed-agno samples (heaviest wasn't agno)")
    # VPS load/swap curve
    loads, swaps = [], []
    for rec in telem["T2"]:
        v = rec.get("vps_resource", {})
        la = v.get("loadavg")
        if la:
            try:
                loads.append(float(la[0]))
            except Exception:
                pass
        for ms in v.get("mem_swap_mb", []):
            if ms and ms[0] == "Swap:" and len(ms) >= 3:
                try:
                    swaps.append(int(ms[2]))
                except Exception:
                    pass
    if loads:
        print(f"  VPS loadavg(1m): min={min(loads):.1f} max={max(loads):.1f} mean={statistics.mean(loads):.1f} (4 cores)")
    if swaps:
        print(f"  VPS swap used MB: min={min(swaps)} max={max(swaps)} (of 16384)")
    # restart timeline
    for t in ("T2", "T3"):
        if restarts[t]:
            print(f"  {t} restart events:")
            for ev in restarts[t]:
                print(f"     pid {ev.get('old_pid')}→{ev.get('new_pid')}  rss {ev.get('old_rss_mb'):.0f}→{ev.get('new_rss_mb'):.0f}MB  agno={ev.get('is_agno')}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else ".")
