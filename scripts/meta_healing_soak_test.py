#!/usr/bin/env python3
"""120-minute focused soak on meta-reasoning / META-CGN / meditation across T1+T2+T3.

Captures snapshot every 10 min for 120 min (13 checkpoints incl t=0).
Writes per-checkpoint log + JSON + final summary report.

Key metrics tracked:
  * Meta-reasoning: total_chains delta, primitive distribution, FORMULATE share,
    diversity_pressure fires, compound reward components, subsystem signals live
  * META-CGN: w_grounded, primitive V values + spread, disagreements_logged
  * Meditation: count in window, promoted count, F7 streak, zero-streak

Timeouts handled defensively — if an endpoint times out, degrades to file-based
read (meta_stats.json, primitive_grounding.json, blend_weights_history.jsonl) via
SSH for remote Titans.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Configuration ──────────────────────────────────────────────────────
DURATION_MIN = 120
CHECKPOINT_MIN = 10
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
LOG_PATH = Path(f"/tmp/meta_soak_{RUN_ID}.log")
REPORT_PATH = Path(f"/tmp/meta_soak_{RUN_ID}_FINAL.md")
SNAP_DIR = Path(f"/tmp/meta_soak_{RUN_ID}_snapshots")
SNAP_DIR.mkdir(exist_ok=True)

TITANS = [
    ("T1", "http://127.0.0.1:7777", "127.0.0.1", "/home/antigravity/projects/titan"),
    ("T2", "http://10.135.0.6:7777", "root@10.135.0.6", "/home/antigravity/projects/titan"),
    ("T3", "http://10.135.0.6:7778", "root@10.135.0.6", "/home/antigravity/projects/titan3"),
]


def ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    line = f"[{ts_utc()}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as fh:
        fh.write(line + "\n")


def http_get(url: str, timeout: float = 15.0, retries: int = 2) -> dict | None:
    """Try HTTP GET with retries. Returns parsed JSON or None on failure."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        if attempt < retries:
            time.sleep(1)
    return None


def read_local_file(path: str) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def ssh_read_file(host: str, remote_path: str) -> dict | None:
    """Read JSON file from remote host via SSH."""
    try:
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", host, f"cat {remote_path}"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0 and r.stdout:
            return json.loads(r.stdout)
    except Exception:
        pass
    return None


def ssh_tail_brain_log(host: str, titan_dir: str, pattern: str, n: int = 100) -> list[str]:
    """grep for pattern in brain log, return last N matching lines."""
    log_file = {
        "/home/antigravity/projects/titan": "/tmp/titan2_brain.log",
        "/home/antigravity/projects/titan3": "/tmp/titan3_brain.log",
    }.get(titan_dir, "/tmp/titan_brain.log")
    if host == "127.0.0.1":
        log_file = "/tmp/titan_brain.log"
    try:
        cmd_str = f"grep -E {pattern!r} {log_file} | tail -{n}"
        if host == "127.0.0.1":
            r = subprocess.run(["bash", "-c", cmd_str], capture_output=True, text=True, timeout=15)
        else:
            r = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", host, cmd_str],
                capture_output=True, text=True, timeout=20,
            )
        return [ln for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def capture_titan(label: str, url: str, host: str, titan_dir: str) -> dict:
    """Capture full snapshot for one Titan. Degrades gracefully on timeout."""
    snap = {"titan": label, "url": url, "ts": ts_utc(), "sources": {}}

    # 1) /v4/meta-reasoning (primary)
    mr = http_get(f"{url}/v4/meta-reasoning", timeout=20) or {}
    mr_data = mr.get("data", {}) if mr else {}
    if mr_data:
        snap["sources"]["meta_reasoning"] = "api"
        snap["meta"] = {
            "total_chains": mr_data.get("total_chains"),
            "total_steps": mr_data.get("total_steps"),
            "total_eurekas": mr_data.get("total_eurekas"),
            "total_wisdom_saved": mr_data.get("total_wisdom_saved"),
            "avg_reward": mr_data.get("avg_reward"),
            "baseline_confidence": mr_data.get("baseline_confidence"),
            "policy_updates": mr_data.get("policy_updates"),
            "primitive_counts": mr_data.get("primitive_counts", {}),
            "chain_length": mr_data.get("chain_length"),
            "meta_cgn_status": (mr_data.get("meta_cgn", {}) or {}).get("status"),
            "meta_cgn_grounded": (mr_data.get("meta_cgn", {}) or {}).get("primitives_grounded"),
            "chain_iql_updates": (mr_data.get("chain_iql", {}) or {}).get("total_updates"),
            "chain_iql_template_count": (mr_data.get("chain_iql", {}) or {}).get("template_count"),
            "chain_iql_lru_evictions": (mr_data.get("chain_iql", {}) or {}).get("lru_evictions"),
        }
    else:
        # Fallback: read meta_stats.json
        ms_path = f"{titan_dir}/data/reasoning/meta_stats.json"
        ms = read_local_file(ms_path) if host == "127.0.0.1" else ssh_read_file(host, ms_path)
        if ms:
            snap["sources"]["meta_reasoning"] = "file"
            snap["meta"] = {
                "total_chains": ms.get("total_chains"),
                "total_steps": ms.get("total_steps"),
                "total_eurekas": ms.get("total_eurekas"),
                "total_wisdom_saved": ms.get("total_wisdom_saved"),
                "baseline_confidence": ms.get("baseline_confidence"),
                "reroute_count": ms.get("reroute_count"),
                "diversity_pressure_target": ms.get("diversity_pressure_target"),
            }

    # 2) /v4/meta-reasoning/audit (signals + per-primitive reward)
    au = http_get(f"{url}/v4/meta-reasoning/audit", timeout=20) or {}
    au_data = au.get("data", {}) if au else {}
    if au_data and "status" not in au_data:
        snap["sources"]["audit"] = "api"
        snap["audit"] = {
            "diversity": au_data.get("diversity", {}),
            "monoculture": au_data.get("monoculture", {}),
            "diversity_pressure": au_data.get("diversity_pressure", {}),
            "rewards_per_primitive": au_data.get("rewards_per_primitive", {}),
            "contracts": au_data.get("contracts", {}),
            "subsystem_signals_status": au_data.get("subsystem_signals_status", {}),
            "introspect_health": au_data.get("introspect_health", {}),
        }

    # 3) primitive_grounding.json (V values, disagreements)
    pg_path = f"{titan_dir}/data/meta_cgn/primitive_grounding.json"
    pg = read_local_file(pg_path) if host == "127.0.0.1" else ssh_read_file(host, pg_path)
    if pg:
        snap["sources"]["grounding"] = "file"
        v_vals = [float(p.get("V", 0)) for p in (pg.get("primitives", {}) or {}).values()]
        snap["grounding"] = {
            "primitives_V": {k: v.get("V") for k, v in (pg.get("primitives", {}) or {}).items()},
            "V_spread": max(v_vals) - min(v_vals) if v_vals else 0,
            "V_mean": sum(v_vals) / len(v_vals) if v_vals else 0,
            "stats": pg.get("stats", {}),
        }

    # 4) blend_weights_history.jsonl (w_grounded — tail)
    bw_path = f"{titan_dir}/data/meta_cgn/blend_weights_history.jsonl"
    try:
        if host == "127.0.0.1":
            r = subprocess.run(["tail", "-3", bw_path], capture_output=True, text=True, timeout=10)
        else:
            r = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", host, f"tail -3 {bw_path}"],
                capture_output=True, text=True, timeout=20,
            )
        if r.returncode == 0 and r.stdout:
            last_line = r.stdout.strip().splitlines()[-1]
            j = json.loads(last_line)
            snap["blend_weights"] = {
                "w_legacy": j.get("w_legacy"),
                "w_compound": j.get("w_compound"),
                "w_grounded": j.get("w_grounded"),
                "status": j.get("status"),
                "beta_dispersion_ema": j.get("beta_dispersion_ema"),
                "latest_chain_id": j.get("chain_id"),
                "latest_domain": j.get("domain"),
            }
    except Exception:
        pass

    # 5) meditation — brain log greps
    med_promoted_lines = ssh_tail_brain_log(host, titan_dir, "MemoryWorker.*Meditation complete", 50)
    med_f7_lines = ssh_tail_brain_log(host, titan_dir, "F7_NOT_DISTILLING", 20)
    last_promoted_counts = []
    for ln in med_promoted_lines[-20:]:
        # parse "promoted=X pruned=Y fading=Z"
        try:
            import re
            m = re.search(r"promoted=(\d+)", ln)
            if m:
                last_promoted_counts.append(int(m.group(1)))
        except Exception:
            pass
    snap["meditation"] = {
        "meditations_in_log_window": len(med_promoted_lines),
        "last_20_promoted_counts": last_promoted_counts,
        "total_promotions_in_window": sum(last_promoted_counts),
        "nonzero_promotion_rate": (sum(1 for x in last_promoted_counts if x > 0) / len(last_promoted_counts)) if last_promoted_counts else None,
        "f7_alerts_in_window": len(med_f7_lines),
        "last_f7_alert": med_f7_lines[-1] if med_f7_lines else None,
    }

    return snap


def format_checkpoint_summary(t_min: int, snaps: list[dict]) -> str:
    """One-screen-friendly summary of the checkpoint."""
    lines = [f"\n{'=' * 78}", f"CHECKPOINT t={t_min:3d} min — {ts_utc()}", "=" * 78]
    for s in snaps:
        lines.append(f"\n  {s['titan']} ({s['url'].split('//')[1]})")
        meta = s.get("meta", {})
        audit = s.get("audit", {})
        ground = s.get("grounding", {})
        blend = s.get("blend_weights", {})
        med = s.get("meditation", {})

        if meta:
            pc = meta.get("primitive_counts", {}) or {}
            total_pc = sum(pc.values()) if pc else 0
            formul = pc.get("FORMULATE", 0)
            formul_pct = (formul / total_pc * 100) if total_pc else 0
            lines.append(f"    META: chains={meta.get('total_chains'):>6} "
                         f"steps={meta.get('total_steps'):>7} "
                         f"eureka={meta.get('total_eurekas'):>4} "
                         f"wisdom={meta.get('total_wisdom_saved'):>4}")
            lines.append(f"          FORMULATE={formul_pct:.0f}% "
                         f"chain_iql_updates={meta.get('chain_iql_updates')} "
                         f"lru_evict={meta.get('chain_iql_lru_evictions')}")
        else:
            lines.append("    META: unreachable (bus timeout or worker lag)")

        if audit:
            div = audit.get("diversity", {}) or {}
            mono = audit.get("monoculture", {}) or {}
            dp = audit.get("diversity_pressure", {}) or {}
            ss = audit.get("subsystem_signals_status", {}) or {}
            lines.append(f"    AUDIT: unique_ema={div.get('unique_prims_ema_50chains', 0):.2f} "
                         f"ε={div.get('current_epsilon', 0):.2f} "
                         f"dominant={mono.get('dominant_primitive','?')}@{mono.get('dominant_share_500', 0)*100:.0f}%")
            lines.append(f"           div_pressure={'ACTIVE' if dp.get('active') else 'idle'} "
                         f"lifetime_fires={dp.get('total_fires_lifetime', 0)} "
                         f"signals={ss.get('live_count', '?')}/{ss.get('total_signals', '?')} live")

        if blend:
            lines.append(f"    BLEND: w_grounded={blend.get('w_grounded', 0):.3f} "
                         f"w_compound={blend.get('w_compound', 0):.3f} "
                         f"w_legacy={blend.get('w_legacy', 0):.3f} "
                         f"β_disp={blend.get('beta_dispersion_ema', 0):.4f}")

        if ground:
            lines.append(f"    CGN-V: spread={ground.get('V_spread', 0):.4f} "
                         f"mean={ground.get('V_mean', 0):.4f} "
                         f"transitions={ground.get('stats', {}).get('transitions_sent')} "
                         f"disagreements={ground.get('stats', {}).get('disagreements_logged')}")

        if med:
            rate = med.get('nonzero_promotion_rate')
            rate_str = f"{rate*100:.0f}%" if rate is not None else "?"
            lines.append(f"    MEDIT: n_in_log={med.get('meditations_in_log_window')} "
                         f"promotions_total={med.get('total_promotions_in_window')} "
                         f"nonzero_rate={rate_str} "
                         f"F7_alerts={med.get('f7_alerts_in_window')}")
    return "\n".join(lines)


def format_final_report(all_snaps: list[list[dict]]) -> str:
    """Delta-focused final report."""
    first = all_snaps[0]
    last = all_snaps[-1]
    out = ["# Meta-Reasoning / META-CGN / Meditation Soak Test — Final Report",
           "",
           f"**Run ID:** `{RUN_ID}`",
           f"**Duration:** {(len(all_snaps) - 1) * CHECKPOINT_MIN} min (planned {DURATION_MIN} min)",
           f"**Checkpoints captured:** {len(all_snaps)}",
           f"**Start:** {first[0]['ts']}",
           f"**End:** {last[0]['ts']}",
           "",
           "---",
           "",
           "## Deltas per Titan (t=0 → t=end)",
           "",
           "| Titan | Δ chains | Δ eurekas | Δ wisdom | FORMULATE_share | Δ w_grounded | Δ V_spread | Δ disagreements | Δ LRU_evict |",
           "|---|---|---|---|---|---|---|---|---|"]

    for i, label in enumerate(["T1", "T2", "T3"]):
        s0 = first[i] if i < len(first) else {}
        s1 = last[i] if i < len(last) else {}
        m0, m1 = s0.get("meta", {}), s1.get("meta", {})
        a0, a1 = s0.get("audit", {}), s1.get("audit", {})
        g0, g1 = s0.get("grounding", {}), s1.get("grounding", {})
        b0, b1 = s0.get("blend_weights", {}), s1.get("blend_weights", {})

        def _delta(a, b, k):
            try: return (a.get(k) or 0) - (b.get(k) or 0)
            except Exception: return "?"

        def _share(m):
            pc = m.get("primitive_counts", {}) or {}
            tot = sum(pc.values())
            f = pc.get("FORMULATE", 0)
            return f"{(f/tot*100) if tot else 0:.0f}%"

        def _diff_stat(g_latest, g_early, k):
            try:
                return (g_latest.get("stats", {}) or {}).get(k, 0) - (g_early.get("stats", {}) or {}).get(k, 0)
            except Exception:
                return "?"

        out.append(f"| {label} "
                   f"| {_delta(m1, m0, 'total_chains')} "
                   f"| {_delta(m1, m0, 'total_eurekas')} "
                   f"| {_delta(m1, m0, 'total_wisdom_saved')} "
                   f"| {_share(m0)} → {_share(m1)} "
                   f"| {(b1.get('w_grounded', 0) or 0) - (b0.get('w_grounded', 0) or 0):+.4f} "
                   f"| {(g1.get('V_spread', 0) or 0) - (g0.get('V_spread', 0) or 0):+.4f} "
                   f"| {_diff_stat(g1, g0, 'disagreements_logged')} "
                   f"| {_delta(m1, m0, 'chain_iql_lru_evictions')} |")

    out.extend(["", "---", "", "## Meditation — activity during soak", "",
                "| Titan | meditations (sum) | promotions (sum) | nonzero_rate (end) | F7_alerts (end) |",
                "|---|---|---|---|---|"])
    for i, label in enumerate(["T1", "T2", "T3"]):
        meds = [cp[i].get("meditation", {}) for cp in all_snaps if i < len(cp)]
        total_med = sum(m.get("meditations_in_log_window", 0) or 0 for m in meds[-1:])
        total_prom = sum(m.get("total_promotions_in_window", 0) or 0 for m in meds[-1:])
        end_rate = meds[-1].get("nonzero_promotion_rate") if meds else None
        end_f7 = meds[-1].get("f7_alerts_in_window") if meds else 0
        rate_str = f"{end_rate*100:.0f}%" if end_rate is not None else "?"
        out.append(f"| {label} | {total_med} | {total_prom} | {rate_str} | {end_f7} |")

    out.extend(["", "---", "", "## Checkpoint-by-checkpoint trends", ""])
    for idx, cp in enumerate(all_snaps):
        t = idx * CHECKPOINT_MIN
        out.append(f"### t={t} min")
        for s in cp:
            m = s.get("meta", {})
            b = s.get("blend_weights", {})
            g = s.get("grounding", {})
            au = s.get("audit", {})
            mono = (au.get("monoculture") or {}) if au else {}
            dom = mono.get("dominant_primitive", "?")
            share = mono.get("dominant_share_500", 0) * 100
            out.append(f"- **{s['titan']}**: chains={m.get('total_chains','?')} "
                       f"dominant={dom}@{share:.0f}% "
                       f"w_grounded={b.get('w_grounded', 0) or 0:.3f} "
                       f"V_spread={g.get('V_spread', 0) or 0:.4f}")
        out.append("")

    return "\n".join(out)


def main():
    log(f"🚀 Soak test starting — RUN_ID={RUN_ID}")
    log(f"   Duration: {DURATION_MIN} min, checkpoint every {CHECKPOINT_MIN} min")
    log(f"   Log: {LOG_PATH}")
    log(f"   Snapshots: {SNAP_DIR}")
    log(f"   Final report: {REPORT_PATH}")

    n_checkpoints = (DURATION_MIN // CHECKPOINT_MIN) + 1
    all_snaps: list[list[dict]] = []
    start_mono = time.monotonic()

    for idx in range(n_checkpoints):
        t_min = idx * CHECKPOINT_MIN
        log(f"📸 Checkpoint {idx+1}/{n_checkpoints} at t={t_min} min")
        cp_snaps = []
        for label, url, host, titan_dir in TITANS:
            s = capture_titan(label, url, host, titan_dir)
            cp_snaps.append(s)
        all_snaps.append(cp_snaps)

        # Write per-checkpoint JSON
        snap_path = SNAP_DIR / f"t{t_min:03d}min.json"
        with snap_path.open("w") as fh:
            json.dump(cp_snaps, fh, indent=2, default=str)

        # Write checkpoint summary
        summary = format_checkpoint_summary(t_min, cp_snaps)
        log(summary)

        # Sleep until next checkpoint (skip if last)
        if idx < n_checkpoints - 1:
            target = start_mono + (idx + 1) * CHECKPOINT_MIN * 60
            to_sleep = max(0, target - time.monotonic())
            log(f"💤 Sleep {to_sleep:.0f}s until next checkpoint")
            time.sleep(to_sleep)

    # Final report
    report = format_final_report(all_snaps)
    REPORT_PATH.write_text(report)
    log(f"\n✅ Soak complete. Final report: {REPORT_PATH}")
    log(f"   All snapshots: {SNAP_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("⏹  Interrupted — partial report at REPORT_PATH if any")
        sys.exit(1)
    except Exception as e:
        log(f"❌ Soak test failed: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(2)
