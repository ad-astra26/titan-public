"""
Titan Telemetry Collector — Scientific Data Capture.

Polls all Titan instances (T1, T2, T3) every 30s, capturing full state
including meta-reasoning, reasoning, and language data for comparative analysis.

Produces:
  data/twin_telemetry_YYYYMMDD_HHMM.json — structured snapshots
  titan-docs/reports/REPORT_twin_experiment_YYYYMMDD_HHMM.md — markdown report

Usage:
    source test_env/bin/activate
    python scripts/twin_telemetry.py --duration 240  # 4 hours
"""
import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime

import httpx

TITAN1_API = "http://localhost:7777"
TITAN2_API = "http://10.135.0.6:7777"
TITAN3_API = "http://10.135.0.6:7778"
INTERNAL_KEY = os.environ.get("TITAN_INTERNAL_KEY", "")
SNAPSHOT_INTERVAL = 30  # seconds

log = logging.getLogger("twin_telemetry")


class TwinTelemetry:
    """Captures parallel telemetry from both Titan instances."""

    def __init__(self):
        self.start_time = time.time()
        self.snapshots = []  # List of {ts, titan1: {...}, titan2: {...}, titan3: {...}}

    @staticmethod
    def _read_phase_status() -> dict:
        """Read current pipeline phase from status file."""
        try:
            status_path = "./data/phase_status.json"
            if os.path.exists(status_path):
                with open(status_path) as f:
                    return json.load(f)
        except Exception:
            pass
        return {"phase": 0, "subphase": "idle"}

    async def capture_snapshot(self, client: httpx.AsyncClient) -> dict:
        """Capture one snapshot from all instances (parallel)."""
        t1, t2, t3 = await asyncio.gather(
            self._capture_instance(client, TITAN1_API),
            self._capture_instance(client, TITAN2_API),
            self._capture_instance(client, TITAN3_API),
            return_exceptions=True,
        )
        snapshot = {
            "timestamp": time.time(),
            "elapsed_s": round(time.time() - self.start_time, 1),
            "titan1": t1 if isinstance(t1, dict) else {"error": str(t1)},
            "titan2": t2 if isinstance(t2, dict) else {"error": str(t2)},
            "titan3": t3 if isinstance(t3, dict) else {"error": str(t3)},
            "phase_status": self._read_phase_status(),
        }
        self.snapshots.append(snapshot)
        return snapshot

    async def _capture_instance(self, client: httpx.AsyncClient, api: str) -> dict:
        """Capture full state from one Titan instance."""
        data = {}

        # 1. Hormonal system (all 10 hormones)
        try:
            r = await client.get(f"{api}/v4/nervous-system", timeout=10)
            ns = r.json().get("data", {})
            hormones = {}
            for name, h in ns.get("hormonal_system", {}).items():
                if isinstance(h, dict):
                    hormones[name] = {
                        "level": round(h.get("level", 0), 4),
                        "fire_count": h.get("fire_count", 0),
                        "threshold": round(h.get("threshold", 0), 4),
                    }
            data["hormones"] = hormones
            data["training_phase"] = ns.get("training_phase", "?")
            data["total_transitions"] = ns.get("total_transitions", 0)
        except Exception as e:
            data["hormones_error"] = str(e)

        # 2. Consciousness state + π-heartbeat
        try:
            r = await client.get(f"{api}/v4/pi-heartbeat", timeout=10)
            pi = r.json().get("data", {})
            data["pi_heartbeat"] = {
                "in_cluster": pi.get("in_cluster"),
                "pi_streak": pi.get("current_pi_streak", 0),
                "zero_streak": pi.get("current_zero_streak", 0),
                "cluster_count": pi.get("cluster_count", 0),
                "developmental_age": pi.get("developmental_age", 0),
                "heartbeat_ratio": round(pi.get("heartbeat_ratio", 0), 4),
                "total_epochs": pi.get("total_epochs_observed", 0),
            }
        except Exception as e:
            data["pi_error"] = str(e)

        # 3. Full Titan state (132D vector + backend-computed coherences)
        try:
            r = await client.get(f"{api}/v4/state", timeout=10)
            cons = r.json().get("data", {}).get("consciousness", {})
            data["state_vector_dim"] = len(cons.get("state_vector", []))
            if "body_coherence" in cons:
                data["inner_body_coh"] = round(cons.get("body_coherence", 0), 4)
                data["inner_mind_coh"] = round(cons.get("mind_coherence", 0), 4)
                data["inner_spirit_coh"] = round(cons.get("spirit_coherence", 0), 4)
            if "outer_body_coherence" in cons:
                data["outer_body_coh"] = round(cons.get("outer_body_coherence", 0), 4)
                data["outer_mind_coh"] = round(cons.get("outer_mind_coherence", 0), 4)
                data["outer_spirit_coh"] = round(cons.get("outer_spirit_coherence", 0), 4)
        except Exception as e:
            data["state_error"] = str(e)

        # 4. Sphere clocks
        try:
            r = await client.get(f"{api}/v4/sphere-clocks", timeout=10)
            clocks = r.json().get("data", {})
            clock_summary = {}
            for name, c in clocks.items():
                if isinstance(c, dict):
                    clock_summary[name] = {
                        "radius": round(c.get("radius", 1.0), 4),
                        "pulse_count": c.get("pulse_count", 0),
                        "balance_streak": c.get("balance_streak", 0),
                    }
            data["sphere_clocks"] = clock_summary
        except Exception:
            pass

        # 5. Consciousness epoch
        try:
            r = await client.get(f"{api}/status/consciousness/history?limit=1", timeout=10)
            hist = r.json()
            if isinstance(hist, dict) and hist.get("data"):
                epochs = hist["data"]
                if epochs:
                    latest = epochs[0] if isinstance(epochs, list) else epochs
                    data["consciousness"] = {
                        "epoch_id": latest.get("epoch_id", 0),
                        "curvature": round(latest.get("curvature", 0), 4),
                        "density": round(latest.get("density", 0), 4),
                        "drift": round(latest.get("drift_magnitude", 0), 4),
                    }
        except Exception:
            pass

        # 6. Neuromodulator state + Expression composites + Chi (via inner-trinity)
        try:
            r = await client.get(f"{api}/v4/inner-trinity", timeout=10)
            coord = r.json().get("data", {})
            # Neuromodulators
            nm = coord.get("neuromodulators", {})
            if nm:
                data["neuromodulators"] = {
                    "emotion": nm.get("current_emotion", "?"),
                    "emotion_confidence": round(nm.get("emotion_confidence", 0), 4),
                    "total_evaluations": nm.get("total_evaluations", 0),
                    "modulators": {
                        name: round(m.get("level", 0), 4)
                        for name, m in nm.get("modulators", {}).items()
                    },
                }
                mod = nm.get("modulation", {})
                if mod:
                    data["neuromodulator_modulation"] = {
                        k: round(v, 4) for k, v in mod.items()
                    }
            # Expression composites
            expr = coord.get("expression_composites", {})
            if expr:
                composites = expr.get("composites", expr)
                data["expression_composites"] = {
                    name: {
                        "fire_count": c.get("fire_count", 0),
                        "threshold": round(c.get("threshold", 0), 4),
                        "last_urge": round(c.get("last_urge", 0), 4),
                        "peak_urge": round(c.get("peak_urge", 0), 4),
                    }
                    for name, c in composites.items()
                    if isinstance(c, dict)
                }
            # Working memory
            wm = coord.get("working_memory", {})
            if wm:
                data["working_memory"] = {
                    "size": wm.get("size", 0),
                    "capacity": wm.get("capacity", 7),
                }
            # Prediction engine
            pred = coord.get("prediction", {})
            if pred:
                data["prediction"] = {
                    "novelty": pred.get("novelty", 0),
                    "familiarity": pred.get("familiarity", 0),
                }
            # Topology
            topo = coord.get("topology", {})
            if topo:
                data["topology"] = {
                    "curvature": round(topo.get("curvature", 0), 4),
                    "coherence": round(topo.get("coherence", 0), 4),
                    "density": round(topo.get("density", 0), 4),
                }
            # Ground-up enrichment
            gu = coord.get("ground_up", {})
            if gu:
                data["ground_up"] = {
                    "total_applications": gu.get("total_applications", 0),
                    "strength": round(gu.get("strength", 0), 4),
                }
        except Exception as e:
            data["coordinator_error"] = str(e)

        # 7. Chi Life Force
        try:
            r = await client.get(f"{api}/v4/chi", timeout=10)
            chi = r.json().get("data", {})
            if chi and "total" in chi:
                data["chi"] = {
                    "total": round(chi.get("total", 0), 4),
                    "spirit": round(chi.get("spirit", {}).get("effective", 0), 4),
                    "mind": round(chi.get("mind", {}).get("effective", 0), 4),
                    "body": round(chi.get("body", {}).get("effective", 0), 4),
                    "circulation": round(chi.get("circulation", 0), 4),
                    "state": chi.get("state", "?"),
                    "phase": chi.get("developmental_phase", "?"),
                }
        except Exception:
            pass

        # 8. Self-Exploration (Outer Interface)
        try:
            r = await client.get(f"{api}/v4/self-exploration", timeout=10)
            se = r.json().get("data", {})
            if se and "mode" in se:
                data["self_exploration"] = {
                    "mode": se.get("mode", "?"),
                    "actions_processed": se.get("total_actions_processed", 0),
                    "actions_queued": se.get("total_actions_queued", 0),
                    "words_reinforced": se.get("total_words_reinforced", 0),
                    "words_unknown": se.get("total_words_unknown", 0),
                    "explore_ticks": se.get("total_explore_ticks", 0),
                    "advisor_explorations": se.get("advisor", {}).get("total_explorations", 0),
                    "advisor_blocked": se.get("advisor", {}).get("total_blocked", 0),
                }
        except Exception:
            pass

        # 9. Phase Events + Dreaming (from coordinator stats via /v4/inner-trinity)
        try:
            r = await client.get(f"{api}/v4/inner-trinity", timeout=10)
            coord = r.json()
            # Phase events
            pe = coord.get("phase_events", {})
            if pe:
                data["phase"] = {
                    "current": pe.get("current_phase", "?"),
                    "total_events": pe.get("total_events", 0),
                    "recent": pe.get("recent_events", [])[-5:],  # last 5 events
                }
            # Dreaming state
            dr = coord.get("dreaming", {})
            if dr:
                data["dreaming"] = {
                    "is_dreaming": dr.get("is_dreaming", False),
                    "fatigue": round(dr.get("fatigue", 0), 4),
                    "readiness": round(dr.get("readiness", 0), 4),
                    "cycle_count": dr.get("cycle_count", 0),
                }
        except Exception:
            pass

        # 10. Vocabulary + Composition progress (Phase 2+3 language learning)
        try:
            r = await client.get(f"{api}/v4/vocabulary", timeout=10)
            vocab = r.json().get("data", r.json())
            words = vocab.get("words", [])
            if words:
                confs = [w.get("confidence", 0) for w in words if isinstance(w, dict)]
                types = {}
                for w in words:
                    if isinstance(w, dict):
                        t = w.get("word_type", "unknown")
                        types[t] = types.get(t, 0) + 1
                data["vocabulary"] = {
                    "total_words": len(words),
                    "avg_confidence": round(sum(confs) / len(confs), 4) if confs else 0,
                    "high_conf_words": sum(1 for c in confs if c >= 0.7),
                    "low_conf_words": sum(1 for c in confs if c < 0.3),
                    "word_types": types,
                }
        except Exception:
            pass

        # 11. Dreaming detailed fields (recovery, wake transition)
        try:
            r = await client.get(f"{api}/v4/dream-inbox", timeout=10)
            di = r.json().get("data", {})
            ds = di.get("dream_state", {})
            if ds:
                data["dream_detail"] = {
                    "is_dreaming": ds.get("is_dreaming", False),
                    "recovery_pct": round(ds.get("recovery_pct", 0), 4),
                    "wake_transition": ds.get("wake_transition", False),
                    "remaining_epochs": ds.get("remaining_epochs", 0),
                    "inbox_count": di.get("inbox_count", 0),
                }
        except Exception:
            pass

        # 12. Reasoning stats (via /v4/reasoning)
        try:
            r = await client.get(f"{api}/v4/reasoning", timeout=10)
            rs = r.json().get("data", r.json())
            if rs and not rs.get("error"):
                data["reasoning"] = {
                    "total_chains": rs.get("total_chains", 0),
                    "commits": rs.get("commits", 0),
                    "abandons": rs.get("abandons", 0),
                    "commit_rate": round(rs.get("commit_rate", 0), 4),
                    "avg_chain_length": round(rs.get("avg_chain_length", 0), 2),
                    "current_chain_active": rs.get("current_chain_active", False),
                }
        except Exception:
            pass

        # 13. Meta-reasoning stats (via /v4/meta-reasoning)
        try:
            r = await client.get(f"{api}/v4/meta-reasoning", timeout=10)
            mr = r.json().get("data", r.json())
            if mr and not mr.get("error"):
                data["meta_reasoning"] = {
                    "total_chains": mr.get("total_chains", 0),
                    "total_steps": mr.get("total_steps", 0),
                    "wisdom_count": mr.get("wisdom_count", 0),
                    "archive_count": mr.get("archive_count", 0),
                    "avg_reward": round(mr.get("avg_reward", 0), 4),
                    "primitive_counts": mr.get("primitive_counts", {}),
                    "is_active": mr.get("is_active", False),
                    "autoencoder_trained": mr.get("autoencoder_trained", False),
                }
        except Exception:
            pass

        # 14. Language compositions (via /v4/compositions)
        try:
            r = await client.get(f"{api}/v4/compositions?limit=1", timeout=10)
            comp = r.json().get("data", r.json())
            if comp and not comp.get("error"):
                data["compositions"] = {
                    "total": comp.get("total_compositions", 0),
                    "latest": comp.get("latest", {}).get("sentence", ""),
                    "latest_level": comp.get("latest", {}).get("level", 0),
                    "latest_confidence": round(comp.get("latest", {}).get("confidence", 0), 4),
                }
        except Exception:
            pass

        return data

    def save(self, path: str):
        """Save all snapshots to JSON + prune older snapshots to bound disk.

        Retention: keep TELEMETRY_RETENTION most recent twin_telemetry files
        in the same directory. Old ones are deleted to prevent the unbounded
        growth that filled T2/T3 disks to 100% on 2026-04-14 (combined with
        the autocommit/pull loop, which was the original hot path).
        """
        with open(path, "w") as f:
            json.dump({
                "experiment": "titan_telemetry_v2",
                "start_time": self.start_time,
                "start_iso": datetime.fromtimestamp(self.start_time).isoformat(),
                "total_snapshots": len(self.snapshots),
                "snapshot_interval_s": SNAPSHOT_INTERVAL,
                "titan1_api": TITAN1_API,
                "titan2_api": TITAN2_API,
                "titan3_api": TITAN3_API,
                "snapshots": self.snapshots,
            }, f, indent=2)
        _prune_telemetry_retention(path)


TELEMETRY_RETENTION = 500


def _prune_telemetry_retention(just_written_path: str) -> None:
    """Keep only the most recent TELEMETRY_RETENTION twin_telemetry_*.json
    files. Non-fatal on errors — save must not fail because prune did."""
    import glob
    try:
        directory = os.path.dirname(just_written_path) or "."
        pattern = os.path.join(directory, "twin_telemetry_*.json")
        files = glob.glob(pattern)
        if len(files) <= TELEMETRY_RETENTION:
            return
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for stale in files[TELEMETRY_RETENTION:]:
            try:
                os.unlink(stale)
            except OSError:
                pass
    except Exception as e:
        log.debug("telemetry prune non-fatal: %s", e)

    def generate_report(self) -> str:
        """Generate comparative markdown report."""
        lines = [
            "# Twin Experiment Telemetry Report",
            f"## {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"**Duration:** {round((time.time() - self.start_time) / 3600, 2)} hours",
            f"**Snapshots:** {len(self.snapshots)}",
            f"**Interval:** {SNAPSHOT_INTERVAL}s",
            "",
            "---",
            "",
            "## Hormonal Comparison (Final State)",
            "",
            "| Hormone | Titan1 Level | Titan1 Fires | Titan2 Level | Titan2 Fires | Delta |",
            "|---------|-------------|-------------|-------------|-------------|-------|",
        ]

        if self.snapshots:
            last = self.snapshots[-1]
            t1h = last.get("titan1", {}).get("hormones", {})
            t2h = last.get("titan2", {}).get("hormones", {})
            all_hormones = sorted(set(list(t1h.keys()) + list(t2h.keys())))
            for h in all_hormones:
                t1 = t1h.get(h, {})
                t2 = t2h.get(h, {})
                t1l = t1.get("level", 0)
                t2l = t2.get("level", 0)
                delta = t1l - t2l
                lines.append(
                    f"| {h} | {t1l:.3f} | {t1.get('fire_count', 0)} | "
                    f"{t2l:.3f} | {t2.get('fire_count', 0)} | {delta:+.3f} |")

        lines.extend(["", "---", "", "## π-Heartbeat Comparison", ""])

        if self.snapshots:
            last = self.snapshots[-1]
            t1p = last.get("titan1", {}).get("pi_heartbeat", {})
            t2p = last.get("titan2", {}).get("pi_heartbeat", {})
            lines.append(f"| Metric | Titan1 | Titan2 |")
            lines.append(f"|--------|--------|--------|")
            for key in ["cluster_count", "developmental_age", "heartbeat_ratio", "total_epochs"]:
                lines.append(f"| {key} | {t1p.get(key, '?')} | {t2p.get(key, '?')} |")

        # Neuromodulator comparison
        lines.extend(["", "---", "", "## Neuromodulator Comparison (Final State)", ""])
        if self.snapshots:
            last = self.snapshots[-1]
            t1n = last.get("titan1", {}).get("neuromodulators", {})
            t2n = last.get("titan2", {}).get("neuromodulators", {})
            lines.append(f"| | Titan1 | Titan2 |")
            lines.append(f"|---|--------|--------|")
            lines.append(f"| Emotion | {t1n.get('emotion', '?')} ({t1n.get('emotion_confidence', 0):.0%}) | {t2n.get('emotion', '?')} ({t2n.get('emotion_confidence', 0):.0%}) |")
            t1m = t1n.get("modulators", {})
            t2m = t2n.get("modulators", {})
            for mod_name in ["DA", "5HT", "NE", "ACh", "Endorphin", "GABA"]:
                lines.append(f"| {mod_name} | {t1m.get(mod_name, '?')} | {t2m.get(mod_name, '?')} |")

        # Expression composites comparison
        lines.extend(["", "---", "", "## Expression Composites (Final State)", ""])
        if self.snapshots:
            last = self.snapshots[-1]
            t1e = last.get("titan1", {}).get("expression_composites", {})
            t2e = last.get("titan2", {}).get("expression_composites", {})
            if t1e or t2e:
                lines.append(f"| Composite | T1 Fires | T1 Peak Urge | T2 Fires | T2 Peak Urge |")
                lines.append(f"|-----------|----------|-------------|----------|-------------|")
                for comp in ["SPEAK", "ART", "MUSIC", "SOCIAL"]:
                    t1c = t1e.get(comp, {})
                    t2c = t2e.get(comp, {})
                    lines.append(f"| {comp} | {t1c.get('fire_count', 0)} | {t1c.get('peak_urge', 0):.3f} | {t2c.get('fire_count', 0)} | {t2c.get('peak_urge', 0):.3f} |")

        # Chi Life Force comparison
        lines.extend(["", "---", "", "## Chi Life Force (Final State)", ""])
        if self.snapshots:
            last = self.snapshots[-1]
            t1c = last.get("titan1", {}).get("chi", {})
            t2c = last.get("titan2", {}).get("chi", {})
            if t1c or t2c:
                lines.append(f"| Metric | Titan1 | Titan2 |")
                lines.append(f"|--------|--------|--------|")
                for key in ["total", "spirit", "mind", "body", "circulation", "state", "phase"]:
                    lines.append(f"| {key} | {t1c.get(key, '?')} | {t2c.get(key, '?')} |")

        # Self-Exploration comparison
        lines.extend(["", "---", "", "## Self-Exploration (Final State)", ""])
        if self.snapshots:
            last = self.snapshots[-1]
            t1se = last.get("titan1", {}).get("self_exploration", {})
            t2se = last.get("titan2", {}).get("self_exploration", {})
            if t1se or t2se:
                lines.append("| Metric | Titan1 | Titan2 |")
                lines.append("|--------|--------|--------|")
                for key in ["mode", "actions_processed", "actions_queued",
                            "words_reinforced", "words_unknown",
                            "explore_ticks", "advisor_explorations"]:
                    lines.append(f"| {key} | {t1se.get(key, '?')} | {t2se.get(key, '?')} |")

        lines.extend(["", "---", "",
                       "## Coherence Timeline (sampled)", ""])

        # Sample every 10th snapshot for timeline
        for i, snap in enumerate(self.snapshots):
            if i % 10 != 0 and i != len(self.snapshots) - 1:
                continue
            t1 = snap.get("titan1", {})
            t2 = snap.get("titan2", {})
            elapsed = round(snap.get("elapsed_s", 0) / 60, 1)
            t1_is = t1.get("inner_spirit_coh", "?")
            t2_is = t2.get("inner_spirit_coh", "?")
            lines.append(f"- **{elapsed}min** — T1 spirit={t1_is} | T2 spirit={t2_is}")

        return "\n".join(lines)


async def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    duration_s = args.duration * 60
    telemetry = TwinTelemetry()

    log.info("Twin Telemetry started — capturing every %ds for %d minutes",
             SNAPSHOT_INTERVAL, args.duration)

    async with httpx.AsyncClient() as client:
        start = time.time()
        snapshot_num = 0

        while time.time() - start < duration_s:
            snapshot_num += 1
            try:
                snap = await telemetry.capture_snapshot(client)
                parts = []
                for label in ["titan1", "titan2", "titan3"]:
                    td = snap.get(label, {})
                    if td.get("error"):
                        parts.append(f"{label[-1]}:DOWN")
                        continue
                    em = td.get("neuromodulators", {}).get("emotion", "?")
                    meta_chains = td.get("meta_reasoning", {}).get("total_chains", 0)
                    commits = td.get("reasoning", {}).get("commits", 0)
                    vocab = td.get("vocabulary", {}).get("total_words", 0)
                    parts.append(f"T{label[-1]}: em={em} meta={meta_chains} commits={commits} vocab={vocab}")
                log.info("[%d] %s", snapshot_num, " | ".join(parts))
            except Exception as e:
                log.warning("[%d] Snapshot error: %s", snapshot_num, e)

            # Save periodically (every 10 snapshots)
            if snapshot_num % 10 == 0:
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                telemetry.save(f"data/twin_telemetry_{ts}.json")

            await asyncio.sleep(SNAPSHOT_INTERVAL)

    # Final save
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = f"data/twin_telemetry_{ts}.json"
    telemetry.save(json_path)
    log.info("Telemetry saved to %s (%d snapshots)", json_path, len(telemetry.snapshots))

    # Generate report
    report = telemetry.generate_report()
    os.makedirs("titan-docs/reports", exist_ok=True)
    report_path = f"titan-docs/reports/REPORT_twin_experiment_{ts}.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Report saved to %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twin Experiment Telemetry")
    parser.add_argument("--duration", type=int, default=240,
                        help="Duration in minutes (default 240 = 4 hours)")
    args = parser.parse_args()
    asyncio.run(main(args))
