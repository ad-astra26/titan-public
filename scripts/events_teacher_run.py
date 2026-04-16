#!/usr/bin/env python3
"""
events_teacher_run.py — Standalone cron script for Events Teacher.

Hot-reloads events_teacher module on each run. Runs one social perception
window, then exits. Designed for cron (every 30 min).

Usage:
    python scripts/events_teacher_run.py --titan T1
    python scripts/events_teacher_run.py --titan T1 --verbose
"""
import sys
import argparse
import logging
import importlib
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TITAN_CONFIGS = {
    "T1": {"host": "127.0.0.1", "port": 7777},
    "T2": {"host": "127.0.0.1", "port": 7777},
    "T3": {"host": "127.0.0.1", "port": 7778},
}


def main():
    parser = argparse.ArgumentParser(
        description="Events Teacher — Social Perception Window")
    parser.add_argument("--titan", required=True, choices=["T1", "T2", "T3"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("events_teacher")

    # Load merged config (config.toml + ~/.titan/secrets.toml)
    from titan_plugin.config_loader import load_titan_config
    config = load_titan_config()
    if not config:
        logger.error("Merged config empty — check titan_plugin/config.toml exists")
        sys.exit(1)

    # API base URL
    net = TITAN_CONFIGS[args.titan]
    api_base = f"http://{net['host']}:{net['port']}"

    # Health check
    import httpx
    try:
        resp = httpx.get(f"{api_base}/health", timeout=10)
        if resp.status_code != 200:
            logger.error("[%s] Health check failed: HTTP %d",
                         args.titan, resp.status_code)
            sys.exit(1)
        logger.info("[%s] Health OK. Running Events Teacher window...",
                    args.titan)
    except Exception as e:
        logger.error("[%s] Health check failed: %s", args.titan, e)
        sys.exit(1)

    # Hot-reload the events_teacher module (picks up code changes)
    import titan_plugin.logic.events_teacher as et_module
    importlib.reload(et_module)

    # Load state, run window, save state
    teacher = et_module.EventsTeacher.from_state()
    result = teacher.run_window(
        titan_id=args.titan,
        api_base=api_base,
        config=config,
    )

    if result.skipped_reason:
        logger.info("[%s] Window skipped: %s",
                    args.titan, result.skipped_reason)
    else:
        logger.info("[%s] Window #%d: %d stored, %d API calls",
                    args.titan, result.window_number,
                    result.events_stored, result.api_calls_used)

    # ── Phase 1: Bridge perception events to DivineBus via API ──
    if result.perception_events:
        try:
            resp = httpx.post(
                f"{api_base}/v4/social-perception",
                json={
                    "titan_id": args.titan,
                    "events": result.perception_events,
                },
                timeout=10,
            )
            data = resp.json()
            published = data.get("data", {}).get("published", 0)
            logger.info("[%s] Social perception: %d events published to bus",
                        args.titan, published)
        except Exception as e:
            logger.warning("[%s] Social perception POST failed: %s",
                           args.titan, e)


if __name__ == "__main__":
    main()
