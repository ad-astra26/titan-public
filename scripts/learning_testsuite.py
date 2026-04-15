"""
Autonomous Learning TestSuite — Titan's external teaching engine.

Runs alongside Titan as a standalone process (like twin_telemetry.py).
Observes Titan's state, selects optimal learning modules, teaches words,
reinforces vocabulary, and tracks developmental progress.

Auto-restarts via cron. Checkpoint/resume survives crashes.
Communicates with Titan exclusively via HTTP API.

Usage:
    source test_env/bin/activate
    python scripts/learning_testsuite.py
    python scripts/learning_testsuite.py --name titan2 --api http://10.135.0.6:7777

Auto-restart (add to cron alongside telemetry):
    pgrep -f "learning_testsuite" || cd /path/to/titan && source test_env/bin/activate && \
        nohup python scripts/learning_testsuite.py > /tmp/learning_testsuite.log 2>&1 &
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time

import httpx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.learning.smart_scheduler import SmartScheduler
from scripts.learning.curriculum_manager import CurriculumManager
from scripts.learning.api_helpers import (
    is_titan_alive, get_titan_state, wait_for_new_epoch, get_epoch_id,
)
from scripts.learning.modules.language_module import LanguageModule
from scripts.learning.modules.composition_module import CompositionModule
from scripts.learning.modules.arc_play_module import ArcPlayModule

log = logging.getLogger("testsuite")

# Check interval when not teaching — derived from Schumann body clock
# SCHUMANN_BODY (3.45s) × 9 = ~31s = one full mind cycle
# This is the natural observation rhythm, not an arbitrary human time
SCHUMANN_BODY = 3.45
CHECK_INTERVAL = SCHUMANN_BODY * 9  # ~31s


class ModuleRegistry:
    """Registry of all available learning modules."""

    def __init__(self, instance_name: str = "titan1"):
        self._instance = instance_name
        self._modules = {}
        self._register_defaults()

    def _register_defaults(self):
        self._modules["language"] = LanguageModule(instance_name=self._instance)
        self._modules["composition"] = CompositionModule(instance_name=self._instance)
        self._modules["arc_play"] = ArcPlayModule(instance_name=self._instance)
        # Future: conversation, music, art_narration, assessment
        # For now, unimplemented modules fall back to language
        log.info("[Registry] Registered modules for %s: %s",
                 self._instance, list(self._modules.keys()))

    def get(self, module_type: str):
        """Get module by type. Falls back to language for unimplemented types."""
        module = self._modules.get(module_type)
        if module:
            return module
        # Fallback for unimplemented modules
        if module_type in ("rest",):
            return None  # Rest = do nothing
        log.debug("[Registry] Module '%s' not implemented, falling back to language",
                  module_type)
        return self._modules.get("language")

    def list_types(self) -> list[str]:
        return list(self._modules.keys())


async def run_module(module, client: httpx.AsyncClient, api: str,
                     state: dict, curriculum: CurriculumManager) -> dict:
    """Run a single learning module and return result."""
    try:
        result = await module.run(client, api, state, curriculum)
        return result
    except Exception as e:
        log.error("[TestSuite] Module execution error: %s", e)
        return {"type": "error", "success": False, "error": str(e), "duration": 0}


async def handle_rest(client: httpx.AsyncClient, api: str, state: dict) -> dict:
    """Rest period: wait for one epoch, let Titan self-explore."""
    log.info("[TestSuite] REST — letting Titan self-explore for 1 epoch")
    epoch_before = await get_epoch_id(client, api)
    await wait_for_new_epoch(client, api, epoch_before, timeout_s=180)
    return {"type": "rest", "success": True, "duration": 0, "words_taught": 0}


async def main_loop(args):
    """Main autonomous learning loop."""
    api = args.api
    name = args.name

    scheduler = SmartScheduler()
    curriculum = CurriculumManager(
        checkpoint_path=f"data/testsuite_checkpoint_{name}.json")
    modules = ModuleRegistry(instance_name=name)

    log.info("[TestSuite] Started for %s at %s — day %d, position %d, phase %s",
             name, api, curriculum.day, curriculum.position, curriculum.phase)

    async with httpx.AsyncClient() as client:
        while True:
            try:
                # 1. Check if Titan is alive
                if not await is_titan_alive(client, api):
                    log.info("[TestSuite] %s not reachable, waiting %ds...", name, CHECK_INTERVAL)
                    await asyncio.sleep(CHECK_INTERVAL)
                    continue

                # 2. Get full state
                state = await get_titan_state(client, api)
                curriculum.update_vocab_count(state.get("vocab_size", 0))

                # 3. Should we teach now?
                if not scheduler.should_teach_now(state):
                    # Not ready — wait and check again
                    await asyncio.sleep(CHECK_INTERVAL)
                    continue

                # 4. Get next module from curriculum
                module_spec = curriculum.get_next_module(state)
                module_type = module_spec.get("type", "language")

                # Let scheduler override type based on emotional state
                # (unless curriculum specifies a specific required type)
                if module_type not in ("rest",):
                    suggested = scheduler.select_module_type(state, module_type)
                    if suggested != module_type:
                        log.info("[TestSuite] Scheduler suggests %s instead of %s",
                                 suggested, module_type)
                        module_type = suggested

                # 5. Handle rest periods
                if module_type == "rest":
                    result = await handle_rest(client, api, state)
                    curriculum.advance_position()
                    curriculum.save_checkpoint(result)
                    continue

                # 6. Get module and run it
                module = modules.get(module_type)
                if not module:
                    log.warning("[TestSuite] No module for type '%s', skipping", module_type)
                    curriculum.advance_position()
                    continue

                # How many modules should we run in this window?
                module_count = scheduler.select_module_count(state)
                log.info("[TestSuite] Running %d module(s), starting with %s",
                         module_count, module_type)

                for i in range(module_count):
                    # Re-check state before each additional module
                    if i > 0:
                        state = await get_titan_state(client, api)
                        if not scheduler.should_teach_now(state):
                            log.info("[TestSuite] Titan no longer ready after %d modules", i)
                            break
                        module_spec = curriculum.get_next_module(state)
                        module_type = module_spec.get("type", "language")
                        if module_type == "rest":
                            await handle_rest(client, api, state)
                            curriculum.advance_position()
                            curriculum.save_checkpoint({"type": "rest", "success": True,
                                                        "words_taught": 0, "duration": 0})
                            continue
                        module = modules.get(module_type)
                        if not module:
                            curriculum.advance_position()
                            continue

                    # Run module
                    log.info("[TestSuite] === Module %d/%d: %s ===", i + 1, module_count, module_type)
                    result = await run_module(module, client, api, state, curriculum)

                    # If language says all words mastered, substitute composition
                    # Check result type (not module_type) because unimplemented modules
                    # fall back to language via ModuleRegistry
                    if (result.get("error") == "no_words"
                            and result.get("all_mastered")
                            and result.get("type") == "language"):
                        comp = modules.get("composition")
                        if comp:
                            log.info("[TestSuite] All vocab mastered — substituting composition")
                            result = await run_module(comp, client, api, state, curriculum)

                    # Save checkpoint + advance
                    curriculum.advance_position()
                    curriculum.save_checkpoint(result)

                    log.info("[TestSuite] Module complete: type=%s words=%d accuracy=%.1f%% duration=%.0fs",
                             result.get("type", "?"),
                             result.get("words_taught", 0),
                             result.get("accuracy", 0) * 100,
                             result.get("duration", 0))

                    # ── Exploration break between modules ──
                    # Let Titan self-explore for 1-2 epochs before next module.
                    # This ensures the two-engine synergy: teach → explore → teach.
                    # Self-exploration during this break creates emotional variability
                    # that makes the NEXT teaching session more effective.
                    epoch_before_break = await get_epoch_id(client, api)
                    log.info("[TestSuite] EXPLORATION BREAK — letting Titan self-explore for 1 epoch")
                    await wait_for_new_epoch(client, api, epoch_before_break, timeout_s=180)
                    log.info("[TestSuite] Exploration break complete — checking readiness for next module")

            except KeyboardInterrupt:
                log.info("[TestSuite] Interrupted — saving checkpoint")
                curriculum.save_checkpoint()
                break
            except Exception as e:
                log.error("[TestSuite] Main loop error: %s", e)
                await asyncio.sleep(CHECK_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description="Autonomous Learning TestSuite")
    parser.add_argument("--api", default="http://localhost:7777",
                        help="Titan API base URL")
    parser.add_argument("--name", default="titan1",
                        help="Instance name (titan1 or titan2)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("[TestSuite] Learning TestSuite v1.0 — %s at %s", args.name, args.api)
    asyncio.run(main_loop(args))


if __name__ == "__main__":
    main()
