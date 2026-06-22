"""Boot-driver parity invariant for cognitive_worker.

Codified 2026-05-10 after the pre-D8 ownership audit closed BUG-COGNITIVE-
WORKER-{REASONING,EXPRESSION,MSL,INNER-LOWER-TOPO}-* + SPEAK-silent-on-T3.
All four bugs were the same disease: cognitive_worker booted Python L2
engines into ``state_refs`` but the corresponding per-epoch driver call
in ``_drive_one_epoch`` was forgotten when chunk 8E moved them off
spirit_worker. Symptoms only surfaced once a downstream signal flat-lined
(expression frozen 2 days, MSL i_confidence frozen, IQL paused, chi
grounding stuck at 0.5 default).

This test enforces the invariant: every Python L2 engine cognitive_worker
boots into ``state_refs`` MUST be referenced in ``_drive_one_epoch``.
Adding a new engine requires extending the canonical table — which is
the explicit, reviewable contract.

2026-05-10 follow-up refactor: switched from method-name-based to
**engine-name-based** detection. The narrow regex variant
(``\\.(tick|evaluate|compute|step|observe)\\(``) was blind to engines
whose driver methods aren't on that fixed list — e.g. WorkingMemory uses
``attend()`` / ``decay()`` / ``get_context()``, EpisodicMemory uses
``record_episode()``, NeuromodRewardObserver had ``levels_provider=none``
silently, etc. The new test treats ANY mention of the engine_key in
``_drive_one_epoch`` source as a valid driver wiring (you must
explicitly mark passive engines as ``DRIVER_PASSIVE`` in the table).

The test uses static source inspection (not a live run) for two reasons:
  1. ``_drive_one_epoch`` has many cross-engine couplings; running it
     end-to-end with mocks would be brittle and high-maintenance.
  2. The bug class we want to catch is "engine never referenced from the
     driver" — exactly what static inspection nails cheaply.
"""
from __future__ import annotations

import inspect
import re

import pytest

from titan_hcl.modules import cognitive_worker


# Sentinel marking engines that don't need a per-tick driver in
# _drive_one_epoch (e.g. inner_state / spirit_state — passive containers
# updated indirectly via coordinator.tick).
DRIVER_PASSIVE = "<passive>"


# Canonical (engine_key_in_state_refs, driver_marker) pairs.
#
# When you add a new engine to ``_init_cognitive_engines``'s return dict:
#   - If it has a per-tick driver, add (engine_key, "<engine_key>") here.
#   - If it's intentionally passive (no per-tick interaction), add
#     (engine_key, DRIVER_PASSIVE).
#
# Engine-name detection rationale (2026-05-10): method-name regex was
# blind to engines using non-canonical method names (working_mem.attend,
# episodic_mem.record_episode, etc.) — surfaced when 13+ engines were
# discovered missing from cognitive_worker after T3 deploy of the
# initial pre-D8 audit closure (commit 391dcab2).
BOOT_DRIVER_PARITY: list[tuple[str, str]] = [
    # ── Already-wired (steps 2-7 in _drive_one_epoch) ──────────────
    ("coordinator", "coordinator"),
    ("life_force_engine", "life_force_engine"),
    ("neural_nervous_system", "neural_nervous_system"),
    ("pi_monitor", "pi_monitor"),
    ("meta_engine", "meta_engine"),
    # ── Closed 2026-05-10 by pre-D8 audit Block A-E ────────────────
    ("reasoning_engine", "reasoning_engine"),
    ("expression_manager", "expression_manager"),
    ("msl", "msl"),
    ("neuromod_reward_observer", "neuromod_reward_observer"),
    # ── Block D — Tier 1 SPEAK deps ────────────────────────────────
    ("exp_orchestrator", "exp_orchestrator"),
    ("social_pressure_meter", "social_pressure_meter"),
    # ── Meta-reasoning foundation (M1-M3) ──────────────────────────
    ("chain_archive", "chain_archive"),
    ("meta_wisdom", "meta_wisdom"),
    ("meta_autoencoder", "meta_autoencoder"),
    # Memory engines used by Tier 1 SPEAK + ExperienceOrchestrator:
    ("ex_mem", "ex_mem"),
    ("e_mem", DRIVER_PASSIVE),  # Backing store consumed by exp_orchestrator
    # ── Block F (2026-05-10): pre-D8 audit Track 1 migrations ──────
    # prediction_engine REMOVED from this list 2026-05-12 (Track 2
    # drift correction commit B8 per rFP_phase_c_self_improvement_subsystem_migration
    # §0 table). PredictionEngine now lives in self_reflection_worker —
    # cognitive_worker consumes PREDICTION_GENERATED bus events via the
    # dispatcher → state_refs["_latest_prediction"] cache slot.
    ("working_mem", "working_mem"),
    # episodic_mem: marker is the REAL driver call, not the engine name. The faculty
    # is driven via _dispatch_episode_record(...) from _drive_one_epoch (great_pulse @
    # SOVEREIGNTY_EPOCH + dreaming_start/end @ the dream edges) per
    # RFP_phase_c_actr_memory_rehoming §4.1. This CLOSES the gaming the old
    # `_ = episodic_mem` parity-anchor enabled (a bare reference with no live call
    # while the faculty was dead) — the marker now demands an actual writer call.
    ("episodic_mem", "_dispatch_episode_record"),
    ("intuition_convergence", "intuition_convergence"),
    ("wallet_observer", "wallet_observer"),
    ("meta_recruitment", "meta_recruitment"),
    ("timeseries_store", "timeseries_store"),
    ("mini_registry", "mini_registry"),
    ("interpreter", "interpreter"),
    ("med_watchdog", "med_watchdog"),
    # ── Passive containers (updated via coordinator.tick) ──────────
    ("inner_state", DRIVER_PASSIVE),
    ("spirit_state", DRIVER_PASSIVE),
    ("observable_engine", DRIVER_PASSIVE),
]


@pytest.fixture(scope="module")
def drive_one_epoch_source() -> str:
    """Return the verbatim source of ``_drive_one_epoch``."""
    return inspect.getsource(cognitive_worker._drive_one_epoch)


@pytest.fixture(scope="module")
def init_cognitive_engines_source() -> str:
    """Return the verbatim source of ``_init_cognitive_engines``."""
    return inspect.getsource(cognitive_worker._init_cognitive_engines)


def test_init_cognitive_engines_returns_expected_keys(init_cognitive_engines_source: str):
    """Sanity: every entry in BOOT_DRIVER_PARITY is in
    ``_init_cognitive_engines``'s return dict. Inverse direction (engines
    booted but missing from the table) is checked by the next test."""
    return_block_match = re.search(
        r"return\s*\{(.*?)\n\s*\}\s*$",
        init_cognitive_engines_source,
        re.DOTALL,
    )
    assert return_block_match, "Could not locate return-dict block"
    return_block = return_block_match.group(1)
    for engine_key, _ in BOOT_DRIVER_PARITY:
        pattern = rf'"{re.escape(engine_key)}"\s*:'
        assert re.search(pattern, return_block), (
            f"Engine '{engine_key}' is in BOOT_DRIVER_PARITY but missing "
            f"from _init_cognitive_engines return dict — drop the entry "
            f"or add the boot.")


def test_no_engine_booted_without_table_entry(init_cognitive_engines_source: str):
    """Catch the inverse direction: engines added to the return dict
    must be added to BOOT_DRIVER_PARITY. Forces an explicit decision:
    is this engine driven or passive?"""
    return_block_match = re.search(
        r"return\s*\{(.*?)\n\s*\}\s*$",
        init_cognitive_engines_source,
        re.DOTALL,
    )
    assert return_block_match, "Could not locate return-dict block"
    return_block = return_block_match.group(1)
    # Find all "engine_key": entries.
    booted_keys = set(re.findall(r'"([a-z_][a-zA-Z0-9_]*)"\s*:', return_block))
    table_keys = {k for k, _ in BOOT_DRIVER_PARITY}
    # Underscore-prefixed keys (e.g. _neuromod_reader) are internal
    # plumbing, not L2 engines; skip them.
    booted_engines = {k for k in booted_keys if not k.startswith("_")}
    missing_from_table = booted_engines - table_keys
    assert not missing_from_table, (
        f"Engine(s) {sorted(missing_from_table)} booted in "
        f"_init_cognitive_engines but missing from BOOT_DRIVER_PARITY. "
        f"Either add to the table with a driver marker, or mark as "
        f"DRIVER_PASSIVE if intentionally not driven per-tick.")


@pytest.mark.parametrize(("engine_key", "driver_marker"), BOOT_DRIVER_PARITY)
def test_drive_one_epoch_references_engine(
    engine_key: str,
    driver_marker: str,
    drive_one_epoch_source: str,
):
    """Boot-driver parity: each booted engine must be referenced in
    ``_drive_one_epoch`` (any method, any reference) UNLESS marked
    DRIVER_PASSIVE."""
    if driver_marker is DRIVER_PASSIVE:
        return  # Explicit no-driver — passive container.

    # Engine name appears anywhere in the driver source.
    # Word-boundary anchored so substrings don't false-match
    # (e.g. "ex_mem" must not match "_msl_concept_confidences" or similar).
    pattern = rf'\b{re.escape(driver_marker)}\b'
    assert re.search(pattern, drive_one_epoch_source), (
        f"BOOT-DRIVER PARITY VIOLATION: engine '{engine_key}' is booted "
        f"into state_refs but '{driver_marker}' is never referenced in "
        f"_drive_one_epoch. Either wire the driver or mark "
        f"DRIVER_PASSIVE if intentionally inert.")


def test_canonical_table_has_no_duplicates():
    """Sanity: the parity table itself is well-formed."""
    keys = [k for k, _ in BOOT_DRIVER_PARITY]
    assert len(keys) == len(set(keys)), (
        f"Duplicate keys in BOOT_DRIVER_PARITY: "
        f"{[k for k in keys if keys.count(k) > 1]}")
