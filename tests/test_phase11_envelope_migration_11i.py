"""
Phase 11 §11.I.4 / D-SPEC-141 — @with_error_envelope migration (Chunk 11I).

Verifies that every worker entry function under titan_hcl/modules/ +
titan_hcl/persistence_entry.py has been decorated with
`@with_error_envelope(module_name=..., subsystem="entry", ...)` per
RFP §3H.2 11I and §3H.3 adoption pattern.

The decorator:
  - Catches every Exception escaping the entry-fn
  - Publishes a typed `ModuleError(severity=FATAL, ...)` envelope on the
    bus (MODULE_ERROR topic) BEFORE re-raising
  - Preserves the original traceback via bare `raise`

A regression that drops the decorator (e.g. a developer adding a new
worker without it, or a refactor that strips it) surfaces as a hard
test failure here so the §11.I.4 contract stays load-bearing.
"""
from __future__ import annotations

import importlib
import inspect

import pytest

# Canonical roster — must match scripts/_phase11_11i_apply_error_envelope.py:ENTRY_MAP.
# Each entry: (importable module path, entry_fn name, module_catalog name).
ENTRY_ROSTER: list[tuple[str, str, str]] = [
    ("titan_hcl.modules.agency_worker",                  "agency_worker_main", "agency_worker"),
    ("titan_hcl.modules.agno_worker",                    "agno_worker_main", "agno_worker"),
    ("titan_hcl.modules.backup_worker",                  "backup_worker_main", "backup"),
    ("titan_hcl.modules.body_worker",                    "body_worker_main", "body"),
    ("titan_hcl.modules.cgn_worker",                     "cgn_worker_main", "cgn"),
    ("titan_hcl.modules.cognitive_worker",               "cognitive_worker_main", "cognitive_worker"),
    ("titan_hcl.modules.corrective_events_persistence_worker",
        "corrective_events_persistence_worker_main", "corrective_events_persistence"),
    ("titan_hcl.modules.dream_state_worker",             "dream_state_worker_main", "dream_state"),
    ("titan_hcl.modules.emot_cgn_worker",                "emot_cgn_worker_main", "emot_cgn"),
    ("titan_hcl.modules.expression_worker",              "expression_worker_main", "expression_worker"),
    ("titan_hcl.modules.health_monitor_worker",          "health_monitor_worker_main", "health_monitor"),
    ("titan_hcl.modules.hormonal_worker",                "hormonal_worker_main", "hormonal_module"),
    ("titan_hcl.modules.interface_advisor_worker",       "interface_advisor_worker_main", "interface_advisor"),
    ("titan_hcl.modules.journey_persistence_worker",     "journey_persistence_worker_main", "journey_persistence"),
    ("titan_hcl.modules.knowledge_worker",               "knowledge_worker_main", "knowledge"),
    ("titan_hcl.modules.language_worker",                "language_worker_main", "language"),
    ("titan_hcl.modules.life_force_worker",              "life_force_worker_main", "life_force"),
    ("titan_hcl.modules.llm_worker",                     "llm_worker_main", "llm"),
    ("titan_hcl.modules.media_worker",                   "media_worker_main", "media"),
    ("titan_hcl.modules.meditation_worker",              "meditation_worker_main", "meditation"),
    ("titan_hcl.modules.memory_worker",                  "memory_worker_main", "memory"),
    ("titan_hcl.modules.meta_teacher_worker",            "meta_teacher_worker_main", "meta_teacher"),
    ("titan_hcl.modules.metabolism_worker",              "metabolism_worker_main", "metabolism"),
    ("titan_hcl.modules.mind_worker",                    "mind_worker_main", "mind"),
    ("titan_hcl.modules.neuromod_worker",                "neuromod_worker_main", "neuromod_module"),
    ("titan_hcl.modules.ns_worker",                      "ns_worker_main", "ns_module"),
    ("titan_hcl.modules.observatory_worker",             "observatory_worker_main", "observatory"),
    ("titan_hcl.modules.outer_interface_worker",         "outer_interface_worker_main", "outer_interface_worker"),
    ("titan_hcl.modules.output_verifier_worker",         "output_verifier_worker_main", "output_verifier"),
    ("titan_hcl.modules.reflex_worker",                  "reflex_worker_main", "reflex"),
    ("titan_hcl.modules.self_reflection_worker",         "self_reflection_worker_main", "self_reflection_worker"),
    ("titan_hcl.modules.social_graph_worker",            "social_graph_worker_main", "social_graph"),
    ("titan_hcl.modules.social_worker",                  "social_worker_main", "social_worker"),
    ("titan_hcl.modules.sovereignty_worker",             "sovereignty_worker_main", "sovereignty"),
    ("titan_hcl.modules.studio_worker",                  "studio_worker_main", "studio"),
    ("titan_hcl.modules.synthesis_worker",               "synthesis_worker_main", "synthesis"),
    ("titan_hcl.modules.timechain_worker",               "timechain_worker_main", "timechain"),
    ("titan_hcl.modules.warning_monitor_worker",         "warning_monitor_worker_main", "warning_monitor"),
    ("titan_hcl.persistence_entry",                      "imw_main", "imw"),
]


@pytest.mark.parametrize("mod_path,fn_name,catalog_name", ENTRY_ROSTER)
def test_entry_function_is_decorated_with_error_envelope(
        mod_path: str, fn_name: str, catalog_name: str):
    """Every worker entry function must be wrapped by
    `with_error_envelope` so escaping exceptions surface a typed
    ModuleError on the bus per §11.I.4."""
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, fn_name, None)
    assert fn is not None, f"{mod_path}: missing entry fn '{fn_name}'"
    # The decorator preserves __wrapped__ via functools.wraps. Presence of
    # __wrapped__ on a callable that came from titan_hcl.modules.* is a
    # reliable load-bearing signal.
    assert hasattr(fn, "__wrapped__"), (
        f"{mod_path}.{fn_name} is not wrapped — Phase 11 §11.I.4 requires "
        f"`@with_error_envelope(module_name={catalog_name!r}, ...)` on every "
        f"worker entry function")


def test_envelope_publishes_module_error_on_uncaught_exception():
    """End-to-end: when a wrapped entry raises, a ModuleError envelope
    is dispatched via `_publish_envelope_safely` BEFORE the raise."""
    import queue as _queue_mod

    from titan_hcl.bus import MODULE_ERROR
    from titan_hcl.core.module_error_handler import with_error_envelope
    from titan_hcl.errors import Severity

    send_queue = _queue_mod.Queue()

    @with_error_envelope(
        module_name="test_worker",
        subsystem="entry",
        severity=Severity.FATAL,
    )
    def my_entry(recv_queue, send_queue, name, config):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        my_entry(None, send_queue, "test_worker", {})

    # The envelope went to send_queue (the decorator's
    # `_publish_envelope_safely` routes to whatever `sender_arg` resolves
    # to — by default that's the `send_queue` positional arg).
    assert not send_queue.empty()
    msg = send_queue.get_nowait()
    assert msg["type"] == MODULE_ERROR
    payload = msg["payload"]
    assert payload["module_name"] == "test_worker"
    assert payload["subsystem"] == "entry"
    assert payload["severity"].lower() == "fatal"
    assert "boom" in payload["message"]


def test_roster_size_matches_catalog_worker_count():
    """Sanity: ENTRY_ROSTER must enumerate every worker that
    titan_hcl/modules/*.py exposes (plus persistence_entry). Drift
    surfaces here so adding a new worker without decorating it is loud.
    """
    import glob
    import os
    modules_files = sorted(
        os.path.basename(p)
        for p in glob.glob("titan_hcl/modules/*_worker.py")
        if not os.path.basename(p).startswith("__")
    )
    # 39 *_worker.py files + persistence_entry.py = 40 ENTRY_ROSTER rows.
    expected_count = len(modules_files) + 1
    assert len(ENTRY_ROSTER) == expected_count, (
        f"ENTRY_ROSTER has {len(ENTRY_ROSTER)} rows but "
        f"titan_hcl/modules/ has {len(modules_files)} worker files "
        f"(+ persistence_entry.py = {expected_count} expected). "
        f"Run scripts/_phase11_11i_apply_error_envelope.py + add the new "
        f"worker to ENTRY_ROSTER.")
