"""
api/v6 — the Observatory's single readout roof (Phase E).

RFP_phase_c_titan_hcl_cleanup §2 Phase E. ONE documented namespace the Observatory
consumes; `/v3` + `/v4` are hard-deprecated (301/410, see v6_deprecation.py). Every
route's data lineage is declared in this module's ROUTE_TABLE → `v6_manifest` and
exposed at `GET /v6/manifest`.

**Consolidation model (reconciles "semantic consolidation" with "renders identically
pre/post"):** each semantic group (taxonomy §1 of DESIGN_phase_e_api_v6.md) is a
namespace under `/v6/<group>`. Within a group, **shape-preserving leaves**
(`/v6/<group>/<leaf>`) keep the exact response shape the Observatory already renders
(the frontend maps each old `/v4/<x>` slug 1:1 to its v6 leaf → byte-identical render),
and a **composite group root** (`GET /v6/<group>`) is the "fatter" consolidated view
for new wiring + the manifest roof.

**Faithfulness + no double-maintenance:** each v6 route is the SAME handler function
already implemented in `dashboard.py`, re-mounted under its v6 path via
`add_api_route` (FastAPI parses the function's real signature, so path/query params
are preserved exactly). There is exactly ONE implementation per readout; the legacy
`/v3`,`/v4` decorators become 301 redirects to the v6 path (E.4), so the dashboard
function is reached only via v6 — no shim, no fork.

**Sourcing (Preamble G18):** readout leaves read SHM-direct via `TitanStateAccessor`
(`request.app.state.titan_state`); a few plugin/kernel-only diagnostics also use the
kernel-RPC proxy (`request.app.state.titan_hcl`) — the manifest marks those `rpc=True`.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from . import dashboard as _dash
from . import v6_manifest as _m
from .v6_manifest import RouteSpec

logger = logging.getLogger(__name__)

router = APIRouter(tags=["v6"])


# ── ROUTE_TABLE — the single declarative source for the v6 surface ───────────
# Each tuple:
#   (v6_path, method, dashboard_func_name, group, kind, accessor, command,
#    shm_slots, producers, rpc, replaces)
# The legacy /v3,/v4 path each row subsumes is `replaces` (drives the 301 map).
# kind: readout | mutation | admin.
_T = (
    # ── trinity ──────────────────────────────────────────────────────────
    ("/v6/trinity", "GET", "get_v3_trinity", "trinity", "readout",
     "spirit.get_trinity", None, ("trinity_state.bin",), ("titan-inner-spirit-rs",), True, "/v3/trinity"),
    ("/v6/trinity/history", "GET", "get_trinity_history", "trinity", "readout",
     None, None, (), ("observatory_worker",), False, "/v3/trinity/history"),
    ("/v6/trinity/shm", "GET", "get_v4_trinity_shm", "trinity", "readout",
     "shm.read_trinity", None, ("trinity_state.bin",), ("titan-inner-spirit-rs",), True, "/v4/trinity-shm"),
    ("/v6/trinity/inner", "GET", "get_v4_inner_trinity", "trinity", "readout",
     "spirit.read_inner_trinity", None,
     ("inner_body_5d.bin", "inner_mind_15d.bin", "inner_spirit_45d.bin", "topology_30d.bin", "sphere_clocks.bin", "chi.bin"),
     ("titan-inner-body-rs", "titan-inner-mind-rs", "titan-inner-spirit-rs"), True, "/v4/inner-trinity"),
    ("/v6/trinity/state", "GET", "get_v4_state", "trinity", "readout",
     "spirit.get_v4_state", None,
     ("sphere_clocks.bin", "chi.bin", "topology_30d.bin", "epoch.bin", "neuromod_state.bin", "resonance_metadata.bin", "unified_spirit_metadata.bin", "filter_down_state.bin"),
     ("titan-inner-spirit-rs", "titan-unified-spirit-rs"), True, "/v4/state"),
    ("/v6/trinity/state-snapshot", "GET", "get_v4_state_snapshot", "trinity", "readout",
     None, None, (), ("kernel",), False, "/v4/state-snapshot"),
    ("/v6/trinity/cache-staleness", "GET", "get_v4_cache_staleness", "trinity", "readout",
     None, None, (), ("kernel",), False, "/v4/cache-staleness"),
    ("/v6/trinity/layers", "GET", "get_v4_layers", "trinity", "readout",
     "guardian.layer_stats", None, ("guardian_state.bin",), ("guardian",), True, "/v4/layers"),
    ("/v6/trinity/sphere-clocks", "GET", "get_v4_sphere_clocks", "trinity", "readout",
     "spirit.get_sphere_clocks", None, ("sphere_clocks.bin",), ("titan-inner-spirit-rs",), False, "/v4/sphere-clocks"),
    ("/v6/trinity/resonance", "GET", "get_v4_resonance", "trinity", "readout",
     "spirit.get_resonance", None, ("resonance_metadata.bin",), ("titan-unified-spirit-rs",), False, "/v4/resonance"),
    ("/v6/trinity/unified-spirit", "GET", "get_v4_unified_spirit", "trinity", "readout",
     "spirit.get_unified_spirit", None, ("unified_spirit_metadata.bin",), ("titan-unified-spirit-rs",), False, "/v4/unified-spirit"),
    ("/v6/trinity/filter-down-status", "GET", "get_v4_filter_down_status", "trinity", "readout",
     "spirit.get_filter_down_status", None, ("filter_down_state.bin",), ("titan-inner-spirit-rs",), False, "/v4/filter-down-status"),
    ("/v6/trinity/sensors", "GET", "get_v4_sensors", "trinity", "readout",
     None, None, (), (), False, "/v4/sensors"),
    ("/v6/trinity/agency", "GET", "get_v3_agency", "trinity", "readout",
     "agency.get_stats", None, ("agency_state.bin",), ("agency_worker",), False, "/v3/agency"),
    # P0.6-C / D-SPEC-132 §6.6.6 PolarityHomeostat telemetry — reads recent
    # trinity_corrective_events rows from consciousness.db + computes per
    # (source_part, side) summary stats (count, rate, sigma_avg, etc.).
    ("/v6/trinity/polarity-homeostat", "GET", "get_v6_polarity_homeostat", "trinity", "readout",
     None, None, (), ("corrective_events_persistence",), False, None),

    # ── nervous-system ───────────────────────────────────────────────────
    ("/v6/nervous-system", "GET", "get_v4_nervous_system", "nervous-system", "readout",
     "spirit.get_nervous_system", None, ("titanvm_registers.bin",), ("ns_worker",), False, "/v4/nervous-system"),
    ("/v6/nervous-system/neuromodulators", "GET", "get_v4_neuromodulators", "nervous-system", "readout",
     "neuromods.read", None, ("neuromod_state.bin",), ("cognitive_worker",), False, "/v4/neuromodulators"),
    ("/v6/nervous-system/hormonal-system", "GET", "get_v4_hormonal_system", "nervous-system", "readout",
     "shm.read_hormonal", None, ("titanvm_registers.bin",), ("ns_worker",), False, "/v4/hormonal-system"),
    ("/v6/nervous-system/ns-health", "GET", "ns_health", "nervous-system", "readout",
     None, None, ("titanvm_registers.bin",), ("ns_worker",), True, "/v4/ns-health"),
    ("/v6/nervous-system/titan-vm", "GET", "titan_vm_diagnostics", "nervous-system", "readout",
     "shm.read_titanvm_registers", None, ("titanvm_registers.bin",), ("ns_worker",), False, "/v4/titan-vm"),
    ("/v6/nervous-system/chi", "GET", "get_v4_chi", "nervous-system", "readout",
     "shm.read_chi", None, ("chi.bin",), ("titan-inner-spirit-rs",), False, "/v4/chi"),
    ("/v6/nervous-system/pi-heartbeat", "GET", "get_v4_pi_heartbeat", "nervous-system", "readout",
     "shm.read_pi_heartbeat", None, ("pi_heartbeat.bin",), ("kernel",), False, "/v4/pi-heartbeat"),

    # ── expression ───────────────────────────────────────────────────────
    ("/v6/expression", "GET", "get_v4_expression_composites", "expression", "readout",
     "spirit.get_expression_composites", None, ("expression_state.bin",), ("expression_worker",), False, "/v4/expression-composites"),
    ("/v6/expression/creative-journal", "GET", "creative_journal", "expression", "readout",
     None, None, (), ("observatory_worker",), False, "/v4/creative-journal"),
    ("/v6/expression/creative-works", "GET", "creative_works", "expression", "readout",
     None, None, (), ("observatory_worker",), False, "/v4/creative-works"),
    ("/v6/expression/mood-narrative", "GET", "mood_narrative", "expression", "readout",
     None, None, (), ("observatory_worker",), False, "/v4/mood-narrative"),
    ("/v6/expression/state-narration", "GET", "state_narration", "expression", "readout",
     None, None, (), ("observatory_worker",), False, "/v4/state-narration"),
    ("/v6/expression/narrate-art", "GET", "narrate_art", "expression", "readout",
     None, None, (), ("observatory_worker",), False, "/v4/narrate-art"),
    ("/v6/expression/narrated-feed", "GET", "narrated_feed", "expression", "readout",
     None, None, (), ("observatory_worker",), False, "/v4/narrated-feed"),
    ("/v6/expression/activity-feed", "GET", "activity_feed", "expression", "readout",
     None, None, (), ("observatory_worker",), False, "/v4/activity-feed"),
    ("/v6/expression/compositions", "GET", "get_v4_compositions", "expression", "readout",
     None, None, (), ("observatory_worker",), False, "/v4/compositions"),
    ("/v6/expression/history", "GET", "get_v4_history", "expression", "readout",
     None, None, (), ("observatory_worker",), False, "/v4/history"),
    ("/v6/expression/meditations", "GET", "get_v4_meditations", "expression", "readout",
     None, None, (), ("observatory_worker",), False, "/v4/meditations"),

    # ── dreaming ─────────────────────────────────────────────────────────
    ("/v6/dreaming", "GET", "get_v4_dreaming", "dreaming", "readout",
     "dreaming.get_state", None, ("dream_state.bin",), ("dream_state_worker",), False, "/v4/dreaming"),
    ("/v6/dreaming/inbox", "GET", "get_dream_inbox", "dreaming", "readout",
     None, None, (), ("agno_worker",), False, "/v4/dream-inbox"),
    ("/v6/dreaming/meditation-health", "GET", "get_v4_meditation_health", "dreaming", "readout",
     "shm.read_meditation_state", None, ("meditation_state.bin",), ("meditation_worker",), False, "/v4/meditation/health"),
    ("/v6/dreaming/force", "POST", "post_v4_dream_force", "dreaming", "mutation",
     None, "commands.dream_force", (), ("dream_state_worker",), False, "/v4/dream/force"),
    ("/v6/dreaming/meditation/force-trigger", "POST", "post_v4_meditation_force_trigger", "dreaming", "mutation",
     None, "commands.meditation_force_trigger", (), ("meditation_worker",), False, "/v4/meditation/force-trigger"),
    ("/v6/dreaming/meditation/force-end", "POST", "post_v4_meditation_force_end", "dreaming", "mutation",
     None, "commands.meditation_force_end", (), ("meditation_worker",), False, "/v4/meditation/force-end"),

    # ── cognition ────────────────────────────────────────────────────────
    ("/v6/cognition/reasoning", "GET", "get_v4_reasoning", "cognition", "readout",
     "reasoning.get_stats", None, ("reasoning_state.bin",), ("cognitive_worker",), False, "/v4/reasoning"),
    ("/v6/cognition/reasoning-rewards", "GET", "get_v4_reasoning_rewards", "cognition", "readout",
     "reasoning.get_stats", None, ("reasoning_state.bin",), ("cognitive_worker",), False, "/v4/reasoning-rewards"),
    ("/v6/cognition/meta-reasoning", "GET", "get_v4_meta_reasoning", "cognition", "readout",
     "shm.read_meta_reasoning_state", None, ("meta_reasoning_state.bin",), ("cognitive_worker",), False, "/v4/meta-reasoning"),
    ("/v6/cognition/meta-reasoning/audit", "GET", "get_v4_meta_reasoning_audit", "cognition", "readout",
     "shm.read_meta_reasoning_state", None, ("meta_reasoning_state.bin",), ("cognitive_worker",), False, "/v4/meta-reasoning/audit"),
    ("/v6/cognition/meta-reasoning/event-reward", "POST", "post_v4_meta_event_reward", "cognition", "mutation",
     None, "commands.meta_event_reward", (), ("cognitive_worker",), False, "/v4/meta-reasoning/event-reward"),
    ("/v6/cognition/meta-cgn", "GET", "get_v4_meta_cgn", "cognition", "readout",
     "cgn.get_stats", None, ("cgn_engine_state.bin",), ("cgn_worker",), False, "/v4/meta-cgn"),
    ("/v6/cognition/meta-cgn/advisor-conflicts", "GET", "get_v4_meta_cgn_advisor_conflicts", "cognition", "readout",
     "cgn.get_stats", None, ("cgn_engine_state.bin",), ("cgn_worker",), False, "/v4/meta-cgn/advisor-conflicts"),
    ("/v6/cognition/meta-cgn/audit", "GET", "get_v4_meta_cgn_audit", "cognition", "readout",
     "cgn.get_stats", None, ("cgn_engine_state.bin",), ("cgn_worker",), False, "/v4/meta-cgn/audit"),
    ("/v6/cognition/meta-cgn/by-domain", "GET", "get_v4_meta_cgn_by_domain", "cognition", "readout",
     "cgn.get_stats", None, ("cgn_engine_state.bin",), ("cgn_worker",), False, "/v4/meta-cgn/by-domain"),
    ("/v6/cognition/meta-cgn/disagreements", "GET", "get_v4_meta_cgn_disagreements", "cognition", "readout",
     "cgn.get_stats", None, ("cgn_engine_state.bin",), ("cgn_worker",), False, "/v4/meta-cgn/disagreements"),
    ("/v6/cognition/meta-cgn/failsafe-status", "GET", "get_v4_meta_cgn_failsafe_status", "cognition", "readout",
     "cgn.get_stats", None, ("cgn_engine_state.bin",), ("cgn_worker",), False, "/v4/meta-cgn/failsafe-status"),
    ("/v6/cognition/meta-cgn/graduation-readiness", "GET", "get_v4_meta_cgn_graduation_readiness", "cognition", "readout",
     "cgn.get_stats", None, ("cgn_engine_state.bin",), ("cgn_worker",), False, "/v4/meta-cgn/graduation-readiness"),
    ("/v6/cognition/meta-cgn/impasse-status", "GET", "get_v4_meta_cgn_impasse_status", "cognition", "readout",
     "cgn.get_stats", None, ("cgn_engine_state.bin",), ("cgn_worker",), False, "/v4/meta-cgn/impasse-status"),
    ("/v6/cognition/meta-service", "GET", "get_v4_meta_service", "cognition", "readout",
     None, None, (), ("cognitive_worker",), True, "/v4/meta-service"),
    ("/v6/cognition/meta-service/queue", "GET", "get_v4_meta_service_queue", "cognition", "readout",
     None, None, (), ("cognitive_worker",), True, "/v4/meta-service/queue"),
    ("/v6/cognition/meta-service/recruitment", "GET", "get_v4_meta_service_recruitment", "cognition", "readout",
     None, None, (), ("cognitive_worker",), True, "/v4/meta-service/recruitment"),
    ("/v6/cognition/meta-service/rewards", "GET", "get_v4_meta_service_rewards", "cognition", "readout",
     None, None, (), ("cognitive_worker",), True, "/v4/meta-service/rewards"),
    ("/v6/cognition/meta-service/timechain", "GET", "get_v4_meta_service_timechain", "cognition", "readout",
     None, None, (), ("cognitive_worker",), True, "/v4/meta-service/timechain"),
    ("/v6/cognition/meta-teacher", "GET", "get_v4_meta_teacher_status", "cognition", "readout",
     "meta_teacher.get_stats", None, ("meta_teacher_state.bin",), ("cognitive_worker",), False, "/v4/meta-teacher/status"),
    ("/v6/cognition/meta-teacher/critiques", "GET", "get_v4_meta_teacher_critiques", "cognition", "readout",
     "meta_teacher.get_stats", None, ("meta_teacher_state.bin",), ("cognitive_worker",), False, "/v4/meta-teacher/critiques"),
    ("/v6/cognition/meta-teacher/memory", "GET", "get_v4_meta_teacher_memory", "cognition", "readout",
     "meta_teacher.get_stats", None, ("meta_teacher_state.bin",), ("cognitive_worker",), False, "/v4/meta-teacher/memory"),
    ("/v6/cognition/meta-teacher/memory/still-needs-push", "GET", "get_v4_meta_teacher_still_needs_push", "cognition", "readout",
     "meta_teacher.get_stats", None, ("meta_teacher_state.bin",), ("cognitive_worker",), False, "/v4/meta-teacher/memory/still-needs-push"),
    ("/v6/cognition/meta-teacher/maker-info", "GET", "get_v4_meta_teacher_maker_info", "cognition", "readout",
     "meta_teacher.get_stats", None, ("meta_teacher_state.bin",), ("cognitive_worker",), False, "/v4/meta-teacher/maker-info"),
    ("/v6/cognition/meta-teacher/voice", "GET", "get_v4_meta_teacher_voice", "cognition", "readout",
     "meta_teacher.get_stats", None, ("meta_teacher_state.bin",), ("cognitive_worker",), False, "/v4/meta-teacher/voice"),
    ("/v6/cognition/meta-teacher/voice/log", "GET", "get_v4_meta_teacher_voice_log", "cognition", "readout",
     "meta_teacher.get_stats", None, ("meta_teacher_state.bin",), ("cognitive_worker",), False, "/v4/meta-teacher/voice/log"),
    ("/v6/cognition/meta-teacher/peer", "GET", "get_v4_meta_teacher_peer", "cognition", "readout",
     "meta_teacher.get_stats", None, ("meta_teacher_state.bin",), ("cognitive_worker",), False, "/v4/meta-teacher/peer"),
    ("/v6/cognition/meta-teacher/peer/query", "POST", "post_v4_meta_teacher_peer_query", "cognition", "mutation",
     None, "commands.meta_teacher_peer_query", (), ("cognitive_worker",), False, "/v4/meta-teacher/peer/query"),
    ("/v6/cognition/meta-teacher/voice/revert", "POST", "post_v4_meta_teacher_voice_revert", "cognition", "mutation",
     None, "commands.meta_teacher_voice_revert", (), ("cognitive_worker",), False, "/v4/meta-teacher/voice/revert"),
    ("/v6/cognition/meta-outer/enable", "POST", "post_v4_meta_outer_enable", "cognition", "mutation",
     None, "commands.meta_outer_enable", (), ("cognitive_worker",), False, "/v4/meta-outer/enable"),
    ("/v6/cognition/meta-outer/disable", "POST", "post_v4_meta_outer_disable", "cognition", "mutation",
     None, "commands.meta_outer_disable", (), ("cognitive_worker",), False, "/v4/meta-outer/disable"),
    ("/v6/cognition/meta-outer/status", "GET", "get_v4_meta_outer_status", "cognition", "readout",
     None, None, (), ("cognitive_worker",), True, "/v4/meta-outer/status"),
    ("/v6/cognition/meta-outer/recall-test", "GET", "get_v4_meta_outer_recall_test", "cognition", "readout",
     None, None, (), ("cognitive_worker",), True, "/v4/meta-outer/recall-test"),
    ("/v6/cognition/meta-outer/stats", "GET", "get_v4_meta_outer_stats", "cognition", "readout",
     None, None, (), ("cognitive_worker",), True, "/v4/meta-outer/stats"),
    ("/v6/cognition/emot-cgn", "GET", "get_v4_emot_cgn", "cognition", "readout",
     None, None, (), ("emot_cgn_worker",), True, "/v4/emot-cgn"),
    ("/v6/cognition/emot-cgn/graduation-readiness", "GET", "get_v4_emot_cgn_graduation_readiness", "cognition", "readout",
     None, None, (), ("emot_cgn_worker",), True, "/v4/emot-cgn/graduation-readiness"),
    ("/v6/cognition/emot-cgn/audit", "GET", "get_v4_emot_cgn_audit", "cognition", "readout",
     None, None, (), ("emot_cgn_worker",), True, "/v4/emot-cgn/audit"),
    ("/v6/cognition/emot-cgn/force-graduate", "POST", "post_v4_emot_cgn_force_graduate", "cognition", "mutation",
     None, "commands.emot_cgn_force_graduate", (), ("emot_cgn_worker",), False, "/v4/emot-cgn/force-graduate"),
    ("/v6/cognition/emot-cgn/force-shadow", "POST", "post_v4_emot_cgn_force_shadow", "cognition", "mutation",
     None, "commands.emot_cgn_force_shadow", (), ("emot_cgn_worker",), False, "/v4/emot-cgn/force-shadow"),
    ("/v6/cognition/cgn-haov-stats", "GET", "get_v4_cgn_haov_stats", "cognition", "readout",
     "cgn.get_stats", None, ("cgn_engine_state.bin",), ("cgn_worker",), False, "/v4/cgn-haov-stats"),
    ("/v6/cognition/cgn-social-action", "GET", "get_v4_cgn_social_action", "cognition", "readout",
     "cgn.get_stats", None, ("cgn_engine_state.bin",), ("cgn_worker",), False, "/v4/cgn-social-action"),
    ("/v6/cognition/self-reflection", "GET", "get_v4_self_reflection", "cognition", "readout",
     None, None, (), ("cognitive_worker",), True, "/v4/self-reflection"),
    ("/v6/cognition/prediction", "GET", "prediction_diagnostics", "cognition", "readout",
     None, None, (), ("cognitive_worker",), True, "/v4/prediction"),
    ("/v6/cognition/self-exploration", "GET", "get_v4_self_exploration", "cognition", "readout",
     None, None, (), ("cognitive_worker",), True, "/v4/self-exploration"),
    ("/v6/cognition/cognitive-contracts", "GET", "get_v4_cognitive_contracts", "cognition", "readout",
     None, None, (), ("cognitive_worker",), True, "/v4/cognitive-contracts"),
    ("/v6/cognition/coding-explorer", "GET", "get_v4_coding_explorer", "cognition", "readout",
     None, None, (), ("cognitive_worker",), True, "/v4/coding-explorer"),
    ("/v6/cognition/arc-status", "GET", "get_v4_arc_status", "cognition", "readout",
     None, None, (), ("arc_worker",), True, "/v4/arc-status"),
    ("/v6/cognition/arc/goal-ingest", "POST", "post_v4_arc_goal_ingest", "cognition", "mutation",
     None, "commands.arc_goal_ingest", (), ("arc_worker",), False, "/v4/arc/goal-ingest"),
    ("/v6/cognition/kin/emot", "GET", "get_v4_kin_emot", "cognition", "readout",
     None, None, (), ("emot_cgn_worker",), True, "/v4/kin/emot"),

    # ── language ─────────────────────────────────────────────────────────
    ("/v6/language/vocabulary", "GET", "get_v4_vocabulary", "language", "readout",
     "language.get_stats", None, ("language_state.bin",), ("language_worker",), False, "/v4/vocabulary"),
    ("/v6/language/vocabulary/update-learning", "POST", "post_v4_vocabulary_update_learning", "language", "mutation",
     None, "commands.vocabulary_update_learning", (), ("language_worker",), False, "/v4/vocabulary/update-learning"),
    ("/v6/language/grounding", "GET", "get_v4_language_grounding", "language", "readout",
     "language.get_stats", None, ("language_state.bin",), ("language_worker",), False, "/v4/language-grounding"),
    ("/v6/language/knowledge-request", "POST", "post_v4_knowledge_request", "language", "mutation",
     None, "commands.knowledge_request", (), ("memory_worker",), False, "/v4/knowledge-request"),
    ("/v6/language/knowledge-stats", "GET", "get_v4_knowledge_stats", "language", "readout",
     "memory.get_knowledge_graph", None, ("memory_state.bin",), ("memory_worker",), False, "/v4/knowledge-stats"),
    ("/v6/language/knowledge-search", "GET", "get_v4_knowledge_search", "language", "readout",
     None, None, (), ("memory_worker",), True, "/v4/knowledge-search"),

    # ── social ───────────────────────────────────────────────────────────
    ("/v6/social", "GET", "get_v4_social", "social", "readout",
     "social.get_stats", None, ("social_graph_state.bin",), ("social_graph_worker",), False, "/v4/social"),
    ("/v6/social/pressure", "GET", "get_v4_social_pressure", "social", "readout",
     "social.get_stats", None, ("social_graph_state.bin",), ("social_graph_worker",), False, "/v4/social-pressure"),
    ("/v6/social/relief", "POST", "post_v4_social_relief", "social", "mutation",
     None, "commands.social_relief", (), ("social_graph_worker",), False, "/v4/social-relief"),
    ("/v6/social/signal-concept", "POST", "post_v4_signal_concept", "social", "mutation",
     None, "commands.signal_concept", (), ("social_graph_worker",), False, "/v4/signal-concept"),
    ("/v6/social/signal-co-occurrence", "POST", "post_v4_signal_co_occurrence", "social", "mutation",
     None, "commands.signal_co_occurrence", (), ("social_graph_worker",), False, "/v4/signal-co-occurrence"),
    ("/v6/social/perception", "POST", "post_v4_social_perception", "social", "mutation",
     None, "commands.social_perception", (), ("social_worker",), False, "/v4/social-perception"),
    ("/v6/social/delegate", "POST", "post_v4_social_delegate", "social", "mutation",
     None, "commands.social_delegate", (), ("social_worker",), False, "/v4/social-delegate"),
    ("/v6/social/delegate-queue", "GET", "get_v4_social_delegate_queue", "social", "readout",
     None, None, (), ("social_worker",), True, "/v4/social-delegate-queue"),
    ("/v6/social/persona-telemetry", "GET", "get_v4_persona_telemetry", "social", "readout",
     "social.get_stats", None, ("social_graph_state.bin",), ("social_graph_worker",), False, "/v4/persona-telemetry"),
    ("/v6/social/persona-profiles", "GET", "get_v4_persona_profiles", "social", "readout",
     "social.get_stats", None, ("social_graph_state.bin",), ("social_graph_worker",), False, "/v4/persona-profiles"),
    ("/v6/social/compose-reply", "POST", "compose_reply", "social", "mutation",
     None, "commands.compose_reply", (), ("social_worker",), False, "/v4/compose-reply"),
    ("/v6/social/community-engagement-stats", "GET", "get_v4_community_engagement_stats", "social", "readout",
     None, None, (), ("social_worker",), True, "/v4/community-engagement-stats"),
    ("/v6/social/kin-signature", "GET", "get_kin_signature", "social", "readout",
     None, None, (), ("social_worker",), True, "/v4/kin-signature"),
    ("/v6/social/kin-exchange", "POST", "kin_exchange", "social", "mutation",
     None, "commands.kin_exchange", (), ("social_worker",), False, "/v4/kin-exchange"),
    ("/v6/social/kin-society", "GET", "kin_society", "social", "readout",
     None, None, (), ("social_worker",), True, "/v4/kin-society"),

    # ── metabolism ───────────────────────────────────────────────────────
    ("/v6/metabolism/evaluate-gate", "GET", "metabolism_evaluate_gate", "metabolism", "readout",
     "shm.read_metabolism_state", None, ("metabolism_state.bin",), ("metabolism_worker",), True, "/v4/metabolism/evaluate-gate"),
    ("/v6/metabolism/gate-status", "GET", "metabolism_gate_status", "metabolism", "readout",
     "shm.read_metabolism_state", None, ("metabolism_state.bin",), ("metabolism_worker",), False, "/v4/metabolism/gate-status"),
    ("/v6/metabolism/sovereignty-status", "GET", "sovereignty_status", "metabolism", "readout",
     "soul.get_active_directives", None, ("soul_state.bin",), ("sovereignty_worker",), False, "/v4/sovereignty/status"),
    ("/v6/metabolism/metabolic-state", "GET", "get_v4_metabolic_state", "metabolism", "readout",
     "shm.read_metabolism_state", None, ("metabolism_state.bin",), ("metabolism_worker",), False, "/v4/metabolic-state"),

    # ── backup ───────────────────────────────────────────────────────────
    ("/v6/backup/verify", "GET", "get_v4_backup_verify", "backup", "readout",
     None, None, (), ("backup_worker",), True, "/v4/backup/verify"),
    ("/v6/backup/status", "GET", "get_v4_backup_status", "backup", "readout",
     None, None, (), ("backup_worker",), True, "/v4/backup/status"),
    ("/v6/backup/history", "GET", "get_v4_backup_history", "backup", "readout",
     None, None, (), ("backup_worker",), True, "/v4/backup/history"),
    ("/v6/backup/wallet-runway", "GET", "get_v4_backup_wallet_runway", "backup", "readout",
     None, None, (), ("backup_worker",), True, "/v4/backup/wallet-runway"),
    ("/v6/backup/manifest", "GET", "get_v4_backup_manifest", "backup", "readout",
     None, None, (), ("backup_worker",), True, "/v4/backup/manifest"),
    ("/v6/backup/trigger", "POST", "post_v4_backup_trigger", "backup", "mutation",
     None, "commands.backup_trigger", (), ("backup_worker",), False, "/v4/backup/trigger"),

    # ── timechain ────────────────────────────────────────────────────────
    ("/v6/timechain/status", "GET", "get_v4_timechain_status", "timechain", "readout",
     "timechain.get_status", None, ("timechain_state.bin",), ("timechain_worker",), False, "/v4/timechain/status"),
    ("/v6/timechain/blocks", "GET", "get_v4_timechain_blocks", "timechain", "readout",
     None, None, (), ("timechain_worker",), True, "/v4/timechain/blocks"),
    ("/v6/timechain/verify", "GET", "get_v4_timechain_verify", "timechain", "readout",
     None, None, (), ("timechain_worker",), True, "/v4/timechain/verify"),
    ("/v6/timechain/pot-stats", "GET", "get_v4_timechain_pot_stats", "timechain", "readout",
     None, None, (), ("timechain_worker",), True, "/v4/timechain/pot-stats"),
    ("/v6/timechain/fork-tree", "GET", "get_v4_timechain_fork_tree", "timechain", "readout",
     None, None, (), ("timechain_worker",), True, "/v4/timechain/fork-tree"),
    ("/v6/timechain/contracts", "GET", "get_v4_timechain_contracts", "timechain", "readout",
     None, None, (), ("timechain_worker",), True, "/v4/timechain/contracts"),
    ("/v6/timechain/contracts/stats", "GET", "get_v4_contracts_stats", "timechain", "readout",
     None, None, (), ("timechain_worker",), True, "/v4/timechain/contracts/stats"),
    ("/v6/timechain/contracts/pending", "GET", "get_v4_contracts_pending", "timechain", "readout",
     None, None, (), ("timechain_worker",), True, "/v4/timechain/contracts/pending"),
    ("/v6/timechain/contracts/approve", "POST", "post_v4_contracts_approve", "timechain", "mutation",
     None, "commands.contracts_approve", (), ("timechain_worker",), False, "/v4/timechain/contracts/approve"),
    ("/v6/timechain/contracts/veto", "POST", "post_v4_contracts_veto", "timechain", "mutation",
     None, "commands.contracts_veto", (), ("timechain_worker",), False, "/v4/timechain/contracts/veto"),
    ("/v6/timechain/block", "GET", "get_v4_timechain_block", "timechain", "readout",
     None, None, (), ("timechain_worker",), True, "/v4/timechain/block"),
    ("/v6/timechain/verify/{height}", "GET", "get_v4_timechain_verify_block", "timechain", "readout",
     None, None, (), ("timechain_worker",), True, "/v4/timechain/verify/{height}"),
    ("/v6/timechain/test-commit", "POST", "post_v4_timechain_test_commit", "timechain", "mutation",
     None, "commands.timechain_test_commit", (), ("timechain_worker",), False, "/v4/timechain/test-commit"),
    ("/v6/timechain/backup-status", "GET", "get_v4_timechain_backup_status", "timechain", "readout",
     None, None, (), ("timechain_worker",), True, "/v4/timechain/backup-status"),
    ("/v6/timechain/verify-memories", "GET", "get_v4_timechain_verify_memories", "timechain", "readout",
     None, None, (), ("timechain_worker",), True, "/v4/timechain/verify-memories"),
    ("/v6/timechain/backup-now", "POST", "post_v4_timechain_backup_now", "timechain", "mutation",
     None, "commands.timechain_backup_now", (), ("timechain_worker",), False, "/v4/timechain/backup-now"),
    ("/v6/timechain/developmental-timeline", "GET", "get_v4_developmental_timeline", "timechain", "readout",
     None, None, (), ("timechain_worker",), True, "/v4/developmental-timeline"),

    # ── reflexes ─────────────────────────────────────────────────────────
    ("/v6/reflexes", "GET", "get_v4_reflexes", "reflexes", "readout",
     "shm.read_reflex_state", None, ("reflex_state.bin",), ("reflex_worker",), False, "/v4/reflexes"),
    ("/v6/reflexes/history", "GET", "get_v4_reflex_history", "reflexes", "readout",
     None, None, (), ("observatory_worker",), False, "/v4/reflexes/history"),

    # ── system ───────────────────────────────────────────────────────────
    ("/v6/system/thread-pool", "GET", "thread_pool_stats", "system", "readout",
     None, None, (), ("kernel",), True, "/v4/thread-pool"),
    ("/v6/system/db-contention", "GET", "db_contention", "system", "readout",
     None, None, (), ("kernel",), True, "/v4/db-contention"),
    ("/v6/system/bus-health", "GET", "bus_health", "system", "readout",
     None, None, (), ("kernel",), True, "/v4/bus-health"),
    ("/v6/system/imw-health", "GET", "imw_health", "system", "readout",
     None, None, (), ("kernel",), True, "/v4/imw-health"),
    ("/v6/system/warning-monitor", "GET", "warning_monitor", "system", "readout",
     None, None, (), ("warning_monitor_worker",), True, "/v4/warning-monitor"),
    ("/v6/system/search-pipeline/health", "GET", "search_pipeline_health", "system", "readout",
     None, None, (), ("memory_worker",), True, "/v4/search-pipeline/health"),
    ("/v6/system/search-pipeline/backend/{name}", "GET", "search_pipeline_backend", "system", "readout",
     None, None, (), ("memory_worker",), True, "/v4/search-pipeline/backend/{name}"),
    ("/v6/system/search-pipeline/budget-reset", "POST", "search_pipeline_budget_reset", "system", "mutation",
     None, "commands.search_pipeline_budget_reset", (), ("memory_worker",), False, "/v4/search-pipeline/budget-reset"),
    ("/v6/system/search-pipeline/learning", "GET", "search_pipeline_learning", "system", "readout",
     None, None, (), ("memory_worker",), True, "/v4/search-pipeline/learning"),
    ("/v6/system/debug/dim-sources", "GET", "get_v4_debug_dim_sources", "system", "readout",
     None, None, (), ("kernel",), True, "/v4/debug/dim-sources"),
    ("/v6/system/guardian", "GET", "get_v3_guardian", "system", "readout",
     "guardian.get_status", None, ("guardian_state.bin",), ("guardian",), True, "/v3/guardian"),
    ("/v6/system/guardian/start/{module_name}", "POST", "start_v3_module", "system", "mutation",
     None, "commands.guardian_start", (), ("guardian",), False, "/v3/guardian/start/{module_name}"),
    ("/v6/system/guardian/enable/{module_name}", "POST", "enable_v3_module", "system", "mutation",
     None, "commands.guardian_start", (), ("guardian",), False, "/v3/guardian/enable/{module_name}"),
    ("/v6/system/timeseries", "GET", "get_v4_timeseries", "system", "readout",
     None, None, (), ("observatory_worker",), False, "/v4/timeseries"),
    ("/v6/system/timeseries/metrics", "GET", "get_v4_timeseries_metrics", "system", "readout",
     None, None, (), ("observatory_worker",), False, "/v4/timeseries/metrics"),
    ("/v6/system/experience-stimulus", "POST", "post_experience_stimulus", "system", "mutation",
     None, "commands.experience_stimulus", (), ("cognitive_worker",), False, "/v4/experience-stimulus"),

    # ── admin ────────────────────────────────────────────────────────────
    ("/v6/admin/reload", "POST", "post_v4_reload", "admin", "admin",
     None, "commands.reload", (), ("kernel",), False, "/v4/reload"),
    ("/v6/admin/reload-api", "POST", "post_v4_reload_api", "admin", "admin",
     None, "commands.reload_api", (), ("kernel",), False, "/v4/reload-api"),
    ("/v6/admin/reload-config", "POST", "post_v4_reload_config", "admin", "admin",
     None, "commands.reload_config", (), ("kernel",), False, "/v4/reload-config"),
    ("/v6/admin/restart-module/{name}", "POST", "post_v4_restart_module", "admin", "admin",
     None, "commands.restart_module", (), ("guardian",), False, "/v4/admin/restart-module/{name}"),
    ("/v6/admin/reload-module/{name}", "POST", "post_v4_reload_module", "admin", "admin",
     None, "commands.reload_module", (), ("guardian",), False, "/v4/admin/reload-module/{name}"),
    ("/v6/admin/msl/reset-homeostasis", "POST", "post_v4_msl_reset_homeostasis", "admin", "admin",
     None, "commands.msl_reset_homeostasis", (), ("cognitive_worker",), False, "/v4/admin/msl/reset-homeostasis"),
    ("/v6/admin/memory-profile", "GET", "get_v4_admin_memory_profile", "admin", "admin",
     None, None, (), ("kernel",), True, "/v4/admin/memory-profile"),
    ("/v6/admin/heap-dump", "GET", "get_v4_admin_heap_dump", "admin", "admin",
     None, None, (), ("kernel",), True, "/v4/admin/heap-dump"),
    ("/v6/admin/parent-threads", "GET", "get_v4_admin_parent_threads", "admin", "admin",
     None, None, (), ("kernel",), True, "/v4/admin/parent-threads"),

    # ── maker ────────────────────────────────────────────────────────────
    ("/v6/maker/proposals", "GET", "get_maker_proposals", "maker", "readout",
     None, None, (), ("maker_worker",), True, "/v4/maker/proposals"),
    ("/v6/maker/proposals/{proposal_id}", "GET", "get_maker_proposal", "maker", "readout",
     None, None, (), ("maker_worker",), True, "/v4/maker/proposals/{proposal_id}"),
    ("/v6/maker/proposals/{proposal_id}/approve", "POST", "approve_maker_proposal", "maker", "mutation",
     None, "commands.maker_proposal_approve", (), ("maker_worker",), False, "/v4/maker/proposals/{proposal_id}/approve"),
    ("/v6/maker/proposals/{proposal_id}/decline", "POST", "decline_maker_proposal", "maker", "mutation",
     None, "commands.maker_proposal_decline", (), ("maker_worker",), False, "/v4/maker/proposals/{proposal_id}/decline"),
    ("/v6/maker/dialogue-history", "GET", "get_maker_dialogue_history", "maker", "readout",
     None, None, (), ("maker_worker",), True, "/v4/maker/dialogue-history"),

    # ── synthesis — Phase 4 §P4.I (D-SPEC-128 forthcoming) ───────────────
    # Concept-spine readouts. Source = data/knowledge_graph.kuzu (read-only
    # cross-process open). Producer = synthesis_worker. No legacy /v4 path
    # replaced (new in P4).
    ("/v6/synthesis/concepts", "GET", "get_v6_synthesis_concepts", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/concepts/heatmap", "GET", "get_v6_synthesis_concepts_heatmap",
     "synthesis", "readout", None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/concepts/{concept_id}", "GET", "get_v6_synthesis_concept",
     "synthesis", "readout", None, None, (), ("synthesis_worker",), False, None),

    # ── synthesis — Phase 5 §P5.I (D-SPEC-PHASE5 forthcoming) ────────────
    # Hypothesis-fork lifecycle readouts. Source = data/forks_snapshot.json
    # (synthesis_worker sole writer per INV-Syn-8). No legacy /v4 path.
    # /v6/synthesis/forks/tombstones MUST be declared BEFORE the {fork_id}
    # route so FastAPI's path matcher catches "tombstones" before the
    # generic parameter route.
    ("/v6/synthesis/forks", "GET", "get_v6_synthesis_forks", "synthesis",
     "readout", None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/forks/summary", "GET", "get_v6_synthesis_fork_summary",
     "synthesis", "readout", None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/forks/tombstones", "GET",
     "get_v6_synthesis_fork_tombstones", "synthesis", "readout", None, None,
     (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/forks/{fork_id}", "GET", "get_v6_synthesis_fork",
     "synthesis", "readout", None, None, (), ("synthesis_worker",), False, None),

    # ── synthesis — Phase 5 §P5.A-G fork-lifecycle write surface ─────────
    # Fire-and-forget POST → SYNTHESIS_FORK_COMMAND bus event → synthesis_worker
    # handles + eager-exports forks_snapshot.json. Order: sweep + each {fork_id}
    # sub-route declared BEFORE the bare {fork_id} POST so static prefixes win
    # FastAPI's path matching.
    ("/v6/synthesis/forks", "POST", "post_v6_synthesis_forks", "synthesis",
     "write", None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/forks/sweep", "POST", "post_v6_synthesis_fork_sweep",
     "synthesis", "write", None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/forks/{fork_id}/record-exploration-tx", "POST",
     "post_v6_synthesis_fork_record_exploration", "synthesis", "write",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/forks/{fork_id}/graduate-manual", "POST",
     "post_v6_synthesis_fork_graduate_manual", "synthesis", "write",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/forks/{fork_id}/abandon", "POST",
     "post_v6_synthesis_fork_abandon", "synthesis", "write",
     None, None, (), ("synthesis_worker",), False, None),

    # ── synthesis — Phase 6 §P6.K oracle + proof readout ─────────────────
    # All 5 routes read data/oracles_snapshot.json (exported every 60s by
    # synthesis_worker via OracleSnapshotExporter); soft-fail to empty
    # payload + snapshot status when the file is missing/stale/corrupt.
    ("/v6/synthesis/oracles/router", "GET",
     "get_v6_synthesis_oracles_router", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/oracles/recent", "GET",
     "get_v6_synthesis_oracles_recent", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/oracles/coverage", "GET",
     "get_v6_synthesis_oracles_coverage", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/oracles/budget", "GET",
     "get_v6_synthesis_oracles_budget", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/proofs/recent", "GET",
     "get_v6_synthesis_proofs_recent", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    # Phase 7 (D-SPEC-PHASE7, 2026-05-27): ACT-R working-memory buffer
    # surface. Reads buffers_snapshot.json (written atomically by
    # synthesis_worker via ActrBufferStore — sole writer per INV-Syn-16).
    # Soft-fail to empty payload + snapshot status on missing/stale/corrupt.
    ("/v6/synthesis/buffers/list_chats", "GET",
     "get_v6_synthesis_buffers_list_chats", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/buffers/read", "GET",
     "get_v6_synthesis_buffers_read", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/buffers/recent_writes", "GET",
     "get_v6_synthesis_buffers_recent_writes", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/buffers/snapshot", "GET",
     "get_v6_synthesis_buffers_snapshot", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    # Phase 8 (D-SPEC-PHASE8, 2026-05-27): procedural skill miner readout.
    # Reads skills_snapshot.json (written atomically by synthesis_worker
    # via ProceduralSkillStore — sole writer per INV-Syn-19). Coverage
    # route derives §A.6 readout from the chain index (read-only sqlite).
    ("/v6/synthesis/skills", "GET",
     "get_v6_synthesis_skills_list", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/skills/detail", "GET",
     "get_v6_synthesis_skills_detail", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/skills/recent", "GET",
     "get_v6_synthesis_skills_recent", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/skills/coverage", "GET",
     "get_v6_synthesis_skills_coverage", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    # Phase 10 (D-SPEC-PHASE10) — observatory + metrics. Read-only over
    # data/synthesis_metrics_snapshot.json (INV-Syn-25, observation-only).
    ("/v6/synthesis/metrics", "GET",
     "get_v6_synthesis_metrics", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/metrics/sovereignty", "GET",
     "get_v6_synthesis_metrics_sovereignty", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/metrics/groundedness", "GET",
     "get_v6_synthesis_metrics_groundedness", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/metrics/retrieval", "GET",
     "get_v6_synthesis_metrics_retrieval", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    ("/v6/synthesis/metrics/chain-growth", "GET",
     "get_v6_synthesis_metrics_chain_growth", "synthesis", "readout",
     None, None, (), ("synthesis_worker",), False, None),
    # Phase 9 INV-Syn-24 — explicit Tier-2 user-feedback producer. POST →
    # publishes USER_FEEDBACK_SIGNAL; synthesis_worker applies the override.
    ("/v6/synthesis/feedback", "POST",
     "post_v6_synthesis_feedback", "synthesis", "write",
     None, None, (), ("synthesis_worker",), False, None),
)


def _wire() -> None:
    """Mount every ROUTE_TABLE row on the v6 router (re-using the dashboard
    handler function so signatures/params are preserved) + register its
    manifest row. Idempotent-safe under api hot-reload (REGISTRY is cleared
    when v6_manifest reloads first)."""
    for (path, method, func_name, group, kind, accessor, command,
         slots, producers, rpc, replaces) in _T:
        fn = getattr(_dash, func_name)
        router.add_api_route(path, fn, methods=[method])
        _m.register(RouteSpec(
            path=path, method=method, group=group, kind=kind,
            summary=(fn.__doc__ or "").strip().split("\n")[0][:160],
            accessor=accessor, command=command,
            shm_slots=tuple(slots), producers=tuple(producers),
            rpc=rpc, replaces=(replaces,) if replaces else (),
        ))


_wire()


# Phase 11 §11.I.5 / D-SPEC-141 (Chunk 11J) — register the two new
# readouts in the manifest so /v6/manifest's drift check stays clean.
# The shm_slots/producers columns make the lineage explicit:
#   /v6/readiness reads titan_hcl_state.bin (orchestrator/titan_hcl) +
#     module_<name>_state.bin × N (each worker is the G21 writer of its
#     own slot per §11.I.5).
#   /v6/errors reads module_<name>_state.bin × N (same writers) — and
#     forward-compats `errors_state.bin` (observatory_worker) once that
#     writer ships.
_m.register(RouteSpec(
    path="/v6/readiness", method="GET", group="phase11", kind="readout",
    summary="Phase 11 fleet readiness snapshot — titan_hcl_state.bin + every "
            "module_<name>_state.bin per §11.I.5.",
    accessor=None, command=None,
    shm_slots=("titan_hcl_state.bin", "module_<name>_state.bin"),
    producers=("titan_hcl", "<each_worker>"), rpc=False, replaces=(),
))
_m.register(RouteSpec(
    path="/v6/errors", method="GET", group="phase11", kind="readout",
    summary="Phase 11 ModuleError envelope readout — per-module last_error "
            "from SHM slots + forward-compat errors_state.bin history.",
    accessor=None, command=None,
    shm_slots=("module_<name>_state.bin", "errors_state.bin"),
    producers=("<each_worker>", "observatory_worker"),
    rpc=False, replaces=(),
))


# ── /v6/manifest — the route→accessor→slot→producer source-of-truth ──────────
@router.get("/v6/manifest")
async def get_v6_manifest(request: Request) -> JSONResponse:
    """Runtime introspection of the v6 manifest + live SHM-slot freshness.

    The debugging superpower: "data X not loading" → find X's route row → read
    `freshness_s` of its `shm_slots` → check the `producers` worker. One lookup.
    Cross-checks that every registered FastAPI v6 route has a manifest row (and
    vice-versa) so the doc can never silently drift from the live router.
    """
    titan_state = getattr(request.app.state, "titan_state", None)
    shm = getattr(titan_state, "shm", None)
    rows = _m.as_rows()

    freshness: dict[str, float | None] = {}
    if shm is not None:
        for row in rows:
            for slot in row["shm_slots"]:
                if slot not in freshness:
                    freshness[slot] = _slot_age_seconds(shm, slot)
    for row in rows:
        ages = [freshness.get(s) for s in row["shm_slots"]]
        ages = [a for a in ages if a is not None]
        row["freshness_s"] = round(min(ages), 3) if ages else None

    live_paths = set()
    for r in request.app.routes:
        rp = getattr(r, "path", "")
        methods = getattr(r, "methods", None)
        if rp.startswith("/v6") and methods:
            for mth in methods:
                if mth in ("GET", "POST", "PUT", "DELETE"):
                    live_paths.add((rp, mth))
    manifest_paths = {(row["route"], row["method"]) for row in rows}
    live_only = sorted(f"{m} {p}" for (p, m) in live_paths
                       if (p, m) not in manifest_paths and p != "/v6/manifest")
    manifest_only = sorted(f"{m} {p}" for (p, m) in manifest_paths
                           if (p, m) not in live_paths)

    return JSONResponse({
        "ok": True,
        "groups": _m.groups(),
        "route_count": len(rows),
        "routes": rows,
        "drift": {
            "live_routes_without_manifest_row": live_only,
            "manifest_rows_without_live_route": manifest_only,
            "in_sync": not live_only and not manifest_only,
        },
    })


def _slot_age_seconds(shm, slot: str) -> float | None:
    """Best-effort age (s) of a named SHM slot via the reader bank's meta."""
    try:
        meta_fn = getattr(shm, "slot_age_seconds", None)
        if callable(meta_fn):
            return meta_fn(slot)
    except Exception:
        pass
    return None


# ── /v6/readiness — Phase 11 §11.I.5 / D-SPEC-141 (Chunk 11J) ────────────────


@router.get("/v6/readiness")
async def get_v6_readiness(request: Request) -> JSONResponse:
    """Phase 11 §11.I.5 — fleet readiness snapshot.

    Reads:
      * `titan_hcl_state.bin` (orchestrator-owned, G21 single-writer)
        for fleet_ready / fleet_optional_ready / boot_phase / counts /
        timestamps.
      * Every per-module `module_<name>_state.bin` slot via
        `ModuleStateReaderBank` — returns state, last_heartbeat,
        last_probe_result, last_error per module.

    No bus call, no orchestrator round-trip: all SHM reads, sub-ms
    latency, safe to poll at 1Hz from the Observatory.
    """
    from titan_hcl.core.module_state import ModuleStateReaderBank
    from titan_hcl.core.state_registry import resolve_shm_root, resolve_titan_id
    from titan_hcl.core.titan_hcl_state import TitanHclStateReader

    titan_id = resolve_titan_id()
    body: dict[str, object] = {"ok": True}

    # 1. Orchestrator-owned slot (titan_hcl_state.bin).
    try:
        fleet_reader = TitanHclStateReader(titan_id=titan_id)
        fleet_entry = fleet_reader.read()
        fleet_reader.close()
    except Exception as e:  # noqa: BLE001
        fleet_entry = None
        body["fleet_read_error"] = str(e)
    if fleet_entry is not None:
        body["fleet"] = fleet_entry.as_wire_dict()
    else:
        body["fleet"] = None

    # 2. Per-module SHM slots — G18-aligned discovery: SHM is the source of
    #    truth (per locked D1 Phase 11 §11.I.5). We discover the module set
    #    by scanning `/dev/shm/titan_<id>/module_<name>_state.bin` — every
    #    worker that has booted (state≥"starting") will have its slot
    #    present. This avoids two failure modes that the prior code hit live:
    #      (a) `_modules.keys()` via the api-side kernel-RPC proxy raises
    #          `MethodNotExposed` under Phase 6 D-SPEC-135 process split.
    #      (b) the manifest-producer fallback included literal placeholder
    #          strings like `"<each_worker>"` because the manifest registers
    #          per-route producer columns containing template tokens, not
    #          a canonical module roster.
    #
    #    Workers that haven't booted yet contribute no slot — they're listed
    #    as `state="not_booted"` via the manifest's discovered producer set,
    #    minus any placeholder tokens (filtered: angle-bracket names).
    try:
        shm_root = resolve_shm_root(titan_id)
        slot_names: list[str] = []
        if shm_root.exists():
            for p in shm_root.glob("module_*_state.bin"):
                base = p.name
                # `module_<name>_state.bin` → extract `<name>`
                if base.startswith("module_") and base.endswith("_state.bin"):
                    name = base[len("module_"):-len("_state.bin")]
                    if name:
                        slot_names.append(name)
        slot_names_set = set(slot_names)
    except Exception:  # noqa: BLE001
        slot_names = []
        slot_names_set = set()

    # Authoritative "expected" roster — the orchestrator publishes the canonical
    # module set (name → boot_priority) into titan_hcl_state.bin (§11.I.5, schema
    # v2). A roster module with no live slot is genuinely not_booted. This
    # REPLACES the prior API-route-manifest producer union, which polluted
    # not_booted with phantom names (rust substrate procs, kernel peers, and
    # `_worker`-suffixed aliases of modules already running under their short
    # name). Graceful fallback: if the roster is absent (writer not yet up, or a
    # pre-v2 slot still in SHM mid-rollout), report only live slots — never
    # resurrect the phantom-prone manifest union.
    roster_map: dict[str, str] = {}
    if fleet_entry is not None:
        roster_map = {str(n): str(p) for n, p in (fleet_entry.roster or ())}

    all_module_names: list[str] = sorted(slot_names_set | set(roster_map))

    bank = ModuleStateReaderBank(titan_id=titan_id)
    modules_payload: list[dict[str, object]] = []
    running_count = 0
    booted_count = 0
    starting_count = 0
    unhealthy_count = 0
    not_booted_count = 0
    try:
        for name in all_module_names:
            entry = None
            if name in slot_names_set:
                try:
                    entry = bank.read(name)
                except Exception:  # noqa: BLE001
                    entry = None
            if entry is None:
                nb_payload: dict[str, object] = {
                    "name": name, "state": "not_booted"}
                if name in roster_map:
                    nb_payload["boot_priority"] = roster_map[name]
                modules_payload.append(nb_payload)
                not_booted_count += 1
            else:
                payload = entry.as_wire_dict()
                modules_payload.append(payload)
                st = str(payload.get("state", ""))
                if st == "running":
                    running_count += 1
                elif st == "booted":
                    booted_count += 1
                elif st == "starting":
                    starting_count += 1
                elif st in ("unhealthy", "crashed", "disabled"):
                    unhealthy_count += 1
    finally:
        bank.close()
    body["modules"] = modules_payload
    body["module_count"] = len(modules_payload)
    body["module_running_count"] = running_count
    body["module_state_summary"] = {
        "running": running_count,
        "booted": booted_count,
        "starting": starting_count,
        "unhealthy_or_crashed": unhealthy_count,
        "not_booted": not_booted_count,
    }

    return JSONResponse(body)


# ── /v6/errors — Phase 11 §11.I.5 / D-SPEC-141 (Chunk 11J) ──────────────────


@router.get("/v6/errors")
async def get_v6_errors(request: Request) -> JSONResponse:
    """Phase 11 §11.I.5 — queryable ModuleError envelope history.

    Each per-module SHM slot carries its `last_error` ModuleError
    envelope (when present). This route surfaces the snapshot of every
    module that currently reports a non-null `last_error`, sorted by
    error timestamp (descending — newest first).

    The fleet-wide error timeline (history beyond `last_error`) lands
    once `observatory_worker` ships its `errors_state.bin` writer per
    §11.I.5 — that source then folds into this same route under a
    `history: [...]` key. For now the surface is per-module
    most-recent-error: a useful operator readout that maps 1:1 to the
    SHM-direct contract Phase 11 standardizes.
    """
    from titan_hcl.core.module_state import ModuleStateReaderBank
    from titan_hcl.core.state_registry import resolve_shm_root, resolve_titan_id

    titan_id = resolve_titan_id()

    # G18-aligned: discover modules via SHM slot scan (same as /v6/readiness
    # above — single source of truth). Filter manifest-derived candidates
    # to exclude placeholder template tokens like `<each_worker>`.
    try:
        shm_root = resolve_shm_root(titan_id)
        slot_names: list[str] = []
        if shm_root.exists():
            for p in shm_root.glob("module_*_state.bin"):
                base = p.name
                if base.startswith("module_") and base.endswith("_state.bin"):
                    name = base[len("module_"):-len("_state.bin")]
                    if name:
                        slot_names.append(name)
    except Exception:  # noqa: BLE001
        slot_names = []
    module_names: list[str] = sorted(set(slot_names))

    bank = ModuleStateReaderBank(titan_id=titan_id)
    errors: list[dict[str, object]] = []
    try:
        for name in module_names:
            try:
                entry = bank.read(name)
            except Exception:  # noqa: BLE001
                continue
            if entry is None or entry.last_error is None:
                continue
            envelope = entry.last_error.as_wire_dict()
            envelope["state"] = entry.state
            envelope["pid"] = entry.pid
            envelope["restart_count"] = entry.restart_count
            envelope["error_count_24h"] = entry.error_count_24h
            errors.append(envelope)
    finally:
        bank.close()
    # Newest first by envelope ts.
    errors.sort(key=lambda e: float(e.get("ts", 0.0)), reverse=True)

    return JSONResponse({
        "ok": True,
        "error_count": len(errors),
        "errors": errors,
        # Forward-compat placeholder: when observatory_worker lands its
        # errors_state.bin writer per §11.I.5, this key carries the
        # rolling history beyond per-module last_error.
        "history": [],
    })
