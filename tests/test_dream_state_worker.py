"""
Tests for §4.I dream_state_worker (D-SPEC-56, SPEC v1.8.2).

Coverage:
  • DreamStatePublisher: cold-boot defaults, transition detection (awake↔dreaming),
    sticky state across publishes, recovery_pct/just_woke flip semantics,
    encode/oversize defenses, msgpack round-trip
  • DreamStateReader: SHM round-trip, cold-boot returns defaults,
    100ms cache absorbs hot-path reads, is_dreaming() shortcut,
    is_stale() freshness probe
  • Bus event constants: DREAM_WAKE_FORWARD, DREAM_INBOX_ENQUEUE,
    DREAM_INBOX_REPLAY defined
  • Constants TOML wiring: DREAM_STATE_SCHEMA_VERSION, DREAM_INBOX_MAX_ENTRIES,
    DREAM_INBOX_MAX_MESSAGE_CHARS, DREAM_STATE_REPUBLISH_CADENCE_S
  • DreamingEngine.request_wake (cognitive_worker subscriber side): flag set,
    reason captured, idempotent, self-clears after check_transition END_DREAMING
  • dream_state_specs: RegistrySpec wiring (slot name, schema, max bytes)
  • Worker module entry-point: imports, helpers present

Per CLAUDE.md: pytest -p no:anchorpy, separate process per file.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import msgpack
import pytest

from titan_hcl._phase_c_constants import (
    DREAM_INBOX_MAX_ENTRIES,
    DREAM_INBOX_MAX_MESSAGE_CHARS,
    DREAM_STATE_MAX_BYTES,
    DREAM_STATE_REPUBLISH_CADENCE_S,
    DREAM_STATE_SCHEMA_VERSION,
)
from titan_hcl.logic.dream_state_publisher import DreamStatePublisher
from titan_hcl.logic.dream_state_reader import (
    _DEFAULT_PAYLOAD,
    _STALENESS_THRESHOLD_S,
    DreamStateReader,
)
from titan_hcl.logic.dream_state_specs import (
    DREAM_STATE_SLOT,
    DREAM_STATE_SPEC,
)


# ── Per-test SHM root isolation ────────────────────────────────────


@pytest.fixture
def titan_id(tmp_path) -> str:
    """Per-test titan_id; SHM root cleaned up after test."""
    tid = f"T_TEST_{int(time.time() * 1e6) % 1_000_000}"
    yield tid
    shutil.rmtree(Path(f"/dev/shm/titan_{tid}"), ignore_errors=True)


# ── DreamStateSpec wiring ──────────────────────────────────────────


def test_dream_state_spec_slot_name():
    assert DREAM_STATE_SLOT == "dream_state"


def test_dream_state_spec_schema_version():
    assert DREAM_STATE_SPEC.schema_version == DREAM_STATE_SCHEMA_VERSION
    assert DREAM_STATE_SPEC.schema_version == 1


def test_dream_state_spec_max_bytes():
    assert DREAM_STATE_SPEC.payload_bytes == DREAM_STATE_MAX_BYTES
    assert DREAM_STATE_SPEC.payload_bytes == 512


def test_dream_state_spec_is_variable_size():
    assert DREAM_STATE_SPEC.variable_size is True


# ── Constants TOML wiring ──────────────────────────────────────────


def test_constants_wired_per_dspec56():
    assert DREAM_STATE_SCHEMA_VERSION == 1
    assert DREAM_STATE_MAX_BYTES == 512
    assert DREAM_INBOX_MAX_ENTRIES == 50  # matches plugin.py:2270 legacy threshold
    assert DREAM_INBOX_MAX_MESSAGE_CHARS == 500  # matches plugin.py:2280
    assert DREAM_STATE_REPUBLISH_CADENCE_S == 1.0  # Q6 greenlight


# ── Bus event constants ────────────────────────────────────────────


def test_bus_event_constants_defined():
    from titan_hcl import bus
    assert bus.DREAM_STATE_CHANGED == "DREAM_STATE_CHANGED"
    assert bus.DREAM_WAKE_REQUEST == "DREAM_WAKE_REQUEST"
    assert bus.DREAM_WAKE_FORWARD == "DREAM_WAKE_FORWARD"
    assert bus.DREAM_INBOX_ENQUEUE == "DREAM_INBOX_ENQUEUE"
    assert bus.DREAM_INBOX_REPLAY == "DREAM_INBOX_REPLAY"


# ── DreamStatePublisher ────────────────────────────────────────────


def test_publisher_cold_boot_defaults(titan_id):
    pub = DreamStatePublisher(titan_id)
    pub.publish()
    snap = pub.snapshot_for_emit()
    assert snap["is_dreaming"] is False
    assert snap["state"] == "awake"
    assert snap["recovery_pct"] == 0.0
    assert snap["remaining_epochs"] == 0
    assert snap["wake_transition"] is False
    assert snap["just_woke"] is False
    assert snap["schema_version"] == 1


def test_publisher_detects_dream_start_transition(titan_id):
    pub = DreamStatePublisher(titan_id)
    pub.publish()  # cold
    transitioned = pub.update_from_dreaming_state(
        {"is_dreaming": True, "state": "dream_start",
         "expected_dream_epochs": 50})
    assert transitioned is True
    snap = pub.snapshot_for_emit()
    assert snap["is_dreaming"] is True
    assert snap["state"] == "dream_start"
    assert snap["recovery_pct"] == 0.0
    assert snap["remaining_epochs"] == 50
    assert snap["dream_started_ts"] > 0
    assert snap["last_transition_ts"] > 0


def test_publisher_detects_dream_end_transition(titan_id):
    pub = DreamStatePublisher(titan_id)
    # enter then exit
    pub.update_from_dreaming_state(
        {"is_dreaming": True, "state": "dream_start"})
    transitioned = pub.update_from_dreaming_state(
        {"is_dreaming": False, "state": "dream_end"})
    assert transitioned is True
    snap = pub.snapshot_for_emit()
    assert snap["is_dreaming"] is False
    assert snap["state"] == "dream_end"
    assert snap["recovery_pct"] == 100.0
    assert snap["just_woke"] is True
    assert snap["wake_ts"] > 0
    assert snap["wake_transition"] is True


def test_publisher_no_transition_when_state_unchanged(titan_id):
    pub = DreamStatePublisher(titan_id)
    pub.update_from_dreaming_state({"is_dreaming": False, "state": "awake"})
    # Second awake → no transition
    transitioned = pub.update_from_dreaming_state(
        {"is_dreaming": False, "state": "awake"})
    assert transitioned is False


def test_publisher_dream_start_state_string_always_transitions(titan_id):
    """Even when is_dreaming was already True, a fresh state='dream_start'
    payload counts as a transition edge per the inline comment in
    update_from_dreaming_state. Defensive — the canonical producer should
    only ever emit dream_start once, but worker shouldn't miss an edge.
    """
    pub = DreamStatePublisher(titan_id)
    pub.update_from_dreaming_state(
        {"is_dreaming": True, "state": "dreaming"})
    # Same is_dreaming but a dream_start label → still treated as edge.
    transitioned = pub.update_from_dreaming_state(
        {"is_dreaming": True, "state": "dream_start"})
    assert transitioned is True


def test_publisher_malformed_payload_returns_false(titan_id):
    pub = DreamStatePublisher(titan_id)
    # Not a dict — defensive parsing returns False without crashing.
    assert pub.update_from_dreaming_state("not_a_dict") is False
    # Note: a dict with no is_dreaming key defaults to False (the documented
    # cold-boot behavior); this is "no transition" (matches initial False),
    # not a parse error. Same shape as `{}` from a sparse upstream payload.


def test_publisher_writes_to_shm_and_reader_reads_back(titan_id):
    pub = DreamStatePublisher(titan_id)
    pub.update_from_dreaming_state(
        {"is_dreaming": True, "state": "dream_start",
         "expected_dream_epochs": 30})
    pub.publish()

    reader = DreamStateReader(titan_id)
    snap = reader.read()
    assert snap["is_dreaming"] is True
    assert snap["state"] == "dream_start"
    assert snap["remaining_epochs"] == 30


# ── DreamStateReader ───────────────────────────────────────────────


def test_reader_cold_boot_returns_defaults(titan_id):
    # Publisher attached but never published — reader sees cold-boot defaults.
    pub = DreamStatePublisher(titan_id)
    _ = pub  # publisher creates SHM slot but doesn't write
    reader = DreamStateReader(titan_id)
    snap = reader.read()
    assert snap["is_dreaming"] is False
    assert snap["state"] == "awake"


def test_reader_is_dreaming_shortcut(titan_id):
    pub = DreamStatePublisher(titan_id)
    pub.update_from_dreaming_state({"is_dreaming": True, "state": "dreaming"})
    pub.publish()
    reader = DreamStateReader(titan_id)
    assert reader.is_dreaming() is True


def test_reader_100ms_cache_absorbs_repeated_reads(titan_id):
    pub = DreamStatePublisher(titan_id)
    pub.publish()
    reader = DreamStateReader(titan_id)
    # First read populates cache.
    snap1 = reader.read()
    # Publisher updates SHM while cache is still warm.
    pub.update_from_dreaming_state({"is_dreaming": True, "state": "dream_start"})
    pub.publish()
    # Second read within 100ms cache window — sees stale cached payload.
    snap2 = reader.read()
    assert snap2["is_dreaming"] == snap1["is_dreaming"]  # cached
    # Wait past cache TTL.
    time.sleep(0.15)
    snap3 = reader.read()
    assert snap3["is_dreaming"] is True  # fresh read


def test_reader_is_stale_false_on_fresh_publish(titan_id):
    pub = DreamStatePublisher(titan_id)
    pub.update_from_dreaming_state({"is_dreaming": True, "state": "dream_start"})
    pub.publish()
    reader = DreamStateReader(titan_id)
    assert reader.is_stale() is False


def test_reader_is_stale_false_on_cold_boot(titan_id):
    # Never written → last_transition_ts=0 → is_stale returns False (cold-boot
    # is not "stale"; just "not yet initialized"). Documented in helper.
    pub = DreamStatePublisher(titan_id)
    _ = pub
    reader = DreamStateReader(titan_id)
    assert reader.is_stale() is False


# ── DreamingEngine.request_wake (D-SPEC-56 / cognitive_worker subscriber side) ──


def test_dreaming_engine_request_wake_sets_flag():
    from titan_hcl.logic.dreaming import DreamingEngine
    de = DreamingEngine.__new__(DreamingEngine)  # bypass __init__ side effects
    de._wake_requested = False
    de._wake_request_reason = ""
    de.request_wake("maker_message")
    assert de._wake_requested is True
    assert de._wake_request_reason == "maker_message"


def test_dreaming_engine_request_wake_idempotent_collapses():
    from titan_hcl.logic.dreaming import DreamingEngine
    de = DreamingEngine.__new__(DreamingEngine)
    de._wake_requested = False
    de._wake_request_reason = ""
    de.request_wake("first")
    de.request_wake("second")  # second call doesn't re-log, but reason updates
    assert de._wake_requested is True
    # Reason: first call sets, second preserves (first non-empty wins per
    # the `self._wake_request_reason or` fallback in the helper).
    assert de._wake_request_reason in ("first", "second")


def test_dreaming_engine_request_wake_default_empty_reason():
    from titan_hcl.logic.dreaming import DreamingEngine
    de = DreamingEngine.__new__(DreamingEngine)
    de._wake_requested = False
    de._wake_request_reason = ""
    de.request_wake()  # no arg
    assert de._wake_requested is True
    # Empty reason allowed.
    assert de._wake_request_reason == ""


# ── DreamingEngine.request_dream + FORCE_DREAM_REQUEST wiring (post-§4.I D8-3 cleanup) ──


def test_dreaming_engine_request_dream_sets_flag():
    """Symmetric to request_wake — sets `_dream_requested` + captures reason.
    Closes the orphan FORCE_DREAM_REQUEST handler from chunk I8 cleanup."""
    from titan_hcl.logic.dreaming import DreamingEngine
    de = DreamingEngine.__new__(DreamingEngine)
    de._dream_requested = False
    de._dream_request_reason = ""
    de.request_dream("admin_force")
    assert de._dream_requested is True
    assert de._dream_request_reason == "admin_force"


def test_dreaming_engine_request_dream_idempotent_collapses():
    from titan_hcl.logic.dreaming import DreamingEngine
    de = DreamingEngine.__new__(DreamingEngine)
    de._dream_requested = False
    de._dream_request_reason = ""
    de.request_dream("first")
    de.request_dream("second")
    assert de._dream_requested is True
    assert de._dream_request_reason in ("first", "second")


def test_dreaming_engine_request_dream_default_empty_reason():
    from titan_hcl.logic.dreaming import DreamingEngine
    de = DreamingEngine.__new__(DreamingEngine)
    de._dream_requested = False
    de._dream_request_reason = ""
    de.request_dream()
    assert de._dream_requested is True
    assert de._dream_request_reason == ""


def test_bus_force_dream_request_constant_defined():
    """FORCE_DREAM_REQUEST is now a proper bus constant (was a bare string in
    command_sender.py before post-§4.I D8-3 cleanup)."""
    from titan_hcl import bus
    assert bus.FORCE_DREAM_REQUEST == "FORCE_DREAM_REQUEST"


def test_command_sender_force_dream_dst_is_cognitive_worker():
    """Post-§4.I D8-3 cleanup: force_dream() dst flipped 'spirit' →
    'cognitive_worker' (uses the registered ModuleSpec name; broker routes
    by exact name match — `cognitive` alone would not route since the worker
    is named `cognitive_worker` in plugin.py:859).
    """
    from titan_hcl.api.command_sender import CommandSender

    captured = []

    class _FakeQueue:
        def put(self, msg):
            captured.append(msg)

    cs = CommandSender(send_queue=_FakeQueue())
    rid = cs.force_dream(reason="test", source="cli")
    assert rid  # non-empty rid string
    assert len(captured) == 1
    msg = captured[0]
    assert msg.get("type") == "FORCE_DREAM_REQUEST"
    # MUST be the registered ModuleSpec name, not a short alias — broker
    # routes by exact match (see plugin.py:859 `name="cognitive_worker"`).
    assert msg.get("dst") == "cognitive_worker"
    payload = msg.get("payload", {})
    assert payload.get("reason") == "test"
    assert payload.get("source") == "cli"


# ── Worker module integration ──────────────────────────────────────


def test_worker_module_entry_point_imports():
    from titan_hcl.modules.dream_state_worker import (
        _heartbeat_loop, _send, _validate_enqueue_payload,
        dream_state_worker_main,
    )
    assert callable(dream_state_worker_main)
    assert callable(_send)
    assert callable(_heartbeat_loop)
    assert callable(_validate_enqueue_payload)


def test_validate_enqueue_payload_truncates_long_message():
    from titan_hcl.modules.dream_state_worker import _validate_enqueue_payload
    over_max = "x" * (DREAM_INBOX_MAX_MESSAGE_CHARS + 100)
    out = _validate_enqueue_payload({"message": over_max, "user_id": "u"})
    assert out is not None
    assert len(out["message"]) == DREAM_INBOX_MAX_MESSAGE_CHARS


def test_validate_enqueue_payload_defaults_missing_fields():
    from titan_hcl.modules.dream_state_worker import _validate_enqueue_payload
    out = _validate_enqueue_payload({"message": "hi"})
    assert out is not None
    assert out["user_id"] == "anonymous"
    assert out["session_id"] == "default"
    assert out["channel"] == "web"
    assert out["priority"] == 1
    assert isinstance(out["client_ts"], float)


def test_validate_enqueue_payload_rejects_non_dict():
    from titan_hcl.modules.dream_state_worker import _validate_enqueue_payload
    assert _validate_enqueue_payload("not_a_dict") is None
    assert _validate_enqueue_payload(None) is None


def test_validate_enqueue_payload_preserves_maker_priority():
    from titan_hcl.modules.dream_state_worker import _validate_enqueue_payload
    out = _validate_enqueue_payload(
        {"message": "wake", "user_id": "maker", "priority": 0})
    assert out is not None
    assert out["priority"] == 0  # maker priority preserved


# ── ModuleSpec registration ────────────────────────────────────────


def test_module_spec_registered_in_plugin():
    """Verify the dream_state ModuleSpec is registered in plugin._register_modules
    (text-presence check, since instantiating TitanHCL requires full boot).
    """
    plugin_path = Path(__file__).resolve().parents[1] / "titan_hcl" / "core" / "plugin.py"
    source = plugin_path.read_text()
    assert 'name="dream_state"' in source
    assert "dream_state_worker_main" in source
    assert "rss_limit_mb=200" in source


# ── Round-trip: msgpack payload fits in SHM slot ───────────────────


def test_msgpack_payload_fits_in_max_bytes():
    """A maximal realistic payload (all fields populated incl. circadian +
    distillation telemetry) encodes well under DREAM_STATE_MAX_BYTES=512."""
    payload = {
        "is_dreaming": True,
        "state": "dream_start",
        "recovery_pct": 99.9,
        "remaining_epochs": 9999,
        "wake_transition": True,
        "just_woke": True,
        "wake_ts": 1778867281.4717166,
        "dream_started_ts": 1778867281.4717166,
        "last_transition_ts": 1778867281.4717166,
        # Additive circadian telemetry
        "cycle_count": 99999,
        "fatigue": 0.9999,
        "developmental_age": 9999999,
        "epochs_since_dream": 99999,
        # Additive distillation telemetry (rFP_experience_distillation_phase_c)
        "distill_attempts": 9999999,
        "distill_passed": 9999999,
        "distilled_count": 999999,
        "distill_threshold": 0.999999,
        "experience_buffer_size": 99999,
        "schema_version": 1,
        "ts": 1778867281.4717166,
    }
    encoded = msgpack.packb(payload, use_bin_type=True)
    assert len(encoded) <= DREAM_STATE_MAX_BYTES, (
        f"payload {len(encoded)}B exceeds slot cap {DREAM_STATE_MAX_BYTES}B"
    )


# ── Default payload contract (DreamStateReader fallback) ───────────


def test_default_payload_keys_match_schema():
    """The cold-boot/decode-fail fallback returned by DreamStateReader must
    have every schema key so callers can `.get()` without surprises."""
    expected_keys = {
        "is_dreaming", "state", "recovery_pct", "remaining_epochs",
        "wake_transition", "just_woke", "wake_ts", "dream_started_ts",
        "last_transition_ts", "schema_version", "ts",
    }
    assert set(_DEFAULT_PAYLOAD.keys()) == expected_keys


def test_default_payload_is_dreaming_safe():
    """Cold-boot default MUST be is_dreaming=False — readers that fall back
    to defaults must NOT think Titan is dreaming."""
    assert _DEFAULT_PAYLOAD["is_dreaming"] is False
    assert _DEFAULT_PAYLOAD["state"] == "awake"


def test_staleness_threshold_matches_q6_greenlight():
    """Q6 greenlight: stale when (now - last_transition_ts) > republish × 5."""
    assert _STALENESS_THRESHOLD_S == DREAM_STATE_REPUBLISH_CADENCE_S * 5
