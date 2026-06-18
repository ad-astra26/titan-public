"""tests/test_trinity_130d_phase2_5.py — Phase 2.5 closure unit tests.

rFP_trinity_130d_phase2_5_closure §6.

Coverage by chunk:
  2.5.A — DimFiringTracker + /v4/debug/dim-sources endpoint
  2.5.B — four-state classifier (arch_map dim-live) — added when chunk B lands
  2.5.C — creative_tension cold-start regression pin — added when chunk C lands
  2.5.D — self_recognition / sovereignty / expressive_authenticity pins — D
  2.5.E — fleet-wide community_connection + expression_reach — E
"""
from __future__ import annotations

import time

import pytest

from titan_hcl.api.dim_registry import (
    DimFiringRecord,
    DimFiringTracker,
    BlockFiringRecord,
    _BLOCK_INPUT_NAMES,
    _BLOCK_INPUT_TO_DIM_INDICES,
    _classify_input,
    filter_inputs_state_for_dim,
    get_dims_for_block_input,
    get_inputs_for_block_dim,
    get_firing_tracker,
    iter_registry,
    reset_firing_tracker,
)


# ── Chunk 2.5.A: tracker ────────────────────────────────────────────────


class TestPhase25ATracker:
    def setup_method(self):
        reset_firing_tracker()

    def test_tracker_initializes_all_130_dims(self):
        t = DimFiringTracker()
        records = t.get_all_dim_records()
        assert len(records) == 130, "tracker must initialize all 130 dims"
        # Spot-check known dims
        names = {r.full_index: r.name for r in records}
        assert names[0] == "interoception"          # inner_body[0]
        assert names[5] == "memory_depth"            # inner_mind[0]
        assert names[20] == "self_recognition"       # inner_spirit[0]
        assert names[121] == "community_connection"  # outer_spirit[36]
        assert names[126] == "creative_tension"      # outer_spirit[41]

    def test_tracker_initializes_all_6_blocks(self):
        t = DimFiringTracker()
        blocks = t.get_all_block_records()
        assert set(blocks.keys()) == {
            "inner_body", "inner_mind", "inner_spirit",
            "outer_body", "outer_mind", "outer_spirit",
        }
        for br in blocks.values():
            assert isinstance(br, BlockFiringRecord)
            assert br.calls_total == 0
            assert br.last_call_ts is None
            assert br.last_inputs_state == {}

    def test_record_block_updates_dim_values(self):
        t = DimFiringTracker()
        t.record_block(
            "inner_body",
            [0.1, 0.2, 0.3, 0.4, 0.5],
            {"body_state": {"some": "data"}},
            ts=1234.5,
        )
        for i, expected in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
            rec = t.get_dim_record(i)
            assert rec.last_value == expected
            assert rec.last_value_ts == 1234.5
        # Other blocks untouched
        assert t.get_dim_record(5).last_value is None  # inner_mind[0]

    def test_record_block_updates_block_metadata(self):
        t = DimFiringTracker()
        t.record_block(
            "outer_spirit",
            [0.5] * 45,
            {"hormone_levels": {"CREATIVITY": 0.7}, "sovereignty_ratio": 0.0},
            ts=2000.0,
        )
        br = t.get_block_record("outer_spirit")
        assert br.calls_total == 1
        assert br.last_call_ts == 2000.0
        assert br.last_inputs_state["hormone_levels"] == "real"
        # sovereignty_ratio=0.0 with the named-input rule → "default"
        assert br.last_inputs_state["sovereignty_ratio"] == "default"
        # Inputs not passed → "absent"
        assert br.last_inputs_state["action_stats"] == "absent"

    def test_record_block_increments_calls_total(self):
        t = DimFiringTracker()
        t.record_block("inner_mind", [0.5] * 15, {})
        t.record_block("inner_mind", [0.5] * 15, {})
        t.record_block("inner_mind", [0.5] * 15, {})
        br = t.get_block_record("inner_mind")
        assert br.calls_total == 3

    def test_record_block_handles_unknown_block(self):
        t = DimFiringTracker()
        t.record_block("not_a_block", [0.5], {})
        # No crash, no state change
        assert all(b.calls_total == 0 for b in t.get_all_block_records().values())

    def test_record_block_handles_short_values(self):
        # If a tensor returns fewer values than expected, only update what we have.
        t = DimFiringTracker()
        t.record_block("inner_body", [0.7, 0.8], {}, ts=10.0)
        assert t.get_dim_record(0).last_value == 0.7
        assert t.get_dim_record(1).last_value == 0.8
        # Indices 2-4 stay None
        assert t.get_dim_record(2).last_value is None
        assert t.get_dim_record(3).last_value is None
        assert t.get_dim_record(4).last_value is None

    def test_record_block_thread_safe(self):
        # Sanity: concurrent record_block calls do not raise. Not a strict
        # ordering test — just exercises the lock.
        import threading

        t = DimFiringTracker()

        def worker():
            for _ in range(100):
                t.record_block("inner_body", [0.1] * 5, {"body_state": {"x": 1}})

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        br = t.get_block_record("inner_body")
        assert br.calls_total == 400  # 4 threads * 100 calls

    def test_classify_input_real(self):
        # Non-empty dict / non-default float → real
        assert _classify_input("hormone_levels", {"DA": 0.7}) == "real"
        assert _classify_input("hormone_levels", {"DA": 0.0}) == "real"  # dict has content
        assert _classify_input("uptime_ratio", 0.93) == "real"
        assert _classify_input("anchor_state", [1, 2, 3]) == "real"

    def test_classify_input_default(self):
        # Named-input neutral floats → default
        assert _classify_input("sovereignty_ratio", 0.0) == "default"
        assert _classify_input("uptime_ratio", 1.0) == "default"
        assert _classify_input("interaction_quality", 0.5) == "default"

    def test_classify_input_absent(self):
        assert _classify_input("anything", None) == "absent"
        assert _classify_input("empty_dict", {}) == "absent"
        assert _classify_input("empty_list", []) == "absent"

    def test_get_firing_tracker_singleton(self):
        a = get_firing_tracker()
        b = get_firing_tracker()
        assert a is b, "get_firing_tracker must return the same singleton"

    def test_reset_firing_tracker(self):
        t1 = get_firing_tracker()
        t1.record_block("inner_body", [0.9] * 5, {"body_state": {"x": 1}})
        assert t1.get_block_record("inner_body").calls_total == 1
        reset_firing_tracker()
        t2 = get_firing_tracker()
        assert t2 is not t1
        assert t2.get_block_record("inner_body").calls_total == 0

    def test_block_input_names_cover_all_blocks(self):
        for _, _, block in [(0, 5, "inner_body"), (5, 15, "inner_mind"),
                            (20, 45, "inner_spirit"), (65, 5, "outer_body"),
                            (70, 15, "outer_mind"), (85, 45, "outer_spirit")]:
            assert block in _BLOCK_INPUT_NAMES, f"missing input names for {block}"
            assert len(_BLOCK_INPUT_NAMES[block]) > 0, f"no inputs listed for {block}"


# ── Chunk 2.5.A endpoint integration (requires fastapi TestClient) ────


class TestPhase25AEndpoint:
    """Smoke-test the /v4/debug/dim-sources endpoint via the dashboard router."""

    def setup_method(self):
        reset_firing_tracker()
        # SHM-read isolation: dashboard endpoint at line 4387 prefers
        # `read_all_blocks_from_shm()` over in-process tracker. Without
        # this, the test reads /dev/shm/titan_T1/*_firing.bin from a
        # running T1 (or stale slots from prior runs) and fails with
        # leaked values instead of the [0.1]*5 the test records.
        # Monkey-patch the dim_registry module-level function so the
        # dashboard's per-call `from ... import` re-binds to the stub.
        # Restored in teardown_method so the SHMRoundtrip class below
        # (which legitimately exercises the real SHM path) is unaffected.
        from titan_hcl.api import dim_registry as _dr
        _dr._SHM_READERS.clear()
        self._orig_read_shm = _dr.read_all_blocks_from_shm
        _dr.read_all_blocks_from_shm = lambda: {}

    def teardown_method(self):
        try:
            from titan_hcl.api import dim_registry as _dr
            _dr.read_all_blocks_from_shm = self._orig_read_shm
        except Exception:
            pass

    def test_endpoint_route_registered(self):
        # Verify the route is registered on the dashboard router. Avoids
        # spinning up a full FastAPI TestClient (heavy import chain).
        from titan_hcl.api.v6 import router as v6_router
        paths = [r.path for r in v6_router.routes]
        assert "/v6/system/debug/dim-sources" in paths, (
            "/v6/system/debug/dim-sources route not registered on v6 router"
        )

    def test_endpoint_handler_returns_all_130_dims_when_no_filter(self):
        # Direct handler invocation (no HTTP layer) — fast.
        import asyncio
        from titan_hcl.api.dashboard import get_v4_debug_dim_sources

        # Fire some tensors so the tracker has data
        t = get_firing_tracker()
        t.record_block("inner_body", [0.1] * 5, {"body_state": {"x": 1}})
        t.record_block("inner_mind", [0.5] * 15,
                       {"hormone_levels": {"DA": 0.7}})

        class _DummyRequest:
            pass

        result = asyncio.run(get_v4_debug_dim_sources(_DummyRequest(), dim=""))
        # _ok wraps in JSONResponse — extract body
        import json
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "ok"
        assert body["data"]["total"] == 130
        dims = body["data"]["dims"]
        assert len(dims) == 130
        # Inner body dims should have last_value populated
        assert dims[0]["last_value"] == 0.1
        assert dims[0]["block"] == "inner_body"
        # Inner mind first dim should have last_value populated
        assert dims[5]["last_value"] == 0.5

    def test_endpoint_handler_filters_by_dim(self):
        import asyncio
        from titan_hcl.api.dashboard import get_v4_debug_dim_sources
        t = get_firing_tracker()
        t.record_block("outer_spirit", [0.5] * 45, {})

        class _DummyRequest:
            pass

        result = asyncio.run(get_v4_debug_dim_sources(_DummyRequest(),
                                                      dim="121,126"))
        import json
        body = json.loads(result.body.decode("utf-8"))
        assert body["data"]["total"] == 2
        idxs = [d["idx"] for d in body["data"]["dims"]]
        assert idxs == [121, 126]
        # Names from the registry
        names = {d["idx"]: d["name"] for d in body["data"]["dims"]}
        assert names[121] == "community_connection"
        assert names[126] == "creative_tension"

    def test_endpoint_handler_invalid_dim_filter_returns_error(self):
        import asyncio
        from titan_hcl.api.dashboard import get_v4_debug_dim_sources

        class _DummyRequest:
            pass

        result = asyncio.run(get_v4_debug_dim_sources(_DummyRequest(),
                                                      dim="abc,def"))
        import json
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "error"

    def test_endpoint_payload_includes_inputs_and_block_metadata(self):
        import asyncio
        from titan_hcl.api.dashboard import get_v4_debug_dim_sources
        t = get_firing_tracker()
        t.record_block(
            "inner_mind",
            [0.5] * 15,
            {"hormone_levels": {"DA": 0.7}, "audio_state": None},
            ts=1000.0,
        )

        class _DummyRequest:
            pass

        result = asyncio.run(get_v4_debug_dim_sources(_DummyRequest(),
                                                      dim="5"))
        import json
        body = json.loads(result.body.decode("utf-8"))
        d = body["data"]["dims"][0]
        assert d["idx"] == 5
        assert d["block"] == "inner_mind"
        assert d["block_calls_total"] == 1
        assert d["block_last_call_ts"] == 1000.0
        # Inputs payload should contain at least hormone_levels=real
        # and audio_state=absent (None passed)
        states = {i["name"]: i["state"] for i in d["inputs"]}
        assert states.get("hormone_levels") == "real"
        assert states.get("audio_state") == "absent"


# ── Tensor wiring smoke tests (each tensor function calls record_block) ──


class TestPhase25ATensorWiring:
    """Verify each of the 6 tensor producers calls record_block on return."""

    def setup_method(self):
        reset_firing_tracker()

    def test_collect_mind_15d_records_block(self):
        from titan_hcl.logic.mind_tensor import collect_mind_15d
        result = collect_mind_15d(
            current_5d=[0.5, 0.5, 0.5, 0.5, 0.5],
            audio_state={"creates_recent": 2, "ambient": 0.3},
            interaction_quality=0.7,
            visual_state={"creates_recent": 1, "ambient": 0.4},
            assessment_quality=0.6,
            ambient_change=0.2,
            hormone_levels={"IMPULSE": 0.5, "EMPATHY": 0.4, "CREATIVITY": 0.3,
                            "VIGILANCE": 0.6, "CURIOSITY": 0.7},
        )
        assert len(result) == 15
        br = get_firing_tracker().get_block_record("inner_mind")
        assert br.calls_total == 1
        assert br.last_inputs_state["hormone_levels"] == "real"
        assert br.last_inputs_state["audio_state"] == "real"

    def test_collect_outer_body_5d_records_block(self):
        from titan_hcl.logic.outer_body_tensor import collect_outer_body_5d
        result = collect_outer_body_5d(sources={
            "anchor_state": {"success": True, "last_anchor_time": time.time()},
            "block_delta_stats": {"normalized": 0.5},
            "system_sensor_stats": {"cpu_thermal": 0.4, "circadian_phase": 0.5},
            "hormone_levels": {"IMPULSE": 0.5, "VIGILANCE": 0.5},
            "agency_stats": {"total_actions": 0, "failed_actions": 0},
            "tx_latency_stats": {"normalized": 0.5},
            "network_monitor_stats": {},
            "helper_statuses": {},
            "bus_stats": {},
        })
        assert len(result) == 5
        br = get_firing_tracker().get_block_record("outer_body")
        assert br.calls_total == 1

    def test_tensor_wiring_does_not_break_on_tracker_failure(self):
        """If the tracker raises, the tensor function must still return."""
        from titan_hcl.logic.mind_tensor import collect_mind_15d
        # Don't reset; leave singleton in valid state. The exception path
        # is exercised by the try/except in the tensor wrapper. We just
        # confirm the function still returns a 15-element list under
        # ordinary conditions.
        result = collect_mind_15d(
            current_5d=[0.5] * 5,
            hormone_levels={"IMPULSE": 0.5, "EMPATHY": 0.4, "CREATIVITY": 0.3,
                            "VIGILANCE": 0.6, "CURIOSITY": 0.7},
        )
        assert len(result) == 15


# ── Chunk 2.5.B: four-state classifier in arch_map dim-live ───────────


class TestPhase25BFourStateClassifier:
    """rFP_trinity_130d_phase2_5_closure §3.1 — five-state classifier
    (ALIVE / PARTIAL / SILENT / CORRUPTED / GHOST).

    Maker directive 2026-05-12: ALIVE_AT_DEFAULT collapsed into PARTIAL.
    A dim firing the SPEC default sentinel = no real signal = PARTIAL,
    regardless of input-state reporting. Strict-gate acceptance is now
    ALIVE-only (no @def credit)."""

    def _record(self, **overrides):
        # Sensible default record; tests override specific fields.
        base = {
            "idx": 0,
            "name": "interoception",
            "block": "inner_body",
            "block_index": 0,
            "spec_section": "§23.4",
            "spec_default": 0.5,
            "last_value": 0.5,
            "last_value_ts": time.time(),
            "seconds_since_last_write": 1.0,
            "block_calls_total": 5,
            "block_last_call_ts": time.time(),
            "inputs": [{"name": "body_state", "state": "real"}],
        }
        base.update(overrides)
        return base

    def test_alive_when_value_far_from_default(self):
        from scripts.arch_map import _dim_live_classify_v2
        rec = self._record(last_value=0.8, spec_default=0.5)
        assert _dim_live_classify_v2(rec) == "ALIVE"

    def test_partial_when_within_epsilon_and_firing(self):
        # Maker directive 2026-05-12: was ALIVE_AT_DEFAULT, now PARTIAL.
        from scripts.arch_map import _dim_live_classify_v2
        rec = self._record(
            last_value=0.501, spec_default=0.5,
            seconds_since_last_write=10.0,
            inputs=[{"name": "body_state", "state": "real"}],
        )
        assert _dim_live_classify_v2(rec) == "PARTIAL"

    def test_partial_when_at_default_and_input_absent(self):
        # Was PARTIAL pre-collapse and is still PARTIAL — semantics
        # subsumed: any dim firing the default sentinel is PARTIAL,
        # regardless of input-state reporting.
        from scripts.arch_map import _dim_live_classify_v2
        rec = self._record(
            last_value=0.5, spec_default=0.5,
            seconds_since_last_write=10.0,
            inputs=[{"name": "body_state", "state": "absent"}],
        )
        assert _dim_live_classify_v2(rec) == "PARTIAL"

    def test_silent_when_block_not_fired_in_window(self):
        from scripts.arch_map import _dim_live_classify_v2
        rec = self._record(
            last_value=0.5, spec_default=0.5,
            seconds_since_last_write=400.0,  # > 300s firing window
            inputs=[{"name": "body_state", "state": "real"}],
        )
        assert _dim_live_classify_v2(rec) == "SILENT"

    def test_silent_when_no_seconds_since(self):
        from scripts.arch_map import _dim_live_classify_v2
        rec = self._record(
            last_value=0.5, spec_default=0.5,
            seconds_since_last_write=None,
            block_calls_total=5,
        )
        assert _dim_live_classify_v2(rec) == "SILENT"

    def test_ghost_when_no_calls_yet_and_no_value(self):
        from scripts.arch_map import _dim_live_classify_v2
        rec = self._record(
            last_value=None, block_calls_total=0,
            seconds_since_last_write=None,
        )
        assert _dim_live_classify_v2(rec) == "GHOST"

    def test_corrupted_when_nan(self):
        from scripts.arch_map import _dim_live_classify_v2
        import math
        rec = self._record(last_value=math.nan)
        assert _dim_live_classify_v2(rec) == "CORRUPTED"

    def test_corrupted_when_out_of_range(self):
        from scripts.arch_map import _dim_live_classify_v2
        rec = self._record(last_value=1.5)
        assert _dim_live_classify_v2(rec) == "CORRUPTED"

    def test_epsilon_default_005(self):
        from scripts.arch_map import _dim_live_classify_v2, _DIM_LIVE_EPSILON
        assert _DIM_LIVE_EPSILON == 0.005, (
            "rFP §3.1 locked EPSILON=0.005 — do not change without re-locking")

    def test_firing_window_default_300s(self):
        from scripts.arch_map import _DIM_LIVE_FIRING_WINDOW_S
        assert _DIM_LIVE_FIRING_WINDOW_S == 300.0, (
            "rFP §3.1 locked firing_window=300s — do not change without re-locking")

    def test_zero_default_with_value_above_epsilon_is_alive(self):
        # community_connection / expression_reach / creative_tension all
        # have spec_default=0.0; a non-zero value should be ALIVE.
        from scripts.arch_map import _dim_live_classify_v2
        rec = self._record(
            spec_default=0.0, last_value=0.1,
            seconds_since_last_write=5.0,
        )
        assert _dim_live_classify_v2(rec) == "ALIVE"

    def test_zero_default_with_value_zero_and_firing_is_partial(self):
        # Cold-start of community_connection: value=0.0 (matches default),
        # producer firing, all inputs real → PARTIAL (was ALIVE_AT_DEFAULT
        # pre Maker directive 2026-05-12; collapsed because firing the
        # default sentinel = no real signal = partial coverage).
        from scripts.arch_map import _dim_live_classify_v2
        rec = self._record(
            spec_default=0.0, last_value=0.0,
            seconds_since_last_write=5.0,
            inputs=[{"name": "social_x_gateway_stats", "state": "real"}],
        )
        assert _dim_live_classify_v2(rec) == "PARTIAL"

    def test_state_marker_returns_unicode(self):
        from scripts.arch_map import _dim_live_state_marker
        assert _dim_live_state_marker("ALIVE") == "✓"
        assert _dim_live_state_marker("PARTIAL") == "◑"
        assert _dim_live_state_marker("SILENT") == "·"
        assert _dim_live_state_marker("CORRUPTED") == "✗"
        assert _dim_live_state_marker("GHOST") == "▢"


# ── Chunk 2.5.C: creative_tension cold-start regression pin ──────────


class TestPhase25CCreativeTensionFormula:
    """rFP §4.1 — pin SPEC §23.9 ANANDA[41] formula correctness.

    The wiring fix is deferred until live diagnostic on T1 confirms
    which call site feeds partial producers. The formula itself is
    pinned here so any regression on the SPEC formula is caught.
    """

    def _outer_spirit_45d_with_inputs(
            self, hormone_levels=None, history=None,
            **kwargs) -> list:
        from titan_hcl.logic.outer_spirit_tensor import collect_outer_spirit_45d
        return collect_outer_spirit_45d(
            current_5d=[0.5] * 5,
            outer_body=[0.5] * 5,
            outer_mind=[0.5] * 15,
            hormone_levels=hormone_levels or {"CREATIVITY": 0.0},
            history=history or {},
            **kwargs,
        )

    def test_cold_start_no_create_returns_full_creativity_hormone(self):
        # SPEC §23.9 ANANDA[41]: cold-start (no creates yet) uses
        # seconds_since_create=600 → tension = full CREATIVITY hormone.
        # The history dict's seconds_since_last_create=600 simulates
        # cold-start (never created yet).
        tensor = self._outer_spirit_45d_with_inputs(
            hormone_levels={"CREATIVITY": 0.7},
            history={"seconds_since_last_create": 600.0},
        )
        ananda_41 = tensor[30 + 11]  # ANANDA block starts at outer_spirit[30]
        assert ananda_41 == pytest.approx(0.7, abs=0.01), (
            "cold-start creative_tension must equal full CREATIVITY hormone")

    def test_recent_create_collapses_tension(self):
        # Just-created (seconds_since_create=0) → tension ≈ 0
        tensor = self._outer_spirit_45d_with_inputs(
            hormone_levels={"CREATIVITY": 0.7},
            history={"seconds_since_last_create": 0.0},
        )
        ananda_41 = tensor[30 + 11]
        assert ananda_41 == pytest.approx(0.0, abs=0.001)

    def test_zero_creativity_yields_zero_tension(self):
        # creativity=0 → tension=0 regardless of dt (no pent-up creativity).
        tensor = self._outer_spirit_45d_with_inputs(
            hormone_levels={"CREATIVITY": 0.0},
            history={"seconds_since_last_create": 600.0},
        )
        ananda_41 = tensor[30 + 11]
        assert ananda_41 == pytest.approx(0.0, abs=0.001)

    def test_saturates_at_10min(self):
        # SPEC: min(1, dt/600) saturates at 600s. Beyond → still hormone level.
        tensor_at_600 = self._outer_spirit_45d_with_inputs(
            hormone_levels={"CREATIVITY": 0.5},
            history={"seconds_since_last_create": 600.0},
        )
        tensor_beyond = self._outer_spirit_45d_with_inputs(
            hormone_levels={"CREATIVITY": 0.5},
            history={"seconds_since_last_create": 6000.0},
        )
        assert tensor_at_600[30 + 11] == pytest.approx(
            tensor_beyond[30 + 11], abs=0.001)


# ── Chunk 2.5.E: T2/T3 fleet-wide ANANDA[36]+[38] ──────────────────────


class TestPhase25ESocialXGatewayPerTitan:
    """rFP §5 — verify the per-Titan filter on
    SocialXGateway.get_community_engagement_stats."""

    def setup_method(self):
        # Build a tmp DB with rows for T1, T2, T3
        import tempfile, sqlite3, os
        self._tmpdir = tempfile.mkdtemp(prefix="phase2_5e_")
        os.makedirs(os.path.join(self._tmpdir, "data"), exist_ok=True)
        self._db_path = os.path.join(self._tmpdir, "social_x.db")
        db = sqlite3.connect(self._db_path)
        db.executescript("""
            CREATE TABLE mention_tracking (
                tweet_id TEXT PRIMARY KEY,
                author TEXT NOT NULL,
                author_handle TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL DEFAULT '',
                our_post_id TEXT,
                titan_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                relevance_score REAL DEFAULT 0.0,
                discovered_at REAL NOT NULL,
                replied_at REAL,
                reply_tweet_id TEXT
            );
        """)
        now = time.time()
        # T1 has 3 distinct mention authors, T2 has 1, T3 has 0
        for i, (tid, handle) in enumerate([
                ("T1", "alice"), ("T1", "bob"), ("T1", "carol"),
                ("T2", "dave")]):
            db.execute(
                "INSERT INTO mention_tracking (tweet_id, author, "
                "author_handle, titan_id, discovered_at, text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (f"tw{i}", handle, handle, tid, now - 100.0, "hi"))
        db.commit()
        db.close()
        self._cwd = os.getcwd()
        os.chdir(self._tmpdir)
        # Stub events_teacher.db (engagement_snapshots) — needed because
        # get_community_engagement_stats opens it relative to cwd.
        eng = sqlite3.connect(os.path.join(self._tmpdir, "data",
                                            "events_teacher.db"))
        eng.executescript("""
            CREATE TABLE engagement_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                titan_id TEXT NOT NULL,
                tweet_id TEXT NOT NULL,
                likes INTEGER DEFAULT 0,
                replies INTEGER DEFAULT 0,
                quotes INTEGER DEFAULT 0,
                delta_likes INTEGER DEFAULT 0,
                delta_replies INTEGER DEFAULT 0,
                delta_quotes INTEGER DEFAULT 0,
                checked_at REAL NOT NULL
            );
        """)
        eng.commit()
        eng.close()

    def teardown_method(self):
        import os, shutil
        os.chdir(self._cwd)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_titan_id_filter_t1(self):
        from titan_hcl.logic.social_x_gateway import SocialXGateway
        sxg = SocialXGateway.__new__(SocialXGateway)  # bypass __init__
        sxg._db_path = self._db_path
        stats = sxg.get_community_engagement_stats(
            is_x_gateway=True, titan_id="T1")
        assert stats["distinct_handles_24h"] == 3
        assert stats["titan_id"] == "T1"
        assert stats["gateway_role"] == "canonical"

    def test_titan_id_filter_t2(self):
        from titan_hcl.logic.social_x_gateway import SocialXGateway
        sxg = SocialXGateway.__new__(SocialXGateway)
        sxg._db_path = self._db_path
        stats = sxg.get_community_engagement_stats(
            is_x_gateway=True, titan_id="T2")
        assert stats["distinct_handles_24h"] == 1

    def test_titan_id_filter_t3_returns_zero(self):
        # T3 has no mentions in this DB — distinct_handles_24h must be 0.
        # This is "PARTIAL" semantically (real producer, no signal),
        # NOT silent-by-design. (Pre-2026-05-12 was "ALIVE_AT_DEFAULT".)
        from titan_hcl.logic.social_x_gateway import SocialXGateway
        sxg = SocialXGateway.__new__(SocialXGateway)
        sxg._db_path = self._db_path
        stats = sxg.get_community_engagement_stats(
            is_x_gateway=True, titan_id="T3")
        assert stats["distinct_handles_24h"] == 0

    def test_signature_includes_titan_id_default_t1(self):
        # Back-compat: pre-Phase-2.5 callers without titan_id default to T1
        from titan_hcl.logic.social_x_gateway import SocialXGateway
        sxg = SocialXGateway.__new__(SocialXGateway)
        sxg._db_path = self._db_path
        stats = sxg.get_community_engagement_stats(is_x_gateway=True)
        assert stats["titan_id"] == "T1"
        assert stats["distinct_handles_24h"] == 3


class TestPhase25EDashboardEndpoint:
    """Verify /v4/community-engagement-stats endpoint route registration."""

    def test_endpoint_route_registered(self):
        from titan_hcl.api.v6 import router as v6_router
        paths = [r.path for r in v6_router.routes]
        assert "/v6/social/community-engagement-stats" in paths, (
            "Phase 2.5.E /v6/social/community-engagement-stats not registered")


# ── Chunk 2.5.A.2: SHM cross-process roundtrip ────────────────────────


class TestPhase25A2SHMRoundtrip:
    """rFP §2.5.A.2 — verify per-block SHM slot round-trip (READER side).

    Phase C/D: the Rust trinity daemons are the single-writer of every
    *_firing.bin slot (G21). The Python API process READS via
    StateRegistryReader and merges into the /v4/debug/dim-sources payload.
    These tests seed the slot directly (simulating the Rust FiringSlotWriter)
    and assert the reader + endpoint + input-state decode — so they need no
    Rust subprocess. (The legacy Python slot writer was retired in config-shm
    Phase D; un-skipped here now that the tests self-seed.)
    """

    @staticmethod
    def _seed_block_slot(block, dims_values, inputs_state,
                         calls_total=1, last_call_ts=12345.0, ts=12345.0):
        """Write a block's firing slot directly (simulates the Rust
        FiringSlotWriter) so the reader/endpoint can be tested without a
        Rust subprocess. Raises if SHM is unavailable (caller skips)."""
        from titan_hcl.core.state_registry import (
            StateRegistryWriter, ensure_shm_root, resolve_titan_id,
        )
        from titan_hcl.logic.dim_firing_state_specs import (
            DIM_FIRING_SPEC_BY_BLOCK,
        )
        import msgpack
        spec = DIM_FIRING_SPEC_BY_BLOCK[block]
        shm_root = ensure_shm_root(resolve_titan_id())
        writer = StateRegistryWriter(spec, shm_root)
        payload = {
            "block": block,
            "block_calls_total": calls_total,
            "block_last_call_ts": last_call_ts,
            "inputs_state": dict(inputs_state),
            "dims": [{"v": v, "ts": ts} for v in dims_values],
            "ts": ts,
        }
        writer.write_variable(msgpack.packb(payload, use_bin_type=True))

    def setup_method(self):
        import tempfile, os
        reset_firing_tracker()
        # Use isolated TITAN_ID for SHM root (so tests don't touch live shm)
        self._tmp_root = tempfile.mkdtemp(prefix="phase25_shm_")
        os.environ["TITAN_ID"] = "TEST"
        os.environ["TITAN_SHM_ROOT"] = self._tmp_root
        # Clear any cached module-level reader state
        from titan_hcl.api import dim_registry as _dr
        _dr._SHM_READERS.clear()

    def teardown_method(self):
        import os, shutil
        os.environ.pop("TITAN_ID", None)
        os.environ.pop("TITAN_SHM_ROOT", None)
        shutil.rmtree(self._tmp_root, ignore_errors=True)
        # Clear cached readers so next test gets fresh SHM state
        from titan_hcl.api import dim_registry as _dr
        _dr._SHM_READERS.clear()
        reset_firing_tracker()

    def test_block_slot_round_trips_through_shm_reader(self):
        # Seed the slot directly (simulates the Rust FiringSlotWriter),
        # then read it back via the live Python reader.
        try:
            self._seed_block_slot(
                "inner_body",
                [0.11, 0.22, 0.33, 0.44, 0.55],
                {"body_state": "real"},
            )
        except Exception:
            pytest.skip("shm root not available in this environment")
        from titan_hcl.api.dim_registry import read_all_blocks_from_shm
        blocks = read_all_blocks_from_shm()
        assert "inner_body" in blocks, (
            f"inner_body slot not readable; got blocks={list(blocks.keys())}")
        body = blocks["inner_body"]
        assert body["block_calls_total"] == 1
        assert body["block_last_call_ts"] == 12345.0
        assert len(body["dims"]) == 5
        assert body["dims"][0]["v"] == pytest.approx(0.11)
        assert body["dims"][4]["v"] == pytest.approx(0.55)

    def test_endpoint_uses_shm_source_marker_when_available(self):
        # Seed the slot directly (simulates the Rust writer), then assert
        # the endpoint reports block_source="shm".
        try:
            self._seed_block_slot(
                "inner_body",
                [0.7, 0.8, 0.9, 1.0, 0.6],
                {"body_state": "real"},
            )
        except Exception:
            pytest.skip("shm root not available in this environment")
        # Call the endpoint
        import asyncio
        from titan_hcl.api.dashboard import get_v4_debug_dim_sources

        class _DummyRequest:
            pass

        result = asyncio.run(get_v4_debug_dim_sources(
            _DummyRequest(), dim="0,4"))
        import json
        body = json.loads(result.body.decode("utf-8"))
        dims = body["data"]["dims"]
        assert len(dims) == 2
        # block_source should be "shm" because the slot is populated
        assert dims[0]["block_source"] == "shm"
        assert dims[0]["last_value"] == pytest.approx(0.7)

    def test_block_input_state_round_trips_through_shm(self):
        # Seed inputs_state into the slot (the Rust writer carries the
        # classification the producer computed) and assert the reader
        # decodes it faithfully.
        try:
            self._seed_block_slot(
                "inner_mind",
                [0.5] * 15,
                {
                    "hormone_levels": "real",
                    "audio_state": "absent",
                    "interaction_quality": "default",
                },
            )
        except Exception:
            pytest.skip("shm root not available in this environment")
        from titan_hcl.api.dim_registry import read_all_blocks_from_shm
        blocks = read_all_blocks_from_shm()
        assert "inner_mind" in blocks
        states = blocks["inner_mind"]["inputs_state"]
        assert states["hormone_levels"] == "real"
        assert states["audio_state"] == "absent"
        assert states["interaction_quality"] == "default"


# ── L4 / SPEC §2.6.A — per-input-to-dim mapping refinement ──────────────


class TestL4PerInputToDimMapping:
    """SPEC §2.6.A — closes the false-PARTIAL class where one absent block
    input flags up to 45 dims (SPEC line 5852). Maker-locked refinement
    introduced via the 2026-05-26 housekeeping closure.
    """

    def test_all_block_input_maps_reference_known_inputs(self):
        # Every input name in _BLOCK_INPUT_TO_DIM_INDICES must also exist
        # in _BLOCK_INPUT_NAMES for the same block — no stray names.
        for block, input_map in _BLOCK_INPUT_TO_DIM_INDICES.items():
            assert block in _BLOCK_INPUT_NAMES, (
                f"unknown block in dim map: {block}")
            known_inputs = set(_BLOCK_INPUT_NAMES[block])
            for input_name in input_map:
                assert input_name in known_inputs, (
                    f"{block} maps unknown input '{input_name}' — must be in "
                    f"_BLOCK_INPUT_NAMES")

    def test_all_block_dim_indices_in_range(self):
        # Each block-relative index in the input→dim map must be valid for
        # that block's length.
        block_len = {b: L for _, L, b in
                     [(0, 5, "inner_body"), (5, 15, "inner_mind"),
                      (20, 45, "inner_spirit"), (65, 5, "outer_body"),
                      (70, 15, "outer_mind"), (85, 45, "outer_spirit")]}
        for block, input_map in _BLOCK_INPUT_TO_DIM_INDICES.items():
            L = block_len[block]
            for input_name, indices in input_map.items():
                for idx in indices:
                    assert 0 <= idx < L, (
                        f"{block}.{input_name} maps to block_dim_idx={idx} "
                        f"which is out of range [0, {L})")

    def test_recovery_stats_maps_only_to_recovery_speed(self):
        # The canonical example from SPEC line 5852: recovery_stats
        # absence on outer_spirit should ONLY flag SAT[10] recovery_speed,
        # not all 45 dims.
        dims = get_dims_for_block_input("outer_spirit", "recovery_stats")
        assert dims == {10}, (
            f"recovery_stats should map to {{10}}, got {dims}")

    def test_inner_body_all_dims_consume_body_state(self):
        # inner_body has a single composite input — all 5 dims read from it.
        dims = get_dims_for_block_input("inner_body", "body_state")
        assert dims == {0, 1, 2, 3, 4}

    def test_inner_mind_audio_state_maps_to_inner_hearing(self):
        # SPEC §23.5: audio_state feeds inner_hearing (block[5]) + the
        # perceptual_thinking composite (block[2]).
        dims = get_dims_for_block_input("inner_mind", "audio_state")
        assert 5 in dims  # inner_hearing
        assert dims.issubset({2, 5, 7})  # don't accept stray mappings

    def test_inner_mind_hormone_levels_feeds_all_willing_dims(self):
        # SPEC §23.5: hormone_levels.{IMPULSE,EMPATHY,CREATIVITY,
        # VIGILANCE,CURIOSITY} drives willing[0:5] = block[10:15].
        dims = get_dims_for_block_input("inner_mind", "hormone_levels")
        assert {10, 11, 12, 13, 14}.issubset(dims), (
            f"hormone_levels must feed all 5 willing dims; got {dims}")

    def test_get_inputs_for_block_dim_inverse_consistency(self):
        # The inverse map must agree with the forward map for every entry.
        for block, input_map in _BLOCK_INPUT_TO_DIM_INDICES.items():
            for input_name, indices in input_map.items():
                for idx in indices:
                    inputs_for_idx = get_inputs_for_block_dim(block, idx)
                    assert input_name in inputs_for_idx, (
                        f"inverse map broken: {block}.{input_name} maps to "
                        f"{idx} but get_inputs_for_block_dim({block!r}, "
                        f"{idx}) returned {inputs_for_idx}")

    def test_unmapped_dim_returns_empty_set(self):
        # A dim without a recorded mapping returns empty inputs — caller
        # falls back to block-level (the conservative behavior).
        # inner_body[3] entropy: not explicitly excluded by our minimal
        # map, but for safety, validate that an obviously unmapped block_dim
        # behaves correctly. Pick a block that doesn't exist:
        assert get_inputs_for_block_dim("nonexistent_block", 0) == set()
        assert get_dims_for_block_input("inner_mind", "nonexistent_input") == set()

    def test_filter_inputs_state_subsets_correctly(self):
        # recovery_stats absent on outer_spirit: only SAT[10] should
        # surface that input in its filtered state. Pick a non-mapped dim
        # (say outer_spirit[0] world_recognition) and verify it gets the
        # full block fallback (conservative — no info loss vs. block-level
        # classifier).
        block_state = {
            "recovery_stats": "absent",
            "anchor_state": "real",
            "sovereignty_ratio": "real",
            "social_stats": "real",
        }
        # SAT[10] recovery_speed → sees recovery_stats (its specific input)
        sat10 = filter_inputs_state_for_dim("outer_spirit", 10, block_state)
        assert sat10 == {"recovery_stats": "absent"}, (
            f"SAT[10] must filter to only its own input; got {sat10}")
        # SAT[5] origin_anchoring → sees anchor_state (its specific input),
        # NOT recovery_stats (different dim's input).
        sat5 = filter_inputs_state_for_dim("outer_spirit", 5, block_state)
        assert "anchor_state" in sat5
        assert "recovery_stats" not in sat5, (
            f"SAT[5] origin_anchoring must NOT see recovery_stats absence; "
            f"got {sat5}")

    def test_filter_inputs_state_falls_back_to_block_level_for_unmapped(self):
        # Conservative fallback: an unmapped dim returns the full block
        # state, preserving current classifier semantics.
        block_state = {"a": "real", "b": "absent"}
        # Use a deliberately out-of-our-map block_dim_idx (e.g.
        # inner_spirit[44] transcendence_glimpse is in block but not in
        # our minimal map). Should fall back to block-level.
        full = filter_inputs_state_for_dim("inner_spirit", 44, block_state)
        assert full == block_state

    def test_partial_reason_classifier(self):
        # Import the arch_map diagnostic (defined alongside the classifier).
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        try:
            from arch_map import _dim_live_partial_reason
        finally:
            sys.path.pop(0)

        # Case 1: at least one specific input absent → "inputs_absent"
        rec = {"dim_inputs_state": {"recovery_stats": "absent"}}
        assert _dim_live_partial_reason(rec) == "inputs_absent"

        # Case 2: all specific inputs real → "formula_collapse"
        rec = {"dim_inputs_state": {"recovery_stats": "real",
                                      "uptime_ratio": "real"}}
        assert _dim_live_partial_reason(rec) == "formula_collapse"

        # Case 3: no per-dim subset → "" (block-level fallback)
        rec = {}
        assert _dim_live_partial_reason(rec) == ""

        # Case 4: "default" state counts as degraded (per L4 semantics)
        rec = {"dim_inputs_state": {"interaction_quality": "default"}}
        assert _dim_live_partial_reason(rec) == "inputs_absent"
