"""tests/test_vocabulary_grounding.py — L5+L6 housekeeping closure tests.

Closes `rFP_higher_cognition_roadmap.md §Remaining CGN Items` deferrals:
  L5 — bulk_bootstrap_word_grounding admin function
  L6 — MSL-informed felt_tensor seeding at vocabulary INSERT time
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile

import pytest

from titan_hcl.logic.vocabulary_grounding import (
    _ATTN_MODALITY_TO_DIMS,
    _CONCEPT_TO_DIMS,
    _NEUTRAL,
    _TENSOR_LEN,
    bulk_bootstrap_word_grounding,
    compute_msl_seed_tensor,
    seed_new_word_felt_tensor,
)


class TestComputeMSLSeedTensor:
    def test_neutral_seed_when_no_state_supplied(self):
        seed = compute_msl_seed_tensor()
        assert len(seed) == _TENSOR_LEN
        assert all(v == _NEUTRAL for v in seed)

    def test_neutral_seed_with_empty_dicts(self):
        seed = compute_msl_seed_tensor({}, {})
        assert all(v == _NEUTRAL for v in seed)

    def test_attention_modality_bumps_mapped_dims_only(self):
        # Visual attention at full weight bumps inner_sight + outer_mind
        # research/knowledge dims; everything else stays neutral.
        seed = compute_msl_seed_tensor({"visual": 1.0}, None)
        bumped_dims = set(_ATTN_MODALITY_TO_DIMS["visual"])
        for idx in range(_TENSOR_LEN):
            if idx in bumped_dims:
                assert seed[idx] > _NEUTRAL, (
                    f"dim {idx} should be bumped by visual attention")
            else:
                assert seed[idx] == _NEUTRAL, (
                    f"dim {idx} should stay neutral")

    def test_attention_weight_scales_nudge(self):
        # Full weight 1.0 saturates at 1.0; half weight bumps half-way.
        seed_full = compute_msl_seed_tensor({"audio": 1.0}, None)
        seed_half = compute_msl_seed_tensor({"audio": 0.5}, None)
        # audio → inner_hearing (idx 10)
        assert seed_full[10] == pytest.approx(1.0, abs=0.001)
        assert seed_half[10] == pytest.approx(0.75, abs=0.001)

    def test_concept_confidence_bumps_mapped_dims(self):
        # "I" at full confidence bumps self_recognition + temporal_continuity.
        seed = compute_msl_seed_tensor(None, {"I": 1.0})
        for idx in _CONCEPT_TO_DIMS["I"]:
            assert seed[idx] == pytest.approx(1.0, abs=0.001)

    def test_concept_case_insensitive(self):
        # lowercase concept keys also work (defensive).
        seed_upper = compute_msl_seed_tensor(None, {"YES": 1.0})
        seed_lower = compute_msl_seed_tensor(None, {"yes": 1.0})
        for idx in _CONCEPT_TO_DIMS["YES"]:
            assert seed_upper[idx] == seed_lower[idx]

    def test_attention_and_concept_combine_via_max(self):
        # If both attention and concept map to the same dim, the higher
        # nudge wins (max — doesn't compound). YES → inner_spirit[22]
        # truth_seeking (idx 42); pattern attention → outer_mind[3]
        # problem_solving (idx 73). Different dims, no overlap; just
        # check that both bumps apply independently.
        seed = compute_msl_seed_tensor(
            {"pattern": 0.8}, {"YES": 0.6})
        # pattern → idx 73 at 0.5 + 0.8*0.5 = 0.9
        assert seed[73] == pytest.approx(0.9, abs=0.001)
        # YES → idx 42 at 0.5 + 0.6*0.5 = 0.8
        assert seed[42] == pytest.approx(0.8, abs=0.001)

    def test_unknown_modality_ignored(self):
        # An unknown modality name is silently dropped (no crash).
        seed = compute_msl_seed_tensor({"made_up": 0.9}, None)
        assert all(v == _NEUTRAL for v in seed)

    def test_negative_weight_ignored(self):
        # Defensive: negative weights are treated as "no contribution".
        seed = compute_msl_seed_tensor({"visual": -0.5}, None)
        assert all(v == _NEUTRAL for v in seed)

    def test_string_weight_ignored(self):
        # Non-numeric weight is silently skipped.
        seed = compute_msl_seed_tensor({"visual": "high"}, None)
        assert all(v == _NEUTRAL for v in seed)

    def test_output_clamped_to_unit_interval(self):
        # Pathological input — over-saturated weight — output still ∈ [0,1].
        seed = compute_msl_seed_tensor({"visual": 100.0}, {"I": 100.0})
        for v in seed:
            assert 0.0 <= v <= 1.0


class TestSeedNewWordFeltTensor:
    def test_returns_none_when_no_state(self):
        assert seed_new_word_felt_tensor() is None
        assert seed_new_word_felt_tensor(None, None) is None

    def test_returns_json_string_when_state_supplied(self):
        result = seed_new_word_felt_tensor({"visual": 0.5}, None)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == _TENSOR_LEN

    def test_seed_matches_compute_msl_seed_tensor(self):
        # The JSON-wrapping helper must produce the same vector as the
        # underlying compute function.
        attn = {"visual": 0.7}
        concepts = {"I": 0.8}
        direct = compute_msl_seed_tensor(attn, concepts)
        via_helper = json.loads(seed_new_word_felt_tensor(attn, concepts))
        assert direct == via_helper


class TestBulkBootstrapWordGrounding:
    @pytest.fixture
    def vocab_db(self):
        """Create a temp vocabulary DB with mixed-state rows."""
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE vocabulary ("
            "word TEXT PRIMARY KEY, "
            "felt_tensor TEXT, "
            "confidence REAL DEFAULT 0.5)"
        )
        # 3 ungrounded rows (NULL), 1 empty-list, 1 wrong-length, 1 already grounded.
        good_tensor = json.dumps([0.5] * _TENSOR_LEN)
        rows = [
            ("hello", None, 0.5),
            ("world", None, 0.5),
            ("titan", None, 0.5),
            ("empty_list_word", "[]", 0.5),
            ("wrong_length", json.dumps([0.5, 0.5, 0.5]), 0.5),
            ("already_grounded", good_tensor, 0.9),
        ]
        conn.executemany(
            "INSERT INTO vocabulary VALUES (?, ?, ?)", rows)
        conn.commit()
        conn.close()
        yield db_path
        os.unlink(db_path)

    def test_bulk_seeds_only_ungrounded_rows(self, vocab_db):
        result = bulk_bootstrap_word_grounding(
            vocab_db,
            attention_weights={"visual": 0.5},
            concept_confidences={"I": 0.5},
        )
        # checked = 6 total rows
        # seeded = hello, world, titan (NULL) + empty_list_word ([])
        #        + wrong_length (wrong size) = 5
        # skipped = already_grounded = 1
        assert result["checked"] == 6
        assert result["seeded"] == 5
        assert result["skipped"] == 1
        assert result["errors"] == 0

    def test_bulk_seeds_with_force_reseeds_all(self, vocab_db):
        result = bulk_bootstrap_word_grounding(
            vocab_db,
            attention_weights={"visual": 0.5},
            force=True,
        )
        assert result["checked"] == 6
        assert result["seeded"] == 6  # force re-seeds even good rows
        assert result["skipped"] == 0

    def test_bulk_idempotent_without_force(self, vocab_db):
        # First run seeds the ungrounded rows.
        first = bulk_bootstrap_word_grounding(
            vocab_db, attention_weights={"visual": 0.5})
        assert first["seeded"] == 5
        # Second run should skip everything (all now grounded).
        second = bulk_bootstrap_word_grounding(
            vocab_db, attention_weights={"visual": 0.5})
        assert second["seeded"] == 0
        assert second["skipped"] == 6

    def test_bulk_actually_writes_tensor(self, vocab_db):
        bulk_bootstrap_word_grounding(
            vocab_db, attention_weights={"visual": 1.0})
        conn = sqlite3.connect(vocab_db)
        cur = conn.cursor()
        cur.execute(
            "SELECT felt_tensor FROM vocabulary WHERE word = 'hello'")
        row = cur.fetchone()
        conn.close()
        assert row is not None
        ft = json.loads(row[0])
        assert isinstance(ft, list)
        assert len(ft) == _TENSOR_LEN
        # visual attention → idx 12 (inner_sight) should be bumped
        assert ft[12] == pytest.approx(1.0, abs=0.001)

    def test_bulk_handles_nonexistent_db(self, tmp_path):
        nonexistent = str(tmp_path / "does_not_exist.db")
        result = bulk_bootstrap_word_grounding(nonexistent)
        # Should not raise; errors logged + returned.
        assert result["errors"] >= 1
        assert result["seeded"] == 0
