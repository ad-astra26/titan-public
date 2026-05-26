"""titan_hcl/logic/vocabulary_grounding.py — L5+L6 housekeeping closure.

Closes long-standing `rFP_higher_cognition_roadmap.md §Remaining CGN Items`
deferrals (2026-05-26 housekeeping audit):

  L5 — Bulk `bootstrap_word_grounding()` for all existing words. Admin
       function that iterates vocabulary rows whose `felt_tensor` is
       NULL or empty and seeds a 130D tensor from the current MSL state
       snapshot.

  L6 — MSL-informed tensor seeding for NEW word acquisition. Replaces
       the previous NULL initialization at INSERT time with a seed
       computed from the same MSL state, so a freshly-acquired word
       carries a meaningful felt-tensor representing the Titan's state
       at the moment of acquisition.

Design principle: a felt_tensor entry is a 130D vector of how a word
"feels" — i.e., which trinity dims were engaged when the word was
acquired. Seeding from MSL state at acquisition time gives a
principled non-zero baseline (instead of NULL, which forces consumers
to special-case empty tensors). The mapping below is the Maker-locked
canonical attention→dim and concept→dim association — additive to
the existing vocabulary contract, never replacing already-seeded
tensors unless the caller passes ``force=True``.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Trinity 130D layout (matches dim_registry._BLOCKS):
#   [0:5)   inner_body 5D
#   [5:20)  inner_mind 15D
#   [20:65) inner_spirit 45D
#   [65:70) outer_body 5D
#   [70:85) outer_mind 15D
#   [85:130) outer_spirit 45D
_TENSOR_LEN = 130
_NEUTRAL = 0.5

# Attention modality → list of full-tensor dim indices that the modality
# semantically maps to. Bumps these dims above the neutral baseline by
# the modality's normalized weight.
_ATTN_MODALITY_TO_DIMS: dict[str, list[int]] = {
    # visual attention → inner_sight (mind[7]→idx 12), outer_mind
    # research_effectiveness (idx 70) + knowledge_retrieval (idx 71)
    "visual":     [12, 70, 71],
    # audio attention → inner_hearing (mind[5]→idx 10)
    "audio":      [10],
    # pattern attention → outer_mind problem_solving (idx 73)
    "pattern":    [73],
    # inner_body attention → all 5 body dims
    "inner_body": [0, 1, 2, 3, 4],
    # inner_mind attention → thinking dims (mind[0:5] → idx 5-9)
    "inner_mind": [5, 6, 7, 8, 9],
    # outer_body attention → all 5 outer_body dims
    "outer_body": [65, 66, 67, 68, 69],
    # neuromod attention → willing dims (mind[10:15] → idx 15-19)
    "neuromod":   [15, 16, 17, 18, 19],
}

# Concept confidence → list of full-tensor dim indices the concept
# semantically maps to. Same nudge mechanism as attention.
_CONCEPT_TO_DIMS: dict[str, list[int]] = {
    # I → inner_spirit self_recognition (spirit[0]→20) + temporal_continuity
    # (spirit[4]→24)
    "I":    [20, 24],
    # YOU → outer_spirit community_connection (idx 121) + outer_mind
    # social_temperature/social_connection (idx 75, 76)
    "YOU":  [75, 76, 121],
    # WE → outer_spirit engagement_depth (idx 108) + outer_mind social
    # (idx 75, 76)
    "WE":   [75, 76, 108],
    # YES → inner_spirit truth_seeking (idx 42)
    "YES":  [42],
    # NO → inner_spirit boundary_clarity (idx 23) + outer_spirit
    # boundary_enforcement (idx 88)
    "NO":   [23, 88],
    # THEY → outer_mind social (idx 75, 76, 79)
    "THEY": [75, 76, 79],
}

# How strongly to nudge a dim above neutral. Modality/concept weight is
# multiplied by this scalar; result is added to neutral baseline 0.5,
# capped at 1.0. A modality weight of 1.0 with full weight = 0.5 + 0.5 *
# 1.0 = 1.0 (full saturation); a 0.1 modality weight nudges by 0.05.
_NUDGE_SCALE = 0.5


def compute_msl_seed_tensor(
    attention_weights: Optional[dict] = None,
    concept_confidences: Optional[dict] = None,
    *,
    length: int = _TENSOR_LEN,
) -> list[float]:
    """Compute a 130D seed felt_tensor from current MSL state.

    Args:
        attention_weights: dict of {modality_name: weight} (7 modalities),
            typically from ``MSL.get_attention_weights_for_kin()`` or
            ``msl._last_output['attention_weights']``. None or empty →
            no attention nudges.
        concept_confidences: dict of {concept_name: confidence ∈ [0,1]}
            for ``I``/``YOU``/``WE``/``YES``/``NO``/``THEY``, typically
            from ``ConceptGrounder.get_concept_confidences()``. None or
            empty → no concept nudges.
        length: tensor length (default 130 = trinity full tensor).

    Returns:
        list[float] of length ``length``, all values in [0.0, 1.0].

    A neutral seed (no MSL state) returns the all-0.5 vector; the more
    engaged a modality or concept is, the higher its mapped dims rise.
    """
    out = [_NEUTRAL] * length

    if attention_weights:
        for modality, raw_weight in attention_weights.items():
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                continue
            if weight <= 0.0:
                continue
            for idx in _ATTN_MODALITY_TO_DIMS.get(modality, []):
                if 0 <= idx < length:
                    nudge = min(1.0 - _NEUTRAL, weight * _NUDGE_SCALE)
                    out[idx] = max(out[idx], _NEUTRAL + nudge)

    if concept_confidences:
        for concept, raw_conf in concept_confidences.items():
            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                continue
            if conf <= 0.0:
                continue
            for idx in _CONCEPT_TO_DIMS.get(concept.upper(), []):
                if 0 <= idx < length:
                    nudge = min(1.0 - _NEUTRAL, conf * _NUDGE_SCALE)
                    out[idx] = max(out[idx], _NEUTRAL + nudge)

    # Defensive clamp (max() above protects upper bound; this catches any
    # rounding excess).
    return [max(0.0, min(1.0, v)) for v in out]


def bulk_bootstrap_word_grounding(
    db_path: str,
    attention_weights: Optional[dict] = None,
    concept_confidences: Optional[dict] = None,
    *,
    force: bool = False,
    batch_size: int = 500,
) -> dict:
    """Iterate vocabulary rows; seed felt_tensor for words that lack one.

    Args:
        db_path: path to the inner_memory database holding the
            ``vocabulary`` table.
        attention_weights: current MSL attention vector (see
            ``compute_msl_seed_tensor``).
        concept_confidences: current MSL concept confidences (same).
        force: when True, re-seed even words whose felt_tensor is already
            non-empty. Default False (only seed where felt_tensor is
            NULL or empty list).
        batch_size: SQL batch UPDATE size (for memory safety on large
            vocabularies).

    Returns:
        dict with keys:
          - ``checked``     (int): total rows examined
          - ``seeded``      (int): rows that received a new felt_tensor
          - ``skipped``     (int): rows already grounded and ``force``
                                   was False
          - ``errors``      (int): rows where the write failed (logged)

    Idempotent (when ``force=False``): re-running is a no-op until new
    words are acquired.
    """
    import sqlite3

    result = {"checked": 0, "seeded": 0, "skipped": 0, "errors": 0}
    seed = compute_msl_seed_tensor(attention_weights, concept_confidences)
    seed_json = json.dumps(seed)

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # SELECT word + felt_tensor — small payload per row.
        cur.execute("SELECT word, felt_tensor FROM vocabulary")
        rows = cur.fetchall()
        result["checked"] = len(rows)

        to_update: list[tuple[str, str]] = []
        for row in rows:
            word = row["word"]
            ft = row["felt_tensor"]
            if not force:
                # Skip if felt_tensor is a populated JSON list of correct
                # length. Treat any parse error / wrong length / empty
                # as "needs seeding".
                if ft:
                    try:
                        parsed = json.loads(ft)
                        if (isinstance(parsed, list)
                                and len(parsed) == _TENSOR_LEN):
                            result["skipped"] += 1
                            continue
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
            to_update.append((seed_json, word))

        # Batch UPDATE.
        for i in range(0, len(to_update), batch_size):
            batch = to_update[i:i + batch_size]
            try:
                cur.executemany(
                    "UPDATE vocabulary SET felt_tensor = ? WHERE word = ?",
                    batch,
                )
                conn.commit()
                result["seeded"] += len(batch)
            except sqlite3.Error as e:
                logger.warning(
                    "[VocabBootstrap] batch UPDATE failed (size %d): %s",
                    len(batch), e)
                result["errors"] += len(batch)
        conn.close()
    except Exception as e:
        logger.error("[VocabBootstrap] %s: %s", db_path, e)
        result["errors"] += 1

    logger.info(
        "[VocabBootstrap] db=%s checked=%d seeded=%d skipped=%d errors=%d "
        "force=%s", db_path, result["checked"], result["seeded"],
        result["skipped"], result["errors"], force)
    return result


def seed_new_word_felt_tensor(
    attention_weights: Optional[dict] = None,
    concept_confidences: Optional[dict] = None,
) -> Optional[str]:
    """L6 hook for INSERT-time seeding. Returns the JSON-encoded seed
    tensor, or None if no MSL state is supplied (caller falls back to
    NULL, preserving current contract).

    Use at vocabulary INSERT sites to populate felt_tensor with a
    meaningful baseline derived from MSL state at acquisition time,
    instead of leaving the column NULL.
    """
    if not attention_weights and not concept_confidences:
        return None
    seed = compute_msl_seed_tensor(attention_weights, concept_confidences)
    return json.dumps(seed)
