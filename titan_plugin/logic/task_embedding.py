"""
titan_plugin/logic/task_embedding.py — TUNING-012 v2 Sub-phase B helpers.

Encodes a meta-reasoning "task" into a fixed-size embedding vector and
extracts normalized chain templates from primitive sequences.

A "task" in meta-reasoning is the (domain, trigger_reason, state_context)
the Titan is currently working on. Two chains with the same task should
hash to similar embeddings — the chain-level Q-net learns "given THIS
kind of task, which CHAIN TEMPLATE has historically worked best?"

Pure numpy + hashlib. No torch dependency. No lazy import concerns.

See: titan-docs/rFP_tuning_012_compound_rewards_v2.md §7.B
"""

from __future__ import annotations

import hashlib

import numpy as np


# Fixed projection matrix for the state-vector half of the embedding.
# Deterministic seed makes embeddings reproducible across restarts.
_STATE_PROJ_SEED = 42
_STATE_VEC_DIM = 132


def _state_projection(target_dim: int) -> np.ndarray:
    """Return a deterministic 132 → target_dim projection matrix."""
    rng = np.random.default_rng(_STATE_PROJ_SEED)
    return rng.standard_normal((_STATE_VEC_DIM, target_dim)).astype(np.float32) * 0.1


# Cache the projection matrix per dim — typically dim=16 for the half-embedding
_PROJ_CACHE: dict = {}


def _get_proj(target_dim: int) -> np.ndarray:
    if target_dim not in _PROJ_CACHE:
        _PROJ_CACHE[target_dim] = _state_projection(target_dim)
    return _PROJ_CACHE[target_dim]


def _hash_categorical(domain: str, trigger_reason: str) -> int:
    """Stable hash of (domain, trigger_reason). Identical inputs → identical seed."""
    payload = f"{domain or '_'}::{trigger_reason or '_'}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def encode_task(
    domain: str | None,
    trigger_reason: str | None,
    state_vector: list | np.ndarray | None,
    dim: int = 32,
) -> np.ndarray:
    """Encode a meta-reasoning task into a fixed-size embedding vector.

    The first half of the embedding is a deterministic hash of the
    categorical fields (domain + trigger_reason). The second half is a
    linear projection of the 132D state vector. Two chains with the same
    domain+trigger but different state vectors will share the categorical
    half and differ in the state half, so the Q-net can learn task-level
    patterns while still distinguishing situational variants.

    Args:
        domain:         e.g. "body_mind", "inner_spirit", "outer_spirit"
        trigger_reason: e.g. "low_commit_rate", "high_reflection"
        state_vector:   132D float vector (current consciousness state)
        dim:            output embedding dimension (must be even)

    Returns:
        np.ndarray of shape (dim,), dtype float32
    """
    if dim < 4 or dim % 2 != 0:
        raise ValueError(f"task embedding dim must be even and >= 4, got {dim}")
    half = dim // 2

    # Categorical half: deterministic RNG seeded by hash of (domain, trigger)
    seed = _hash_categorical(domain or "", trigger_reason or "")
    cat_rng = np.random.default_rng(seed)
    cat_emb = cat_rng.standard_normal(half).astype(np.float32)

    # State half: project the 132D state vector through a fixed random matrix
    if state_vector is None:
        state_emb = np.zeros(half, dtype=np.float32)
    else:
        sv = np.asarray(list(state_vector)[:_STATE_VEC_DIM], dtype=np.float32)
        if sv.size < _STATE_VEC_DIM:
            sv = np.pad(sv, (0, _STATE_VEC_DIM - sv.size))
        proj = _get_proj(half)
        state_emb = sv @ proj

    return np.concatenate([cat_emb, state_emb]).astype(np.float32)


def extract_chain_template(chain: list) -> str:
    """Extract a normalized template string from a meta-chain.

    Strips sub-modes — only the primitive sequence matters for template
    matching. Identical primitive sequences (regardless of sub-modes)
    map to the same template.

    Example:
        ["FORMULATE.define", "RECALL.chain_archive", "EVALUATE.check_progress"]
        → "FORMULATE→RECALL→EVALUATE"
    """
    if not chain:
        return ""
    prims = []
    for step in chain:
        if not isinstance(step, str):
            continue
        prim = step.split(".")[0]
        prims.append(prim)
    return "→".join(prims)


def template_to_primitive_list(template: str) -> list[str]:
    """Inverse of extract_chain_template — split a template back to a list."""
    if not template:
        return []
    return template.split("→")
