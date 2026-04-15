"""
titan_plugin/logic/chain_iql.py — TUNING-012 v2 Sub-phase B.

Chain-Level IQL Hierarchy: a SECOND learning layer above per-primitive
compound rewards. Learns "given THIS task, which CHAIN TEMPLATE works best?"

Architecture:
  LAYER 3 (this module): P(chain_template | task_embedding)
                          Reward = chain success on the concrete task
                          Learning = offline regression on chain_outcomes buffer
  LAYER 2 (Sub-phase A):  P(primitive | meta_state, neuromods)
                          Reward = compound per-step from memory+timechain+contracts
  LAYER 1 (existing):     Reasoning Engine chain execution

Implementation: pure numpy MLP, matching the existing MetaPolicy style.
The Q-net is tiny (~3.5K params) so torch is overkill — keeps the lazy
imports test green and avoids any new dependency.

Persistence: JSON file alongside meta_policy.json. Buffer + template
registry survive worker restarts so chain learning compounds over time.

See: titan-docs/rFP_tuning_012_compound_rewards_v2.md §7.B
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from collections import deque
from typing import Any

import numpy as np

from titan_plugin.logic.task_embedding import (
    encode_task,
    extract_chain_template,
    template_to_primitive_list,
)

logger = logging.getLogger("titan.chain_iql")


# ── Q-Network (pure numpy MLP) ────────────────────────────────────


class ChainTemplateQNet:
    """Q(task_embedding, template_id) → predicted task_success in [0, 1].

    Architecture: (task_dim + template_emb_dim) → hidden → 1
    Activation: ReLU. Output is unbounded — clipped to [0, 1] at inference.
    Loss: MSE between Q(task, template) and observed task_success.

    Pure numpy. Forward + backward implemented in matching style to
    titan_plugin.logic.meta_reasoning.MetaPolicy.
    """

    def __init__(
        self,
        task_dim: int = 32,
        template_count: int = 50,
        template_emb_dim: int = 16,
        hidden: int = 64,
        lr: float = 0.001,
    ):
        self.task_dim = task_dim
        self.template_count = template_count
        self.template_emb_dim = template_emb_dim
        self.hidden = hidden
        self.lr = lr
        rng = np.random.default_rng(0)
        # Embedding table for chain templates
        self.template_emb = rng.standard_normal(
            (template_count, template_emb_dim)
        ).astype(np.float32) * 0.1
        # MLP weights — He initialization
        in_dim = task_dim + template_emb_dim
        self.w1 = rng.standard_normal((in_dim, hidden)).astype(np.float32) * math.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = rng.standard_normal((hidden, 1)).astype(np.float32) * math.sqrt(2.0 / hidden)
        self.b2 = np.zeros(1, dtype=np.float32)
        self.total_updates = 0

    def forward(
        self,
        task_emb: np.ndarray,
        template_id: int,
    ) -> tuple[float, dict]:
        """Forward pass. Returns (Q value, intermediate cache for backward)."""
        if template_id < 0 or template_id >= self.template_count:
            template_id = template_id % self.template_count
        t_emb = self.template_emb[template_id]
        x = np.concatenate([task_emb.astype(np.float32), t_emb])
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ self.w2 + self.b2
        cache = {
            "x": x, "z1": z1, "h1": h1, "z2": z2,
            "template_id": template_id, "t_emb": t_emb,
        }
        return float(z2[0]), cache

    def predict(self, task_emb: np.ndarray, template_id: int) -> float:
        """Inference-only forward. Returns clipped Q in [0, 1]."""
        q, _ = self.forward(task_emb, template_id)
        return float(max(0.0, min(1.0, q)))

    def train_step(
        self,
        task_emb: np.ndarray,
        template_id: int,
        target: float,
    ) -> float:
        """Single SGD update on MSE loss. Returns the loss value."""
        q, cache = self.forward(task_emb, template_id)
        # Loss = 0.5 * (q - target)^2 → dL/dq = (q - target)
        d_q = q - float(target)
        loss = 0.5 * d_q * d_q

        # Backward through w2/b2
        d_z2 = np.array([d_q], dtype=np.float32)
        d_w2 = cache["h1"].reshape(-1, 1) @ d_z2.reshape(1, -1)
        d_b2 = d_z2

        # Backward through w1/b1 (ReLU)
        d_h1 = d_z2 @ self.w2.T
        d_z1 = d_h1 * (cache["z1"] > 0).astype(np.float32)
        d_w1 = cache["x"].reshape(-1, 1) @ d_z1.reshape(1, -1)
        d_b1 = d_z1

        # Backward through input → template_emb gradient
        d_x = d_z1 @ self.w1.T
        d_t_emb = d_x[self.task_dim:]

        # SGD update with gradient clipping (norm 1.0)
        for g in (d_w1, d_w2, d_t_emb):
            n = float(np.linalg.norm(g))
            if n > 1.0:
                g *= (1.0 / n)
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        self.template_emb[cache["template_id"]] -= self.lr * d_t_emb
        self.total_updates += 1
        return float(loss)

    def to_dict(self) -> dict:
        return {
            "task_dim": self.task_dim,
            "template_count": self.template_count,
            "template_emb_dim": self.template_emb_dim,
            "hidden": self.hidden,
            "lr": self.lr,
            "total_updates": self.total_updates,
            "template_emb": self.template_emb.tolist(),
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
        }

    def from_dict(self, d: dict) -> None:
        self.template_emb = np.array(d.get("template_emb", []), dtype=np.float32)
        if self.template_emb.size == 0:
            return
        self.w1 = np.array(d["w1"], dtype=np.float32)
        self.b1 = np.array(d["b1"], dtype=np.float32)
        self.w2 = np.array(d["w2"], dtype=np.float32)
        self.b2 = np.array(d["b2"], dtype=np.float32)
        self.total_updates = int(d.get("total_updates", 0))


# ── Chain IQL Manager ─────────────────────────────────────────────


class ChainIQL:
    """Manages chain-level Q-net training, template registry, and best-template lookup.

    Lifecycle:
      __init__       — load DNA, init Q-net, load persisted state
      record_chain_outcome — called from MetaReasoningEngine._conclude_chain
                             when each chain finishes. Adds to buffer.
      query_best_template  — called from MetaReasoningEngine._start_chain.
                             Returns the highest-Q template (or None) for the
                             given task embedding.
      consolidate_during_dream — called from MetaReasoningEngine.consolidate_training
                                 during dream cycles. Trains Q-net on buffer.
      save / load    — JSON persistence alongside meta_policy.json
    """

    def __init__(self, dna: dict | None = None, save_dir: str = "./data/reasoning"):
        dna = dna or {}
        self._task_dim = int(dna.get("task_embedding_dim", 32))
        self._template_max = int(dna.get("chain_template_max_count", 50))
        self._lr = float(dna.get("chain_qnet_lr", 0.001))
        self._buffer_size = int(dna.get("chain_iql_buffer_size", 1000))
        self._blend_alpha = float(dna.get("chain_blend_alpha", 0.5))
        self._enabled = bool(dna.get("chain_iql_enabled", True))

        self.qnet = ChainTemplateQNet(
            task_dim=self._task_dim,
            template_count=self._template_max,
            lr=self._lr,
        )
        # Template registry: template_string → integer id (slot in qnet.template_emb)
        self.template_registry: dict[str, int] = {}
        # Buffer of (task_emb, template_id, task_success, primitives, domain)
        self.buffer: deque = deque(maxlen=self._buffer_size)
        # Phase D.1: counter for external rewards that arrived after buffer
        # eviction (expected late-drop case). Exposed in get_stats() for tuning.
        self._external_reward_late_drops = 0
        # Save path
        self._save_path = os.path.join(save_dir, "chain_iql.json")
        os.makedirs(save_dir, exist_ok=True)
        self.load()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def blend_alpha(self) -> float:
        return self._blend_alpha

    def get_or_assign_template_id(self, template: str) -> int:
        """Look up the int id for a template string. Assigns a new slot if novel."""
        if template in self.template_registry:
            return self.template_registry[template]
        if len(self.template_registry) >= self._template_max:
            # Registry full — recycle the least-used slot. For now, simple
            # FIFO eviction by reusing slot 0 (oldest). A more sophisticated
            # eviction strategy can come later.
            return 0
        new_id = len(self.template_registry)
        self.template_registry[template] = new_id
        return new_id

    def record_chain_outcome(
        self,
        task_emb: np.ndarray,
        chain: list,
        task_success: float,
        primitives: list[str],
        domain: str,
        chain_id: int = -1,
    ) -> None:
        """Add a completed chain outcome to the training buffer.

        Phase D.1: chain_id is carried through to allow later external
        reward blending via apply_external_reward(). Default -1 means
        "no tracking" (legacy callers continue to work unchanged).
        """
        if not self._enabled:
            return
        template = extract_chain_template(chain)
        if not template:
            return
        template_id = self.get_or_assign_template_id(template)
        self.buffer.append({
            "task_emb": task_emb.tolist() if isinstance(task_emb, np.ndarray) else list(task_emb),
            "template": template,
            "template_id": template_id,
            "task_success": float(max(0.0, min(1.0, task_success))),
            "primitives": list(primitives),
            "domain": str(domain or "general"),
            "chain_id": int(chain_id),
        })

    def apply_external_reward(
        self,
        chain_id: int,
        external_reward: float,
        alpha: float,
    ) -> bool:
        """Phase D.1 — blend an externally-measured reward into a recorded chain.

        Option B: in-place buffer entry update. Searches newest → oldest
        for the entry with matching chain_id and blends:
            new_task_success = old_task_success * (1 - alpha) + external_reward * alpha

        Args:
            chain_id: the chain to correlate against (from MetaChainState).
            external_reward: reward in [0, 1] (will be clipped).
            alpha: blend weight in [0, 1] (will be clipped).

        Returns:
            True if the entry was found and updated, False if the chain
            has already been evicted from the buffer (late-drop case,
            increments self._external_reward_late_drops counter).
        """
        if not self._enabled or chain_id < 0:
            return False
        ext = float(max(0.0, min(1.0, external_reward)))
        a = float(max(0.0, min(1.0, alpha)))
        # Newest → oldest: external rewards correlate with recent chains
        for i in range(len(self.buffer) - 1, -1, -1):
            entry = self.buffer[i]
            if entry.get("chain_id", -1) == chain_id:
                old = float(entry["task_success"])
                entry["task_success"] = old * (1.0 - a) + ext * a
                entry["external_applied"] = True
                return True
        self._external_reward_late_drops += 1
        return False

    def query_best_template(
        self,
        task_emb: np.ndarray,
        min_q: float = 0.0,
    ) -> tuple[str | None, float]:
        """Return (best_template_string, q_value) for the given task embedding.

        Searches all known templates and returns the one with the highest Q.
        Returns (None, 0.0) if no templates are registered or all Qs are
        below min_q.
        """
        if not self._enabled or not self.template_registry:
            return None, 0.0
        best_template = None
        best_q = -1.0
        for template, tid in self.template_registry.items():
            q = self.qnet.predict(task_emb, tid)
            if q > best_q:
                best_q = q
                best_template = template
        if best_q < min_q:
            return None, 0.0
        return best_template, best_q

    def query_top_k_templates(
        self,
        task_emb: np.ndarray,
        k: int = 3,
        min_q: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Return top-K templates sorted by Q descending (META-CGN Phase 4).

        Used by META-CGN active-mode reranking: β scores each of the top-K
        candidates via compose_template_score, blended score reranks. Falls
        back to empty list if registry empty or all Qs below min_q.

        Returns list of (template_string, q_value) — may be shorter than k
        if registry has <k templates.
        """
        if not self._enabled or not self.template_registry:
            return []
        scored: list[tuple[str, float]] = []
        for template, tid in self.template_registry.items():
            q = self.qnet.predict(task_emb, tid)
            if q >= min_q:
                scored.append((template, float(q)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:max(1, k)]

    def template_visit_count(self, template: str) -> int:
        """Count buffer entries associated with this template (for λ compute).

        Used by META-CGN composition to gauge chain_iql's confidence in a
        specific template pick. Returns 0 if template unknown.
        """
        tid = self.template_registry.get(template)
        if tid is None:
            return 0
        return sum(1 for e in self.buffer if e.get("template_id") == tid)

    def consolidate_during_dream(self, batch_size: int = 32) -> dict:
        """Train the Q-net on a STRATIFIED batch from the buffer.

        Called from MetaReasoningEngine.consolidate_training (during dream
        cycles). Returns training stats.

        TUNING-012 v2 Phase D pre-flight: stratified sampling replaces uniform
        random.sample to fix buffer-imbalance bias. When the policy collapses
        on one primitive (T1: 62% FORMULATE in measured snapshot), the buffer
        is dominated by that template's outcomes. Uniform random.sample then
        gives the Q-net mostly the dominant template's gradient, and rare
        templates never escape their initial Q estimates — exactly the
        behavior TUNING-012 is supposed to fix. Stratified sampling groups
        the buffer by template_id and takes ~equal samples from each, so
        rare-but-high-success templates get adequate training signal.

        Algorithm:
          1. Group buffer entries by template_id
          2. Take ceil(batch_size / num_templates) per group, capped at
             available count
          3. Top up with random remainder if some groups were undersized
          4. Truncate if num_templates > batch_size (one per template at most)
        """
        if not self._enabled or len(self.buffer) < batch_size:
            return {"trained": False, "buffer_size": len(self.buffer)}

        # Group buffer entries by template_id (preserves identity)
        by_tid: dict[int, list] = {}
        for entry in self.buffer:
            tid = entry.get("template_id", -1)
            by_tid.setdefault(tid, []).append(entry)

        num_templates = len(by_tid)
        if num_templates == 0:
            return {"trained": False, "buffer_size": len(self.buffer)}

        # Stratified sample: aim for batch_size / num_templates per group
        per_template = max(1, batch_size // num_templates)
        batch: list = []
        for tid, items in by_tid.items():
            n_take = min(per_template, len(items))
            batch.extend(random.sample(items, n_take))

        # Top up with uniform random remainder (rare templates may have
        # undersupplied; we still want a full batch_size)
        if len(batch) < batch_size:
            # Use id() to avoid double-sampling the same dict entry
            sampled_ids = {id(s) for s in batch}
            remainder = [s for s in self.buffer if id(s) not in sampled_ids]
            if remainder:
                n_extra = min(batch_size - len(batch), len(remainder))
                batch.extend(random.sample(remainder, n_extra))

        # Truncate if num_templates > batch_size (rare; would mean very
        # diverse buffer and small batch)
        if len(batch) > batch_size:
            batch = random.sample(batch, batch_size)

        total_loss = 0.0
        for sample in batch:
            task_emb = np.array(sample["task_emb"], dtype=np.float32)
            tid = int(sample["template_id"])
            target = float(sample["task_success"])
            loss = self.qnet.train_step(task_emb, tid, target)
            total_loss += loss

        avg_loss = total_loss / max(1, len(batch))
        return {
            "trained": True,
            "samples": len(batch),
            "avg_loss": round(avg_loss, 6),
            "buffer_size": len(self.buffer),
            "template_count": len(self.template_registry),
            "total_updates": self.qnet.total_updates,
            "stratified_groups": num_templates,
        }

    def get_stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "buffer_size": len(self.buffer),
            "template_count": len(self.template_registry),
            "total_updates": self.qnet.total_updates,
            "blend_alpha": self._blend_alpha,
            "external_reward_late_drops": self._external_reward_late_drops,
        }

    def save(self) -> None:
        try:
            payload = {
                "qnet": self.qnet.to_dict(),
                "template_registry": self.template_registry,
                "buffer": list(self.buffer),
            }
            tmp = self._save_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self._save_path)
        except Exception as e:
            logger.warning("[ChainIQL] Save error: %s", e)

    def load(self) -> None:
        if not os.path.exists(self._save_path):
            return
        try:
            with open(self._save_path) as f:
                payload = json.load(f)
            self.qnet.from_dict(payload.get("qnet", {}))
            self.template_registry = dict(payload.get("template_registry", {}))
            saved_buffer = payload.get("buffer", [])
            self.buffer = deque(saved_buffer, maxlen=self._buffer_size)
            logger.info(
                "[ChainIQL] Loaded: templates=%d, buffer=%d, updates=%d",
                len(self.template_registry), len(self.buffer), self.qnet.total_updates,
            )
        except Exception as e:
            logger.warning("[ChainIQL] Load error: %s", e)
