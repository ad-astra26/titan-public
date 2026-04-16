"""
titan_plugin/logic/meta_autoencoder.py — Contrastive autoencoder for problem embedding.

Learns compressed 16D representation from 132D Unified Spirit state.
Trained during dreams with dual objective:
  1. Reconstruction: minimize MSE(input, decode(encode(input)))
  2. Contrastive: similar-outcome chains cluster, different-outcome chains separate

Used by: ChainArchive and MetaWisdomStore for embedding-based similarity search.

Architecture: ~10K params, ~2s per dream training cycle.
"""

import json
import logging
import math
import os
import random
from typing import Optional

import numpy as np

logger = logging.getLogger("titan.meta_autoencoder")

INPUT_DIM = 132
HIDDEN_DIM = 32
EMBED_DIM = 16


class MetaAutoencoder:
    """Contrastive autoencoder: 132D Unified Spirit → 16D problem embedding."""

    def __init__(self, save_dir: str = "./data/reasoning",
                 learning_rate: float = 0.001,
                 contrastive_weight: float = 0.3):
        self._save_dir = save_dir
        self._lr = learning_rate
        self._contrastive_weight = contrastive_weight
        self._training_steps = 0
        self._momentum = 0.9
        os.makedirs(save_dir, exist_ok=True)

        # Encoder: 132 → 32 (ReLU) → 16 (tanh)
        self._enc_w1 = None  # 132×32
        self._enc_b1 = None  # 32
        self._enc_w2 = None  # 32×16
        self._enc_b2 = None  # 16

        # Decoder: 16 → 32 (ReLU) → 132 (sigmoid)
        self._dec_w1 = None  # 16×32
        self._dec_b1 = None  # 32
        self._dec_w2 = None  # 32×132
        self._dec_b2 = None  # 132

        # Momentum buffers
        self._velocity = {}

        self._init_weights()
        self._load()

    def _init_weights(self) -> None:
        """Xavier initialization."""
        def xavier(fan_in, fan_out):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

        self._enc_w1 = xavier(INPUT_DIM, HIDDEN_DIM)
        self._enc_b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self._enc_w2 = xavier(HIDDEN_DIM, EMBED_DIM)
        self._enc_b2 = np.zeros(EMBED_DIM, dtype=np.float32)

        self._dec_w1 = xavier(EMBED_DIM, HIDDEN_DIM)
        self._dec_b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self._dec_w2 = xavier(HIDDEN_DIM, INPUT_DIM)
        self._dec_b2 = np.zeros(INPUT_DIM, dtype=np.float32)

        self._velocity = {}

    def encode(self, state_132d: list) -> list:
        """Encode 132D → 16D embedding. Returns zeros if input too short."""
        x = np.array(state_132d[:INPUT_DIM], dtype=np.float32)
        if len(x) < INPUT_DIM:
            x = np.pad(x, (0, INPUT_DIM - len(x)))

        # Layer 1: ReLU
        h = np.maximum(0, x @ self._enc_w1 + self._enc_b1)
        # Layer 2: tanh (bounded [-1, 1])
        emb = np.tanh(h @ self._enc_w2 + self._enc_b2)
        return emb.tolist()

    def decode(self, embedding_16d: list) -> list:
        """Decode 16D → 132D reconstruction."""
        z = np.array(embedding_16d[:EMBED_DIM], dtype=np.float32)
        if len(z) < EMBED_DIM:
            z = np.pad(z, (0, EMBED_DIM - len(z)))

        # Layer 1: ReLU
        h = np.maximum(0, z @ self._dec_w1 + self._dec_b1)
        # Layer 2: sigmoid (bounded [0, 1])
        out = 1.0 / (1.0 + np.exp(-np.clip(h @ self._dec_w2 + self._dec_b2, -15, 15)))
        return out.tolist()

    def dream_train(
        self,
        chain_archive,
        batch_size: int = 32,
        contrastive_margin: float = 0.5,
    ) -> dict:
        """Train during dreams on archived chains.

        Returns stats dict with training metrics.
        """
        chains = chain_archive.get_unconsolidated(limit=200)
        chains_with_obs = [c for c in chains if c.get("observation_snapshot")
                           and len(c["observation_snapshot"]) >= 65]

        if len(chains_with_obs) < 10:
            return {"trained": False, "reason": "insufficient_chains",
                    "available": len(chains_with_obs)}

        # Sample batch
        batch = random.sample(chains_with_obs, min(batch_size, len(chains_with_obs)))

        total_recon_loss = 0.0
        total_contrastive_loss = 0.0
        n_steps = 0

        for chain in batch:
            obs = chain["observation_snapshot"][:INPUT_DIM]
            if len(obs) < INPUT_DIM:
                obs = obs + [0.5] * (INPUT_DIM - len(obs))
            x = np.array(obs, dtype=np.float32)

            # Forward pass
            h1 = np.maximum(0, x @ self._enc_w1 + self._enc_b1)
            emb = np.tanh(h1 @ self._enc_w2 + self._enc_b2)
            h2 = np.maximum(0, emb @ self._dec_w1 + self._dec_b1)
            logits = h2 @ self._dec_w2 + self._dec_b2
            recon = 1.0 / (1.0 + np.exp(-np.clip(logits, -15, 15)))

            # Reconstruction loss (MSE)
            recon_loss = np.mean((x - recon) ** 2)
            total_recon_loss += recon_loss

            # Backprop reconstruction
            d_recon = 2.0 * (recon - x) / INPUT_DIM
            d_logits = d_recon * recon * (1.0 - recon)  # sigmoid derivative

            d_dec_w2 = np.outer(h2, d_logits)
            d_dec_b2 = d_logits
            d_h2 = d_logits @ self._dec_w2.T
            d_h2 *= (h2 > 0).astype(np.float32)  # ReLU derivative

            d_dec_w1 = np.outer(emb, d_h2)
            d_dec_b1 = d_h2
            d_emb = d_h2 @ self._dec_w1.T

            # Contrastive loss: pair with random chain
            if len(chains_with_obs) > 1:
                partner = random.choice(chains_with_obs)
                p_obs = partner["observation_snapshot"][:INPUT_DIM]
                if len(p_obs) < INPUT_DIM:
                    p_obs = p_obs + [0.5] * (INPUT_DIM - len(p_obs))
                p_x = np.array(p_obs, dtype=np.float32)
                p_h1 = np.maximum(0, p_x @ self._enc_w1 + self._enc_b1)
                p_emb = np.tanh(p_h1 @ self._enc_w2 + self._enc_b2)

                # Euclidean distance in embedding space
                dist = np.sqrt(np.sum((emb - p_emb) ** 2) + 1e-8)
                outcome_diff = abs(chain["outcome_score"] - partner["outcome_score"])

                if outcome_diff < 0.15:
                    # Positive pair: should be close
                    c_loss = max(0.0, dist - contrastive_margin * 0.3)
                    if c_loss > 0:
                        d_contrast = (emb - p_emb) / dist
                    else:
                        d_contrast = np.zeros_like(emb)
                elif outcome_diff > 0.30:
                    # Negative pair: should be far
                    c_loss = max(0.0, contrastive_margin - dist)
                    if c_loss > 0:
                        d_contrast = -(emb - p_emb) / dist
                    else:
                        d_contrast = np.zeros_like(emb)
                else:
                    c_loss = 0.0
                    d_contrast = np.zeros_like(emb)

                total_contrastive_loss += c_loss
                d_emb += self._contrastive_weight * d_contrast

            # Backprop through encoder
            d_emb_pre_tanh = d_emb * (1.0 - emb ** 2)  # tanh derivative
            d_enc_w2 = np.outer(h1, d_emb_pre_tanh)
            d_enc_b2 = d_emb_pre_tanh
            d_h1 = d_emb_pre_tanh @ self._enc_w2.T
            d_h1 *= (h1 > 0).astype(np.float32)  # ReLU derivative

            d_enc_w1 = np.outer(x, d_h1)
            d_enc_b1 = d_h1

            # SGD with momentum
            self._sgd_update("enc_w1", self._enc_w1, d_enc_w1)
            self._sgd_update("enc_b1", self._enc_b1, d_enc_b1)
            self._sgd_update("enc_w2", self._enc_w2, d_enc_w2)
            self._sgd_update("enc_b2", self._enc_b2, d_enc_b2)
            self._sgd_update("dec_w1", self._dec_w1, d_dec_w1)
            self._sgd_update("dec_b1", self._dec_b1, d_dec_b1)
            self._sgd_update("dec_w2", self._dec_w2, d_dec_w2)
            self._sgd_update("dec_b2", self._dec_b2, d_dec_b2)

            n_steps += 1
            self._training_steps += 1

        # Backfill embeddings for chains that don't have them
        embeddings_updated = self.backfill_embeddings(chain_archive)

        avg_recon = total_recon_loss / max(n_steps, 1)
        avg_contrast = total_contrastive_loss / max(n_steps, 1)

        return {
            "trained": True,
            "recon_loss": round(float(avg_recon), 6),
            "contrastive_loss": round(float(avg_contrast), 6),
            "samples": n_steps,
            "embeddings_updated": embeddings_updated,
            "total_steps": self._training_steps,
        }

    def _sgd_update(self, name: str, param: np.ndarray, grad: np.ndarray) -> None:
        """SGD with momentum, in-place."""
        if name not in self._velocity:
            self._velocity[name] = np.zeros_like(param)
        v = self._velocity[name]
        v[:] = self._momentum * v + self._lr * grad
        param -= v

    def backfill_embeddings(self, chain_archive) -> int:
        """Encode chains that have snapshots but no embeddings.

        Computes all encodings in memory first, then persists them in a single
        batched transaction. Prior implementation did N separate commits, which
        under SQLite contention produced 17-minute dream cycles (see rFP
        inner_memory_db_write_contention.md § Layer 3 investigation + INVESTIGATION
        _spirit_hang_root_cause.md, 2026-04-16).
        """
        if not self.is_trained:
            return 0
        chains = chain_archive.get_chains_without_embedding(limit=100)
        pairs = []
        for chain in chains:
            obs = chain["observation_snapshot"]
            if len(obs) >= 65:
                emb = self.encode(obs)
                pairs.append((chain["id"], emb))
        if not pairs:
            return 0
        return chain_archive.update_embeddings_batch(pairs)

    @property
    def is_trained(self) -> bool:
        return self._training_steps >= 100

    def save(self) -> None:
        path = os.path.join(self._save_dir, "meta_autoencoder.json")
        try:
            data = {
                "training_steps": self._training_steps,
                "enc_w1": self._enc_w1.tolist(),
                "enc_b1": self._enc_b1.tolist(),
                "enc_w2": self._enc_w2.tolist(),
                "enc_b2": self._enc_b2.tolist(),
                "dec_w1": self._dec_w1.tolist(),
                "dec_b1": self._dec_b1.tolist(),
                "dec_w2": self._dec_w2.tolist(),
                "dec_b2": self._dec_b2.tolist(),
            }
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning("[MetaAutoencoder] Save error: %s", e)

    def _load(self) -> None:
        path = os.path.join(self._save_dir, "meta_autoencoder.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._training_steps = data.get("training_steps", 0)
            self._enc_w1 = np.array(data["enc_w1"], dtype=np.float32)
            self._enc_b1 = np.array(data["enc_b1"], dtype=np.float32)
            self._enc_w2 = np.array(data["enc_w2"], dtype=np.float32)
            self._enc_b2 = np.array(data["enc_b2"], dtype=np.float32)
            self._dec_w1 = np.array(data["dec_w1"], dtype=np.float32)
            self._dec_b1 = np.array(data["dec_b1"], dtype=np.float32)
            self._dec_w2 = np.array(data["dec_w2"], dtype=np.float32)
            self._dec_b2 = np.array(data["dec_b2"], dtype=np.float32)
            logger.info("[MetaAutoencoder] Loaded weights (steps=%d, trained=%s)",
                        self._training_steps, self.is_trained)
        except Exception as e:
            logger.warning("[MetaAutoencoder] Load error: %s", e)
            self._init_weights()

    @staticmethod
    def cosine_similarity(emb_a: list, emb_b: list) -> float:
        if len(emb_a) != len(emb_b) or not emb_a:
            return 0.0
        dot = sum(x * y for x, y in zip(emb_a, emb_b))
        mag_a = math.sqrt(sum(x * x for x in emb_a))
        mag_b = math.sqrt(sum(x * x for x in emb_b))
        if mag_a < 1e-8 or mag_b < 1e-8:
            return 0.0
        return dot / (mag_a * mag_b)
