"""
titan_plugin/logic/arc/forward_model.py — Action-Effect Prediction for ARC.

Learns to predict what the grid will look like after taking an action.
This is Titan's "imagination" for puzzle-solving — evaluate all actions
before choosing, enabling 1-step lookahead planning.

Architecture: (30D features + 7D action one-hot) → 64 → 32 → 30D predicted features
Training: MSE on actual vs predicted next-step features from replay buffer.

Integration:
  - ArcSession records (state, action, next_state) transitions during play
  - Forward model trains on replay buffer between steps (lightweight)
  - Before action selection: predict next_features for ALL available actions
  - Score each predicted state using action-scorer → pick best predicted outcome
  - CGN grounding: high prediction accuracy = "understood" pattern

See: titan-docs/rFP_arc_persona_capability_upgrade.md (Phase A1)
"""
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class ForwardModel:
    """Predicts next grid features from current features + action.

    Enables lookahead planning: "if I do action X, what will happen?"
    This is a primitive form of imagination — the first step toward
    deliberate reasoning about puzzle strategies.

    Uses simple numpy MLP (no torch dependency in ARC hot path).
    """

    def __init__(self, feature_dim: int = 30, num_actions: int = 7,
                 hidden_1: int = 64, hidden_2: int = 32,
                 learning_rate: float = 0.001):
        self._feature_dim = feature_dim
        self._num_actions = num_actions
        self._input_dim = feature_dim + num_actions  # 37D
        self._output_dim = feature_dim  # 30D
        self._lr = learning_rate

        # Network weights (Xavier initialization)
        scale1 = np.sqrt(2.0 / self._input_dim)
        scale2 = np.sqrt(2.0 / hidden_1)
        scale3 = np.sqrt(2.0 / hidden_2)

        self._W1 = np.random.randn(self._input_dim, hidden_1).astype(np.float32) * scale1
        self._b1 = np.zeros(hidden_1, dtype=np.float32)
        self._W2 = np.random.randn(hidden_1, hidden_2).astype(np.float32) * scale2
        self._b2 = np.zeros(hidden_2, dtype=np.float32)
        self._W3 = np.random.randn(hidden_2, self._output_dim).astype(np.float32) * scale3
        self._b3 = np.zeros(self._output_dim, dtype=np.float32)

        # Replay buffer: (state_features, action, next_state_features)
        self._buffer: list[tuple[np.ndarray, int, np.ndarray]] = []
        self._max_buffer = 5000

        # Stats
        self.total_updates = 0
        self.total_predictions = 0
        self._recent_losses: list[float] = []
        self._prediction_errors: list[float] = []

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _action_one_hot(self, action: int) -> np.ndarray:
        oh = np.zeros(self._num_actions, dtype=np.float32)
        if 0 <= action < self._num_actions:
            oh[action] = 1.0
        return oh

    def predict(self, features: np.ndarray, action: int) -> np.ndarray:
        """Predict next grid features given current features and action.

        Args:
            features: 30D current grid features (from GridPerception)
            action: action index (0-6)

        Returns:
            30D predicted next features
        """
        x = np.concatenate([features, self._action_one_hot(action)])
        h1 = self._relu(x @ self._W1 + self._b1)
        h2 = self._relu(h1 @ self._W2 + self._b2)
        out = h2 @ self._W3 + self._b3  # Linear output (features can be any range)
        self.total_predictions += 1
        return out

    def predict_all_actions(self, features: np.ndarray,
                            available_actions: list[int]) -> dict[int, np.ndarray]:
        """Predict next features for ALL available actions (1-step lookahead).

        Args:
            features: 30D current grid features
            available_actions: list of available action indices

        Returns:
            Dict mapping action → predicted 30D features
        """
        predictions = {}
        for action in available_actions:
            predictions[action] = self.predict(features, action)
        return predictions

    def record_transition(self, features: np.ndarray, action: int,
                          next_features: np.ndarray) -> None:
        """Record a (state, action, next_state) transition for training."""
        self._buffer.append((
            features.copy().astype(np.float32),
            action,
            next_features.copy().astype(np.float32),
        ))
        if len(self._buffer) > self._max_buffer:
            self._buffer = self._buffer[-self._max_buffer:]

    def train_step(self, batch_size: int = 32) -> float:
        """Train on a random batch from replay buffer.

        Returns MSE loss (0.0 if insufficient data).
        """
        if len(self._buffer) < batch_size:
            return 0.0

        # Sample random batch
        indices = np.random.choice(len(self._buffer), batch_size, replace=False)

        total_loss = 0.0
        for idx in indices:
            features, action, target = self._buffer[idx]
            x = np.concatenate([features, self._action_one_hot(action)])

            # Forward pass
            z1 = x @ self._W1 + self._b1
            h1 = self._relu(z1)
            z2 = h1 @ self._W2 + self._b2
            h2 = self._relu(z2)
            pred = h2 @ self._W3 + self._b3

            # Loss: MSE
            error = pred - target
            loss = np.mean(error ** 2)
            total_loss += loss

            # Backward pass (manual gradient descent)
            # Gradient of MSE: 2 * error / output_dim
            d_out = 2 * error / self._output_dim

            # Layer 3: d_out → d_h2
            d_W3 = np.outer(h2, d_out)
            d_b3 = d_out
            d_h2 = d_out @ self._W3.T

            # ReLU gradient
            d_z2 = d_h2 * (z2 > 0).astype(np.float32)

            # Layer 2: d_z2 → d_h1
            d_W2 = np.outer(h1, d_z2)
            d_b2 = d_z2
            d_h1 = d_z2 @ self._W2.T

            # ReLU gradient
            d_z1 = d_h1 * (z1 > 0).astype(np.float32)

            # Layer 1: d_z1 → d_x
            d_W1 = np.outer(x, d_z1)
            d_b1 = d_z1

            # Gradient clip
            max_grad = 1.0
            for g in [d_W1, d_b1, d_W2, d_b2, d_W3, d_b3]:
                np.clip(g, -max_grad, max_grad, out=g)

            # Update weights
            self._W1 -= self._lr * d_W1
            self._b1 -= self._lr * d_b1
            self._W2 -= self._lr * d_W2
            self._b2 -= self._lr * d_b2
            self._W3 -= self._lr * d_W3
            self._b3 -= self._lr * d_b3

        avg_loss = total_loss / batch_size
        self.total_updates += 1
        self._recent_losses.append(avg_loss)
        if len(self._recent_losses) > 100:
            self._recent_losses = self._recent_losses[-100:]
        return avg_loss

    def prediction_accuracy(self, features: np.ndarray, action: int,
                            actual_next: np.ndarray) -> float:
        """Measure how well the model predicted the actual outcome.

        Returns cosine similarity between predicted and actual (0.0-1.0).
        """
        predicted = self.predict(features, action)
        # Cosine similarity
        dot = np.dot(predicted, actual_next)
        norm_p = np.linalg.norm(predicted) + 1e-10
        norm_a = np.linalg.norm(actual_next) + 1e-10
        similarity = max(0.0, dot / (norm_p * norm_a))
        self._prediction_errors.append(1.0 - similarity)
        if len(self._prediction_errors) > 100:
            self._prediction_errors = self._prediction_errors[-100:]
        return similarity

    def get_stats(self) -> dict:
        return {
            "buffer_size": len(self._buffer),
            "total_updates": self.total_updates,
            "total_predictions": self.total_predictions,
            "avg_loss": round(np.mean(self._recent_losses), 6) if self._recent_losses else 0.0,
            "avg_prediction_error": round(np.mean(self._prediction_errors), 4) if self._prediction_errors else 0.0,
            "avg_prediction_accuracy": round(1.0 - np.mean(self._prediction_errors), 4) if self._prediction_errors else 0.0,
        }

    def save(self, path: str) -> bool:
        """Save model weights and buffer to JSON."""
        try:
            state = {
                "feature_dim": self._feature_dim,
                "num_actions": self._num_actions,
                "W1": self._W1.tolist(), "b1": self._b1.tolist(),
                "W2": self._W2.tolist(), "b2": self._b2.tolist(),
                "W3": self._W3.tolist(), "b3": self._b3.tolist(),
                "total_updates": self.total_updates,
                "total_predictions": self.total_predictions,
                "buffer_size": len(self._buffer),
            }
            # Save last 1000 buffer entries for warm-start
            if self._buffer:
                state["buffer"] = [
                    {"f": t[0].tolist(), "a": t[1], "nf": t[2].tolist()}
                    for t in self._buffer[-1000:]
                ]
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(state, f)
            os.replace(tmp_path, path)
            return True
        except Exception as e:
            logger.warning("[ForwardModel] Save failed: %s", e)
            return False

    def load(self, path: str) -> bool:
        """Load model weights and buffer from JSON."""
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                state = json.load(f)
            self._W1 = np.array(state["W1"], dtype=np.float32)
            self._b1 = np.array(state["b1"], dtype=np.float32)
            self._W2 = np.array(state["W2"], dtype=np.float32)
            self._b2 = np.array(state["b2"], dtype=np.float32)
            self._W3 = np.array(state["W3"], dtype=np.float32)
            self._b3 = np.array(state["b3"], dtype=np.float32)
            self.total_updates = state.get("total_updates", 0)
            self.total_predictions = state.get("total_predictions", 0)
            # Restore buffer
            for bt in state.get("buffer", []):
                self._buffer.append((
                    np.array(bt["f"], dtype=np.float32),
                    bt["a"],
                    np.array(bt["nf"], dtype=np.float32),
                ))
            logger.info("[ForwardModel] Loaded: %d updates, %d buffer entries",
                        self.total_updates, len(self._buffer))
            return True
        except Exception as e:
            logger.warning("[ForwardModel] Load failed: %s", e)
            return False
