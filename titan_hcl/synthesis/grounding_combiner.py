"""§7.E — offline learned grounding combiner.

A thin logistic model over the 4 decomposed grounding axes (used / verified /
felt / fluent), trained OFFLINE at the dream boundary to predict recall-CITATION
from `engram_recall_events`. It becomes ACTIVE only when it BEATS the §7.D
percentile-blend on held-out citation prediction (self-gating); until then (and
whenever it loses) `EngramStore` falls back to the §7.D blend — so shipping it is
always safe.

Inference is an O(1) dot-product `sigmoid(w·axes + b)` over the RAW axes — no
population read, no heavy dependency. Training lazy-imports scikit-learn (idle-
gated, dream boundary / boot only) and persists ONLY the fitted coefficients as
plain JSON, so load + score never import it. With the current near-degenerate
data (citation ≈ constant until the §7.E.0 cited=false events accumulate) the
guard short-circuits before any sklearn import and the model stays inactive.

Scope-fence (§7.E): offline only; no continuous net; falls back to the blend;
designed to be subsumed by BRAIN `time_cost`.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from typing import Any, Callable, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

# Fixed feature order — MUST match `engram_store._AXIS_NAMES` (the persisted
# weights are positional). verified is procedural-only (≈0 on thought-Engrams,
# §6.2.3); the learner naturally assigns it ~0 weight — kept for BRAIN-compat.
AXES: tuple[str, ...] = ("used", "verified", "felt", "fluent")

# Self-gating thresholds. A learned model REPLACES a principled population blend,
# so it must clearly WIN, not tie — conservative on purpose.
_MIN_SAMPLES = 200      # below this the fit is unstable → keep the blend
_MIN_PER_CLASS = 30     # need a real negative class (fed by the §7.E.0 fix)
_MIN_AUC = 0.55         # must beat chance
_AUC_MARGIN = 0.02      # must beat the blend by a clear margin


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _axes_vec(axes: Any) -> list[float]:
    """Coerce an axes record (dict keyed by AXES, or a positional sequence) to a
    fixed-order [used, verified, felt, fluent] float vector."""
    if isinstance(axes, dict):
        return [float(axes.get(k, 0.0) or 0.0) for k in AXES]
    seq = list(axes or [])
    return [float(seq[i] or 0.0) if i < len(seq) else 0.0
            for i in range(len(AXES))]


class GroundingCombiner:
    """Persisted logistic combiner. Inactive (falls back to the §7.D blend) until
    a train-step ACTIVATES it by beating the blend on held-out citation AUC."""

    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        bias: float = 0.0,
        active: bool = False,
        meta: Optional[dict] = None,
    ) -> None:
        self._w: Optional[list[float]] = (
            [float(x) for x in weights] if weights is not None else None)
        self._b = float(bias)
        self._active = bool(active) and self._w is not None and len(self._w) == len(AXES)
        self.meta: dict = dict(meta or {})

    # ── Persistence ──────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: Optional[str]) -> "GroundingCombiner":
        """Load from JSON; a missing / corrupt / unset file → inactive default
        (safe fallback to the blend). Never raises."""
        try:
            if path and os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    d = json.load(f)
                return cls(weights=d.get("weights"), bias=d.get("bias", 0.0),
                           active=bool(d.get("active", False)), meta=d.get("meta"))
        except Exception as e:  # noqa: BLE001
            logger.warning("[GroundingCombiner] load failed (%s) — inactive", e)
        return cls()

    def save(self, path: str) -> bool:
        """Atomic JSON write (tmp + os.replace). Soft-fail → False."""
        try:
            payload = {"weights": self._w, "bias": self._b, "active": self._active,
                       "axes": list(AXES), "meta": self.meta}
            d = os.path.dirname(os.path.abspath(path))
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("[GroundingCombiner] save failed: %s", e)
            return False

    # ── Inference (O(1), dependency-free) ────────────────────────────────
    def is_active(self) -> bool:
        return self._active and self._w is not None

    def score(self, axes: Any) -> float:
        """Learned groundedness ∈ [0,1] = sigmoid(w·axes + b). Inactive → 0.0
        (the caller branches on is_active() and uses the §7.D blend instead)."""
        if not self.is_active():
            return 0.0
        x = _axes_vec(axes)
        z = self._b + sum(w * v for w, v in zip(self._w, x))  # type: ignore[arg-type]
        return _sigmoid(z)

    # ── Training (offline, lazy sklearn) ─────────────────────────────────
    def train(
        self,
        events: Iterable[tuple[Any, bool]],
        *,
        blend_scorer: Callable[[list[list[float]]], list[float]],
        clock: Optional[Callable[[], float]] = None,
    ) -> dict:
        """Fit a logistic over the RAW axes to predict `cited` and ACTIVATE iff it
        beats the §7.D blend on held-out citation AUC by a margin.

        `events`: iterable of (axes-record, cited-bool). `blend_scorer(axes_list)
        -> [score]` returns the §7.D percentile-blend score for a list of axes
        vectors (injected by EngramStore to avoid a circular import — the model
        must beat the very reduction it would replace). Never raises (soft-fail →
        keep prior state). Returns a metrics dict."""
        ev = [(_axes_vec(a), bool(c)) for (a, c) in (events or [])]
        n = len(ev)
        pos = sum(1 for _, c in ev if c)
        neg = n - pos
        out: dict = {"n": n, "pos": pos, "neg": neg, "activated": False}
        if n < _MIN_SAMPLES or pos < _MIN_PER_CLASS or neg < _MIN_PER_CLASS:
            out["reason"] = "insufficient_data"   # short-circuits BEFORE sklearn import
            return out
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import roc_auc_score

            X = np.asarray([x for x, _ in ev], dtype=float)
            y = np.asarray([1 if c else 0 for _, c in ev], dtype=int)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.30, random_state=0, stratify=y)
            if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
                out["reason"] = "degenerate_split"
                return out
            clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)
            clf.fit(Xtr, ytr)
            auc_learned = float(roc_auc_score(yte, clf.decision_function(Xte)))
            # Baseline: the §7.D percentile-blend score on the SAME holdout.
            try:
                blend_scores = blend_scorer([list(row) for row in Xte])
                auc_blend = float(roc_auc_score(yte, blend_scores))
            except Exception as _be:  # noqa: BLE001
                logger.debug("[GroundingCombiner] blend baseline failed: %s", _be)
                auc_blend = 0.5
            out["auc_learned"] = round(auc_learned, 4)
            out["auc_blend"] = round(auc_blend, 4)

            if auc_learned >= _MIN_AUC and auc_learned >= auc_blend + _AUC_MARGIN:
                self._w = [float(c) for c in clf.coef_[0]]
                self._b = float(clf.intercept_[0])
                self._active = True
                out["activated"] = True
                self.meta = {"auc_learned": out["auc_learned"],
                             "auc_blend": out["auc_blend"], "n": n,
                             "trained_at": float(clock() if clock else 0.0)}
            else:
                out["reason"] = "did_not_beat_blend"
                if self._active and auc_learned < auc_blend:
                    # Previously active but a fresh fit no longer beats the blend
                    # (data shifted) → fall back to §7.D.
                    self._active = False
                    out["deactivated"] = True
            return out
        except Exception as e:  # noqa: BLE001
            logger.warning("[GroundingCombiner] train failed: %s", e)
            out["reason"] = f"error:{e}"
            return out
