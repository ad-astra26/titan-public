"""introspection — the inward consumer of the text oracle (RFP_text_extraction_introspection Phase B).

Introspection is *research pointed inward*: Titan runs the deterministic `text_oracle`
over his OWN structured self-telemetry → a verifiable self-observation → grounds a
`SELF:<aspect>` concept (via the §7.P3 curiosity grounding path). This module owns the
two pieces that are NOT just "reuse the curiosity loop":

  1. `IntrospectionDamper` — the navel-gaze guard (INV-TX-6, Maker MUST): bounded
     per-window budget + a novelty gate (an unchanged self-fact does not re-ground)
     + a hard refusal of introspection-about-introspection. Self-awareness is a
     seasoning, not the meal.
  2. `run_introspection(...)` — orchestrates damper → corpus → `text_oracle.extract`
     → a narratable self-observation + the grounding payload (`_research_target` shaped
     so it rides the existing curiosity grounding 3a/3b/3c).

Lock-safety (cold-review): the corpus is supplied by a `corpus_provider(aspect)->str`
that the CALLER wires to lock-safe sources (SHM readers / synthesis read endpoints) —
NEVER a direct `mode=ro` open of the live DuckDB the writer holds (it conflicts).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from titan_hcl.synthesis.text_oracle import extract, ExtractResult, ExtractError

# Aspects whose name signals "introspection about introspecting" — refused so the
# faculty cannot recurse into watching-itself-watch-itself (INV-TX-6c).
_SELF_REFERENTIAL = ("introspect", "introspection", "navel", "self_observation")


@dataclass
class IntrospectionDamper:
    """INV-TX-6 — keeps introspection bounded + novel. Pure state machine; the caller
    persists `_window_start`/`_count`/`_last_sha` across calls (or accepts per-process)."""
    max_per_window: int = 4          # introspections allowed per window
    window_s: float = 3600.0         # the rolling window
    _window_start: float = 0.0
    _count: int = 0
    _last_sha: dict[str, str] = field(default_factory=dict)   # aspect → last extract sha

    def _roll(self, now: float) -> None:
        if now - self._window_start >= self.window_s:
            self._window_start = now
            self._count = 0

    def allow(self, aspect: str, *, now: Optional[float] = None) -> tuple[bool, str]:
        """Return (allowed, reason). Refuses self-referential aspects + over-budget."""
        now = time.time() if now is None else now
        a = str(aspect or "").lower()
        if any(s in a for s in _SELF_REFERENTIAL):
            return False, "self-referential (introspection-about-introspection refused)"
        self._roll(now)
        if self._count >= self.max_per_window:
            return False, f"over budget ({self._count}/{self.max_per_window} this window)"
        return True, "ok"

    def is_novel(self, aspect: str, sha: str) -> bool:
        """A self-fact whose extract sha is unchanged from last time is NOT novel —
        it must not re-ground (→ ~0 reward), so introspection converges, not loops."""
        return self._last_sha.get(str(aspect)) != sha

    def commit(self, aspect: str, sha: str, *, now: Optional[float] = None) -> None:
        """Record an accepted introspection (advances budget + remembers the sha)."""
        now = time.time() if now is None else now
        self._roll(now)
        self._count += 1
        self._last_sha[str(aspect)] = sha


@dataclass
class IntrospectionResult:
    aspect: str
    grounded: bool                       # True iff substantive + novel → should ground
    observation: str                     # the human-readable self-observation (narratable)
    extract: Optional[dict] = None       # the verifiable ExtractResult.to_dict()
    research_target: Optional[dict] = None  # rides curiosity 3a/3b/3c if grounded
    reason: str = ""                     # why not grounded (damper/empty), if applicable


def run_introspection(
    aspect: str,
    query: dict,
    corpus_provider: Callable[[str], str],
    damper: IntrospectionDamper,
    *,
    min_evidence_chars: int = 1,
    now: Optional[float] = None,
) -> IntrospectionResult:
    """Damper → corpus → extract → a self-observation + (if novel) a grounding target.

    `corpus_provider(aspect)` MUST be lock-safe (SHM/endpoint readers). Pure besides
    that call + the damper state. Never raises on a bad query — returns un-grounded.
    """
    allowed, why = damper.allow(aspect, now=now)
    if not allowed:
        return IntrospectionResult(aspect=aspect, grounded=False, observation="",
                                   reason=why)
    try:
        corpus = corpus_provider(aspect) or ""
    except Exception as e:  # noqa: BLE001 — a flaky reader must never crash the faculty
        return IntrospectionResult(aspect=aspect, grounded=False, observation="",
                                   reason=f"corpus unavailable: {e}")
    if len(corpus.strip()) < int(min_evidence_chars):
        return IntrospectionResult(aspect=aspect, grounded=False, observation="",
                                   reason="no telemetry to read")
    try:
        res: ExtractResult = extract(corpus, query)
    except ExtractError as e:
        return IntrospectionResult(aspect=aspect, grounded=False, observation="",
                                   reason=f"bad query: {e}")

    observation = _observe(aspect, res)
    if not damper.is_novel(aspect, res.corpus_sha):
        return IntrospectionResult(aspect=aspect, grounded=False, observation=observation,
                                   extract=res.to_dict(),
                                   reason="unchanged since last read (novelty gate)")
    # novel + substantive → ground it (rides the curiosity 3a/3b/3c path)
    damper.commit(aspect, res.corpus_sha, now=now)
    rt = {
        "concept_id": f"SELF:{aspect}",
        "name": f"SELF:{aspect}",
        "domain_hint": "self",
        "baseline_groundedness": 0.0,
        "source": "introspection",
    }
    return IntrospectionResult(aspect=aspect, grounded=True, observation=observation,
                               extract=res.to_dict(), research_target=rt)


def _observe(aspect: str, res: ExtractResult) -> str:
    """A compact, FACTUAL self-observation string over the verified extract — the
    substrate the LLM may later narrate over (it never authors the numbers)."""
    if res.kind == "count" and res.counts:
        top = sorted(res.counts.items(), key=lambda x: -x[1])
        body = ", ".join(f"{k}={v}" for k, v in top[:8])
        return f"[SELF:{aspect}] counts → {body} (n={res.n}, sha={res.corpus_sha})"
    if res.kind == "fields" and res.fields:
        body = ", ".join(f"{k}={v}" for k, v in res.fields.items())
        return f"[SELF:{aspect}] {body} (sha={res.corpus_sha})"
    return (f"[SELF:{aspect}] {res.kind}: {res.n} match(es) "
            f"(sha={res.corpus_sha})")
