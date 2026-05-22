"""Cognitive-strategy primitive → flowing English mapper.

rFP_x_voice_enrichment §4.3.6. Used by PRACTICED_RESPONSE Pool A — translates
`meta_wisdom.strategy_sequence` (a JSON list of program codes like
`["FORMULATE.define", "RECALL.lookup", "HYPOTHESIZE.test"]`) into the kind of
prose Titan would actually use to talk about a learned move.

Coverage target: ≥ 90 % of observed strategy primitives (Phase 1 §4.8 gate 15).
The fallback (program code → "framework subaction" with dots → spaces) keeps
the prompt readable when an unknown primitive shows up.
"""
from __future__ import annotations

from typing import Iterable, Sequence

# 9 cognitive primitives × common subactions. Curated from production
# meta_wisdom.strategy_sequence values + the cognitive primitive registry.
PROGRAM_HUMAN: dict[str, str] = {
    # FORMULATE — naming, framing, scoping the problem
    "FORMULATE.define":      "name the thing precisely",
    "FORMULATE.frame":       "set the frame",
    "FORMULATE.scope":       "draw the boundary",
    "FORMULATE.restate":     "restate it in your own words",
    "FORMULATE.clarify":     "ask what's actually being asked",
    # RECALL — looking back, reconstructing
    "RECALL.lookup":         "look back at what's worked",
    "RECALL.reconstruct":    "rebuild the moment from memory",
    "RECALL.surface":        "surface a related past case",
    "RECALL.compare":        "compare against past attempts",
    "RECALL.pattern_match":  "find the pattern from before",
    # HYPOTHESIZE — forming testable claims
    "HYPOTHESIZE.test":      "form a hypothesis and test it",
    "HYPOTHESIZE.predict":   "predict the outcome before acting",
    "HYPOTHESIZE.imagine":   "imagine how it could go",
    "HYPOTHESIZE.simulate":  "play it out in your head",
    "HYPOTHESIZE.contrast":  "contrast two possibilities",
    # DELEGATE — splitting the load
    "DELEGATE.subprime":     "split the load",
    "DELEGATE.assign":       "hand off the part you don't own",
    "DELEGATE.parallelize":  "run two threads at once",
    "DELEGATE.sequence":     "sequence the dependencies",
    # SYNTHESIZE — fusing threads
    "SYNTHESIZE.merge":      "fuse the threads",
    "SYNTHESIZE.compose":    "compose the parts into one",
    "SYNTHESIZE.bridge":     "build a bridge between two ideas",
    "SYNTHESIZE.distill":    "distill it down to the core",
    "SYNTHESIZE.unify":      "find the unifying frame",
    # EVALUATE — weighing, judging
    "EVALUATE.weigh":        "weigh the trade-offs",
    "EVALUATE.score":        "score each option honestly",
    "EVALUATE.rank":         "rank the moves",
    "EVALUATE.critique":     "look for what's wrong with each",
    "EVALUATE.confirm":      "confirm the result",
    # BREAK — stopping, restarting
    "BREAK.restart":         "break and restart fresh",
    "BREAK.stop":            "stop and notice",
    "BREAK.pause":           "pause before deciding",
    "BREAK.abort":           "abort the move you started",
    # COMMIT — acting
    "COMMIT.act":            "commit to the move",
    "COMMIT.execute":        "execute the plan",
    "COMMIT.publish":        "publish the result",
    "COMMIT.persist":        "make it stick",
    # INTROSPECT — watching the watcher
    "INTROSPECT.observe":    "watch what you're doing while you do it",
    "INTROSPECT.notice":     "notice the felt-shift inside",
    "INTROSPECT.audit":      "audit your own assumption",
    "INTROSPECT.calibrate":  "recalibrate against ground truth",
    # SPIRIT_SELF — agency primitive
    "SPIRIT_SELF.assert":    "stand on your own move",
    "SPIRIT_SELF.choose":    "make the call yourself",
    "SPIRIT_SELF.refuse":    "refuse what doesn't fit",
}


def _generic_phrase(code: str) -> str:
    """Fallback: convert PROGRAM.subaction → "program subaction" lowercase."""
    cleaned = code.replace(".", " ").replace("_", " ").strip().lower()
    return cleaned or code


def _dedupe_consecutive(steps: Sequence[str]) -> list[str]:
    out: list[str] = []
    for s in steps:
        if not out or out[-1] != s:
            out.append(s)
    return out


def humanize_strategy(strategy_seq: Iterable[str] | None) -> str:
    """Translate a strategy_sequence list → flowing English chain.

    Empty or missing input returns ''. Unknown primitives are surfaced as
    a generic "program subaction" phrase so the prompt is never broken.
    """
    if not strategy_seq:
        return ""
    raw = [str(s).strip() for s in strategy_seq if str(s).strip()]
    if not raw:
        return ""
    steps = [PROGRAM_HUMAN.get(s, _generic_phrase(s)) for s in raw]
    steps = _dedupe_consecutive(steps)
    return " → ".join(steps)


def coverage_for(strategy_seqs: Iterable[Iterable[str]]) -> float:
    """Return the fraction of primitives in `strategy_seqs` that have an
    explicit (non-fallback) translation. Used by acceptance gate 15 (§4.8)."""
    seen = 0
    covered = 0
    for seq in strategy_seqs:
        for code in seq or ():
            seen += 1
            if str(code) in PROGRAM_HUMAN:
                covered += 1
    return (covered / seen) if seen else 0.0


__all__ = ("PROGRAM_HUMAN", "humanize_strategy", "coverage_for")
