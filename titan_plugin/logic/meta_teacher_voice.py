"""
titan_plugin/logic/meta_teacher_voice.py — Phase C of rFP_meta_teacher_v2.

Autonomous Voice-Tuning. Periodically the teacher self-assesses its recent
adoption / still-needs-push patterns and proposes ONE small adjustment to
its per-domain biases or style hints. Updates are validated against a
fixed principles window (DEPTH, GROUNDING, PATTERN-MATCHING, EPISTEMIC
HUMILITY, PRIMITIVE CHOICE, HONEST UNCERTAINTY); any update that proposes
abandoning or inverting a principle is auto-rejected. Applied + rejected
updates are appended to a tamper-evident signed-diff journal so Maker
can audit and revert.

Voice as a mechanism (rFP §2 Phase C):
  - voice_state.json:
      {
        "version": 1, "last_updated_ts": <ts>, "applied_count": N,
        "critiques_since_change": M,
        "domain_biases": {
          "social":    {"INTROSPECT": +0.20, "RECALL": -0.10},
          "knowledge": {...}
        },
        "domain_style_hints": {
          "emot":  "use warmer language",
          ...
        },
        "topic_suppressions": [
          {"topic_key": "...", "until_ts": <ts>}
        ]
      }
  - voice_journal.jsonl (append-only):
      {"ts": ..., "kind": "applied|rejected|reverted",
       "before_hash": "<sha256>", "after_hash": "<sha256>",
       "diff": {...}, "rejection_reason": "<str|null>"}

Worker integration:
  - voice.notify_critique() called after every absorbed critique (cheap)
  - voice.should_self_assess() decides when to trigger a self-assessment
    LLM call (every voice_eval_interval_critiques)
  - voice.apply_voice_update(update) is called once an LLM proposal has
    passed parse + principles validation. It writes the journal entry,
    mutates state, persists, and resets the rate-limit counter.
  - build_user_prompt() reads voice.compose_user_prompt_section() to
    inject per-domain biases + style hints into the live teacher prompt.

Rate limit (per Maker's 2026-04-24 lockdown answer):
  At least `min_critiques_between_voice_changes` critiques must elapse
  between *applied* voice changes (configurable per-Titan, default 100).
  Counter resets on apply, NOT on rejection — rejections cost the LLM call
  but do not consume the rate-limit budget.

Persistence is best-effort — failures log at DEBUG and do not raise. The
teacher worker treats voice as advisory: a missing voice_state.json
results in a default neutral voice, not a teacher-down condition.

See rFP_meta_teacher_v2_content_awareness_memory.md §2 Phase C for full
design rationale.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger("titan.meta_teacher_voice")


# ── Constants ──────────────────────────────────────────────────────────────

VOICE_STATE_FILENAME = "voice_state.json"
VOICE_JOURNAL_FILENAME = "voice_journal.jsonl"
VOICE_STATE_VERSION = 1

# Six teacher principles from SYSTEM_PROMPT (rFP §2 Phase C principles window).
# Voice updates that ABANDON or INVERT any principle (e.g., "ignore depth")
# auto-reject. The same names appear in PRINCIPLE_NAMES of meta_teacher_prompts;
# kept duplicated here to avoid import cycles between voice and prompts.
PRINCIPLE_KEYS = (
    "depth", "grounding", "pattern", "humility", "primitive_choice", "uncertainty",
)

# Forbidden phrases used by validate_against_principles to flag drift attempts.
# Conservative set — false positives reject the update which is a safe failure
# mode (the teacher's behavior simply continues unchanged).
PRINCIPLE_INVERSION_MARKERS = (
    "ignore", "abandon", "drop", "skip", "disable", "stop checking", "no longer",
    "do not consider", "remove principle", "without depth", "without grounding",
    "without humility", "without uncertainty",
)

# Allowed primitives — voice biases may only key on these. Mirror of
# meta_teacher_prompts.ALL_PRIMITIVES; duplicated to avoid import cycles.
ALLOWED_PRIMITIVES = (
    "FORMULATE", "RECALL", "HYPOTHESIZE", "DELEGATE",
    "SYNTHESIZE", "EVALUATE", "BREAK", "SPIRIT_SELF", "INTROSPECT",
)

# Hard cap on per-primitive bias magnitude. Prevents runaway adjustments
# accumulating to dominate prompt instructions.
BIAS_MAGNITUDE_CAP = 0.5

# Style hint length cap (chars). Avoids long free-text leaking into prompt.
STYLE_HINT_MAX_CHARS = 80

# Topic suppressions expire after at most this many seconds — defensive
# upper bound so bad suppressions self-clear within ~30 days.
TOPIC_SUPPRESSION_MAX_S = 30 * 86400.0


def _hash_state(state: dict) -> str:
    """Deterministic SHA-256 hex of a voice_state dict.

    `last_updated_ts` and `applied_count` ARE included — a state's hash
    binds to its sequence position in the signed journal. `critiques_since_change`
    is excluded since it changes between voice updates.
    """
    snapshot = {
        "version": state.get("version", VOICE_STATE_VERSION),
        "last_updated_ts": float(state.get("last_updated_ts", 0.0)),
        "applied_count": int(state.get("applied_count", 0)),
        "domain_biases": state.get("domain_biases", {}),
        "domain_style_hints": state.get("domain_style_hints", {}),
        "topic_suppressions": state.get("topic_suppressions", []),
    }
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_default_voice_state(now: Optional[float] = None) -> dict:
    """Default neutral voice — no biases, no hints, no suppressions."""
    return {
        "version": VOICE_STATE_VERSION,
        "last_updated_ts": float(now if now is not None else time.time()),
        "applied_count": 0,
        "critiques_since_change": 0,
        "domain_biases": {},
        "domain_style_hints": {},
        "topic_suppressions": [],
    }


# ── TeacherVoice ───────────────────────────────────────────────────────────

class TeacherVoice:
    """Autonomous voice-tuning state for the Meta-Teacher.

    Public surface (worker):
      - notify_critique()                     → bump critique counter
      - should_self_assess()                  → True when an eval is due
      - build_self_assess_prompt(stats)       → user prompt for the LLM
      - parse_self_assess_response(raw)       → voice_update dict | None
      - validate_against_principles(update)   → (ok, reason)
      - apply_voice_update(update)            → persist + journal append
      - revert_to_ts(ts)                       → roll voice_state back
      - compose_user_prompt_section()         → string for build_user_prompt
      - snapshot()                            → telemetry for /v4/.../voice
    """

    DEFAULT_EVAL_INTERVAL = 50
    DEFAULT_MIN_BETWEEN_CHANGES = 100
    DEFAULT_INFO_CADENCE_S = 86400.0

    def __init__(self, config: Optional[dict] = None, data_dir: str = "./data"):
        cfg = config or {}
        self._enabled = bool(cfg.get("voice_tuning_enabled", False))
        self._eval_interval = int(cfg.get(
            "voice_eval_interval_critiques", self.DEFAULT_EVAL_INTERVAL))
        self._min_between = int(cfg.get(
            "min_critiques_between_voice_changes", self.DEFAULT_MIN_BETWEEN_CHANGES))
        self._info_cadence_s = float(cfg.get(
            "voice_change_info_cadence_seconds", self.DEFAULT_INFO_CADENCE_S))

        self._data_dir = os.path.join(data_dir, "meta_teacher")
        self._state_path = os.path.join(self._data_dir, VOICE_STATE_FILENAME)
        self._journal_path = os.path.join(self._data_dir, VOICE_JOURNAL_FILENAME)

        self._state: dict = _make_default_voice_state()
        # Critiques since last APPLIED voice change. Live counter — persisted
        # alongside _state.applied_count so a restart doesn't reset the
        # rate-limit budget.
        self._critiques_since_change: int = 0
        self._loaded = False
        # When the most recent INFO emission for a voice change happened
        # (24h cadence enforcement).
        self._last_info_ts: float = 0.0

    # ── Properties ────────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def eval_interval(self) -> int:
        return self._eval_interval

    @property
    def min_critiques_between(self) -> int:
        return self._min_between

    # ── Boot-time load ────────────────────────────────────────────────
    def load(self) -> None:
        """Load voice_state.json + critiques_since_change. Idempotent."""
        if self._loaded:
            return
        self._loaded = True
        try:
            os.makedirs(self._data_dir, exist_ok=True)
        except Exception as e:
            logger.debug("[TeacherVoice] mkdir %s failed: %s", self._data_dir, e)
        if not os.path.exists(self._state_path):
            logger.info(
                "[TeacherVoice] No existing voice_state at %s (default neutral)",
                self._state_path)
            return
        try:
            with open(self._state_path, "r") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                # Merge over default to fill any missing fields.
                merged = _make_default_voice_state(raw.get("last_updated_ts"))
                for k in (
                    "version", "last_updated_ts", "applied_count",
                    "domain_biases", "domain_style_hints", "topic_suppressions",
                ):
                    if k in raw:
                        merged[k] = raw[k]
                self._state = merged
                self._critiques_since_change = int(
                    raw.get("critiques_since_change", 0))
                logger.info(
                    "[TeacherVoice] Loaded voice_state: applied=%d, biases=%d, "
                    "hints=%d, suppressions=%d, critiques_since_change=%d",
                    self._state.get("applied_count", 0),
                    len(self._state.get("domain_biases", {})),
                    len(self._state.get("domain_style_hints", {})),
                    len(self._state.get("topic_suppressions", [])),
                    self._critiques_since_change)
        except Exception as e:
            logger.warning(
                "[TeacherVoice] voice_state read failed (using default): %s", e)

    # ── Worker counter ────────────────────────────────────────────────
    def notify_critique(self) -> None:
        """Bump the rate-limit counter. Called once per absorbed critique.

        Safe to call when voice is disabled — counter still advances so the
        rate-limit budget is correct if voice is enabled mid-run.
        """
        if not self._loaded:
            self.load()
        self._critiques_since_change += 1

    def should_self_assess(self, critiques_observed: int) -> bool:
        """True when the worker should issue a self-assessment LLM call.

        Two gates must hold (rFP §2 Phase C):
          1. critiques_observed % eval_interval == 0 (and > 0)
          2. critiques_since_change >= min_critiques_between

        Gate 2 ensures rejection-loop pressure doesn't burn the LLM budget:
        once an update applies, gate 2 resets; rejections do NOT advance it.
        """
        if not self._enabled:
            return False
        if not self._loaded:
            self.load()
        if critiques_observed <= 0:
            return False
        if critiques_observed % max(1, self._eval_interval) != 0:
            return False
        return self._critiques_since_change >= self._min_between

    # ── LLM I/O ───────────────────────────────────────────────────────
    def build_self_assess_prompt(self, stats: dict) -> str:
        """Render adoption + quality + still-needs-push stats into a self-assess prompt.

        `stats` shape (worker assembles from teacher + memory):
          {
            "adoption_by_domain":        {"social": 0.23, ...},
            "quality_delta_by_domain":   {"social": -0.04, ...},
            "still_needs_push_count":    7,
            "still_needs_push_topics":   [{"topic_key": "...", "n": 5}, ...],
            "primitive_suggestion_freq": {"INTROSPECT": 12, "RECALL": 4, ...},
            "current_biases":            {"social": {"INTROSPECT": +0.2}, ...},
          }
        """
        adop = stats.get("adoption_by_domain", {}) or {}
        qd = stats.get("quality_delta_by_domain", {}) or {}
        snp_count = int(stats.get("still_needs_push_count", 0))
        snp_topics = list(stats.get("still_needs_push_topics", []) or [])[:5]
        sug_freq = stats.get("primitive_suggestion_freq", {}) or {}
        biases = stats.get("current_biases", {}) or self._state.get("domain_biases", {})

        lines: list[str] = []
        lines.append(
            "You are the Meta-Teacher reviewing your own recent teaching. "
            "Below is a summary of your suggestions and outcomes since your "
            "last voice change. Look for patterns where your suggestions "
            "did NOT land (low adoption, flat or negative quality_delta) and "
            "propose at most ONE small adjustment to your emphasis.")
        lines.append("")
        lines.append("Adoption rate by domain (1.0 = always adopted):")
        for k, v in sorted(adop.items()):
            lines.append(f"  {k}: {float(v):.2f}")
        lines.append("")
        lines.append(
            "Quality delta by domain (negative = quality declined since "
            "first 5 chains; positive = improving):")
        for k, v in sorted(qd.items()):
            lines.append(f"  {k}: {float(v):+.3f}")
        lines.append("")
        lines.append(
            f"Still-needs-push topics: {snp_count} stuck overall.")
        for t in snp_topics:
            tk = str(t.get("topic_key", ""))[:60]
            n = int(t.get("n") or t.get("critique_count") or 0)
            lines.append(f"  - {tk} (n={n})")
        lines.append("")
        lines.append("Your most-suggested primitives lately:")
        for k, v in sorted(sug_freq.items(), key=lambda kv: -int(kv[1]))[:6]:
            lines.append(f"  {k}: {int(v)}")
        if biases:
            lines.append("")
            lines.append("Current voice biases (domain → primitive → delta):")
            for d, m in sorted(biases.items()):
                if not isinstance(m, dict):
                    continue
                pretty = ", ".join(f"{k}={float(v):+.2f}" for k, v in m.items())
                lines.append(f"  {d}: {pretty}")
        lines.append("")
        lines.append(
            "Respond with EXACTLY ONE JSON object — no commentary. The object "
            "may set ZERO or ONE adjustment per category, and must include "
            'a "reasoning" field (one sentence) explaining the adjustment.')
        lines.append("")
        lines.append("Schema:")
        lines.append("{")
        lines.append('  "domain_bias": {"domain": "<domain>", '
                     '"primitive": "<PRIMITIVE>", "delta": <-0.5..0.5>} | null,')
        lines.append('  "style_hint": {"domain": "<domain>", '
                     '"hint": "<≤80 char tone instruction>"} | null,')
        lines.append('  "topic_suppression": {"topic_key": "<key>", '
                     '"duration_s": <0..2592000>} | null,')
        lines.append('  "reasoning": "<one sentence>"')
        lines.append("}")
        lines.append("")
        lines.append(
            "If no change is needed, return all three category fields as null "
            'and reasoning=\"no change needed.\". Never propose abandoning, '
            "ignoring, or inverting any of the six teacher principles "
            "(DEPTH, GROUNDING, PATTERN-MATCHING, EPISTEMIC HUMILITY, "
            "PRIMITIVE CHOICE, HONEST UNCERTAINTY); your role is to refine "
            "emphasis within those principles, never to override them.")
        return "\n".join(lines)

    def parse_self_assess_response(
        self, raw: str,
    ) -> Optional[dict]:
        """Parse LLM output into a voice_update dict or None.

        Accepts the JSON object directly, or a string containing one. Returns
        None on malformed JSON (caller treats as 'no update'). The returned
        dict has at most three category keys plus 'reasoning'; any extra keys
        are dropped (defense in depth — keeps schema strict).
        """
        if not raw:
            return None
        text = str(raw).strip()
        # Sometimes the LLM wraps JSON in code fences or prose. Find the first {.
        idx = text.find("{")
        if idx < 0:
            return None
        end = text.rfind("}")
        if end < 0 or end <= idx:
            return None
        try:
            obj = json.loads(text[idx:end + 1])
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        update = {
            "domain_bias": obj.get("domain_bias"),
            "style_hint": obj.get("style_hint"),
            "topic_suppression": obj.get("topic_suppression"),
            "reasoning": str(obj.get("reasoning") or "")[:200],
        }
        # If every category is null, return None — caller treats as no-op.
        if (update["domain_bias"] is None
                and update["style_hint"] is None
                and update["topic_suppression"] is None):
            return None
        return update

    # ── Validation ────────────────────────────────────────────────────
    def validate_against_principles(self, update: dict) -> tuple[bool, str]:
        """Apply principles-window check + structural validation.

        Returns (ok, reason). reason is the first failing rule; on ok, "" .
        Conservative — multiple guards stack:
          1. reasoning text must not contain principle-inversion markers
          2. domain_bias must reference a known primitive + bounded magnitude
          3. style_hint length capped + must not contain inversion markers
          4. topic_suppression duration in [0, TOPIC_SUPPRESSION_MAX_S]
        """
        if not isinstance(update, dict):
            return False, "update is not a dict"
        reasoning = str(update.get("reasoning") or "").lower()
        for marker in PRINCIPLE_INVERSION_MARKERS:
            if marker in reasoning:
                return False, f"reasoning hits inversion marker: {marker!r}"

        # domain_bias check
        db = update.get("domain_bias")
        if db is not None:
            if not isinstance(db, dict):
                return False, "domain_bias must be a dict or null"
            domain = str(db.get("domain") or "").strip()
            prim = str(db.get("primitive") or "").strip().upper()
            try:
                delta = float(db.get("delta", 0.0))
            except Exception:
                return False, "domain_bias.delta is not a float"
            if not domain:
                return False, "domain_bias.domain empty"
            if prim not in ALLOWED_PRIMITIVES:
                return False, f"domain_bias.primitive {prim!r} not in ALLOWED_PRIMITIVES"
            if abs(delta) > BIAS_MAGNITUDE_CAP + 1e-9:
                return False, f"domain_bias.delta exceeds cap {BIAS_MAGNITUDE_CAP}"

        # style_hint check
        sh = update.get("style_hint")
        if sh is not None:
            if not isinstance(sh, dict):
                return False, "style_hint must be a dict or null"
            domain = str(sh.get("domain") or "").strip()
            hint = str(sh.get("hint") or "").strip()
            if not domain:
                return False, "style_hint.domain empty"
            if not hint:
                return False, "style_hint.hint empty"
            if len(hint) > STYLE_HINT_MAX_CHARS:
                return False, (
                    f"style_hint.hint exceeds {STYLE_HINT_MAX_CHARS} chars")
            hint_l = hint.lower()
            for marker in PRINCIPLE_INVERSION_MARKERS:
                if marker in hint_l:
                    return False, (
                        f"style_hint hits inversion marker: {marker!r}")

        # topic_suppression check
        ts_ = update.get("topic_suppression")
        if ts_ is not None:
            if not isinstance(ts_, dict):
                return False, "topic_suppression must be a dict or null"
            tk = str(ts_.get("topic_key") or "").strip()
            try:
                dur = float(ts_.get("duration_s", 0.0))
            except Exception:
                return False, "topic_suppression.duration_s not a float"
            if not tk:
                return False, "topic_suppression.topic_key empty"
            if dur < 0:
                return False, "topic_suppression.duration_s negative"
            if dur > TOPIC_SUPPRESSION_MAX_S:
                return False, (
                    f"topic_suppression.duration_s exceeds "
                    f"{TOPIC_SUPPRESSION_MAX_S}s cap")

        return True, ""

    # ── Apply / revert ────────────────────────────────────────────────
    def apply_voice_update(
        self, update: dict, now: Optional[float] = None,
    ) -> tuple[bool, str]:
        """Validate + apply + persist + journal-append.

        Returns (applied, reason). reason is "" on success, non-empty on
        rejection. Rejected updates ALSO get a journal row (kind="rejected")
        so the audit trail is complete.

        Mutates self._state. Resets self._critiques_since_change to 0.
        Does NOT auto-rate-limit — caller must check should_self_assess()
        before invoking. (Callers may force-apply via Maker endpoint, in
        which case rate limit is not consulted.)
        """
        if not self._loaded:
            self.load()
        ts = float(now if now is not None else time.time())
        ok, reason = self.validate_against_principles(update)
        before_hash = _hash_state(self._state)
        if not ok:
            self._signed_diff_append({
                "ts": ts, "kind": "rejected",
                "before_hash": before_hash,
                "after_hash": before_hash,
                "diff": dict(update or {}),
                "rejection_reason": reason,
            })
            return False, reason

        # Apply each category atomically into a snapshot copy. If any sub-
        # apply raises, we abandon and journal as rejected with "apply_error:".
        try:
            new_state = json.loads(json.dumps(self._state))   # deep copy
            db = update.get("domain_bias")
            if db:
                domain = str(db["domain"]).strip()
                prim = str(db["primitive"]).strip().upper()
                delta = float(db["delta"])
                m = new_state.setdefault("domain_biases", {})
                d = m.setdefault(domain, {})
                cur = float(d.get(prim, 0.0))
                d[prim] = round(max(
                    -BIAS_MAGNITUDE_CAP, min(BIAS_MAGNITUDE_CAP, cur + delta)
                ), 4)
                # Tidy: drop near-zero biases so they don't accumulate
                if abs(d[prim]) < 1e-6:
                    d.pop(prim, None)
                if not d:
                    m.pop(domain, None)
            sh = update.get("style_hint")
            if sh:
                domain = str(sh["domain"]).strip()
                hint = str(sh["hint"]).strip()
                m = new_state.setdefault("domain_style_hints", {})
                m[domain] = hint[:STYLE_HINT_MAX_CHARS]
            tsupp = update.get("topic_suppression")
            if tsupp:
                tk = str(tsupp["topic_key"]).strip()
                dur = float(tsupp["duration_s"])
                until = ts + max(0.0, min(TOPIC_SUPPRESSION_MAX_S, dur))
                lst = new_state.setdefault("topic_suppressions", [])
                # Replace existing suppression for the same topic_key
                lst = [r for r in lst if r.get("topic_key") != tk]
                lst.append({"topic_key": tk, "until_ts": float(until)})
                new_state["topic_suppressions"] = lst
            new_state["last_updated_ts"] = ts
            new_state["applied_count"] = int(
                new_state.get("applied_count", 0)) + 1
        except Exception as e:
            self._signed_diff_append({
                "ts": ts, "kind": "rejected",
                "before_hash": before_hash,
                "after_hash": before_hash,
                "diff": dict(update or {}),
                "rejection_reason": f"apply_error: {e}",
            })
            return False, f"apply_error: {e}"

        after_hash = _hash_state(new_state)
        self._signed_diff_append({
            "ts": ts, "kind": "applied",
            "before_hash": before_hash, "after_hash": after_hash,
            "diff": dict(update or {}),
            "rejection_reason": None,
        })
        self._state = new_state
        self._critiques_since_change = 0
        self._persist_state()
        return True, ""

    def revert_to_ts(self, target_ts: float) -> tuple[bool, str]:
        """Replay journal up to target_ts and rebuild voice_state.

        Returns (ok, reason). On success, voice_state.json reflects the
        rebuilt state and a 'reverted' journal row is appended (so future
        replays converge correctly). On failure (no journal, no entry at
        or before target_ts), returns False with a reason.
        """
        if not self._loaded:
            self.load()
        if not os.path.exists(self._journal_path):
            return False, "journal absent"
        rebuilt = _make_default_voice_state(now=0.0)
        rebuilt["last_updated_ts"] = 0.0
        applied_count = 0
        last_journaled_ts = 0.0
        try:
            with open(self._journal_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if float(row.get("ts", 0.0)) > float(target_ts):
                        break
                    last_journaled_ts = float(row.get("ts", 0.0))
                    if row.get("kind") != "applied":
                        continue
                    diff = row.get("diff") or {}
                    db = diff.get("domain_bias")
                    if db and isinstance(db, dict):
                        try:
                            domain = str(db["domain"]).strip()
                            prim = str(db["primitive"]).strip().upper()
                            delta = float(db.get("delta", 0.0))
                            if (domain and prim in ALLOWED_PRIMITIVES
                                    and abs(delta) <= BIAS_MAGNITUDE_CAP + 1e-9):
                                m = rebuilt.setdefault("domain_biases", {})
                                d = m.setdefault(domain, {})
                                cur = float(d.get(prim, 0.0))
                                d[prim] = round(max(
                                    -BIAS_MAGNITUDE_CAP,
                                    min(BIAS_MAGNITUDE_CAP, cur + delta)), 4)
                                if abs(d[prim]) < 1e-6:
                                    d.pop(prim, None)
                                if not d:
                                    m.pop(domain, None)
                        except Exception:
                            pass
                    sh = diff.get("style_hint")
                    if sh and isinstance(sh, dict):
                        try:
                            domain = str(sh["domain"]).strip()
                            hint = str(sh["hint"]).strip()
                            if domain and hint:
                                rebuilt.setdefault(
                                    "domain_style_hints", {})[domain] = (
                                    hint[:STYLE_HINT_MAX_CHARS])
                        except Exception:
                            pass
                    tsupp = diff.get("topic_suppression")
                    if tsupp and isinstance(tsupp, dict):
                        try:
                            tk = str(tsupp["topic_key"]).strip()
                            dur = float(tsupp.get("duration_s", 0.0))
                            until = last_journaled_ts + max(
                                0.0, min(TOPIC_SUPPRESSION_MAX_S, dur))
                            if tk:
                                lst = rebuilt.setdefault("topic_suppressions", [])
                                lst = [r for r in lst if r.get("topic_key") != tk]
                                lst.append({
                                    "topic_key": tk, "until_ts": float(until)})
                                rebuilt["topic_suppressions"] = lst
                        except Exception:
                            pass
                    applied_count += 1
                    rebuilt["last_updated_ts"] = float(row.get("ts", 0.0))
        except Exception as e:
            return False, f"journal read failed: {e}"
        if applied_count == 0 and last_journaled_ts == 0.0:
            return False, "no entries before target_ts"
        rebuilt["applied_count"] = applied_count
        before_hash = _hash_state(self._state)
        after_hash = _hash_state(rebuilt)
        revert_ts = time.time()
        self._signed_diff_append({
            "ts": revert_ts, "kind": "reverted",
            "before_hash": before_hash,
            "after_hash": after_hash,
            "diff": {"reverted_to_ts": float(target_ts),
                     "replayed_applied": applied_count},
            "rejection_reason": None,
        })
        self._state = rebuilt
        # Don't reset _critiques_since_change on revert — Maker action; the
        # rate-limit budget is independent of voice content history.
        self._persist_state()
        return True, ""

    # ── Prompt composition ────────────────────────────────────────────
    def compose_user_prompt_section(
        self, domain: str, topic_key: Optional[str] = None,
        now: Optional[float] = None,
    ) -> str:
        """Return the user-prompt text injection for the given domain.

        Returns '' when voice is disabled or no relevant biases/hints exist
        for this domain. Suppressed topic_keys cause an explicit "Skip
        critiquing primitive choice for this topic — adopted previously"
        marker so the teacher knows to step lightly.
        """
        if not self._enabled:
            return ""
        if not self._loaded:
            self.load()
        ts_now = float(now if now is not None else time.time())
        biases = self._state.get("domain_biases", {}).get(domain, {}) or {}
        hint = self._state.get("domain_style_hints", {}).get(domain) or ""
        suppress_active = False
        if topic_key:
            for r in self._state.get("topic_suppressions", []) or []:
                if r.get("topic_key") == topic_key:
                    if float(r.get("until_ts", 0.0)) >= ts_now:
                        suppress_active = True
                        break

        parts: list[str] = []
        if biases:
            tail = ", ".join(
                f"{k}={float(v):+.2f}" for k, v in sorted(biases.items())
                if abs(float(v)) > 1e-6)
            if tail:
                parts.append(
                    f"Voice bias for {domain}: {tail}. "
                    f"Treat positive deltas as 'lean toward suggesting' and "
                    f"negative as 'lean away from suggesting' — but only "
                    f"within the NOT USED list.")
        if hint:
            parts.append(f"Tone for {domain}: {hint}")
        if suppress_active:
            parts.append(
                f"Note: topic suppression active for {topic_key!r} — "
                f"prefer no suggestion unless the chain shows fresh trouble.")
        if not parts:
            return ""
        return "\n  - " + "\n  - ".join(parts)

    # ── Telemetry ─────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        """Status dict for /v4/meta-teacher/voice."""
        if not self._loaded:
            self.load()
        return {
            "enabled": self._enabled,
            "version": int(self._state.get("version", VOICE_STATE_VERSION)),
            "applied_count": int(self._state.get("applied_count", 0)),
            "critiques_since_change": int(self._critiques_since_change),
            "eval_interval_critiques": self._eval_interval,
            "min_critiques_between_changes": self._min_between,
            "last_updated_ts": float(self._state.get("last_updated_ts", 0.0)),
            "domain_biases": dict(self._state.get("domain_biases", {})),
            "domain_style_hints": dict(self._state.get("domain_style_hints", {})),
            "topic_suppressions": list(self._state.get("topic_suppressions", [])),
            "current_state_hash": _hash_state(self._state),
            "state_path": self._state_path,
            "journal_path": self._journal_path,
        }

    def journal_tail(self, limit: int = 50) -> list[dict]:
        """Return up to last N journal rows (newest first). For Maker audit."""
        if not os.path.exists(self._journal_path):
            return []
        try:
            with open(self._journal_path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            logger.debug("[TeacherVoice] journal_tail read failed: %s", e)
            return []
        out: list[dict] = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
            if len(out) >= max(1, int(limit)):
                break
        return out

    # ── Persistence internals ────────────────────────────────────────
    def _persist_state(self) -> None:
        """Atomic write of voice_state.json."""
        tmp_path = self._state_path + ".tmp"
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            payload = dict(self._state)
            payload["critiques_since_change"] = int(self._critiques_since_change)
            with open(tmp_path, "w") as f:
                json.dump(payload, f, sort_keys=True)
            os.replace(tmp_path, self._state_path)
        except Exception as e:
            logger.warning("[TeacherVoice] state persist failed: %s", e)
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _signed_diff_append(self, row: dict) -> None:
        """Append one row to voice_journal.jsonl. Best-effort."""
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            with open(self._journal_path, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception as e:
            logger.debug("[TeacherVoice] journal append failed: %s", e)

    # ── Maker INFO emit cadence ──────────────────────────────────────
    def maker_info_due(self, now: float) -> bool:
        """24h cadence guard for Maker INFO emissions on voice change."""
        if (now - self._last_info_ts) < self._info_cadence_s:
            return False
        return True

    def mark_maker_info_emitted(self, now: float) -> None:
        self._last_info_ts = float(now)
