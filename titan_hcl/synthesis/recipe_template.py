"""titan_hcl.synthesis.recipe_template — §7.E (E1.1/E1.3) safe templated recipe replay.

The substrate for E.1 compose-reuse (Maker-locked Option B, 2026-06-13): a verified
tool-call is captured as a PARAM TEMPLATE so a *similar* prompt with DIFFERENT params
reuses the recipe with zero LLM derivation (genuinely multiples-useful + future-
extensible), and the oracle re-verify gates every emit.

Precision: the template is built at CAPTURE time from the param values ALREADY KNOWN
for that turn (the numeric tokens in the prompt that produced the call) — never by
guessing later which literal is a param. A value becomes a typed slot ONLY when it
appears exactly once as a standalone token (unambiguous); otherwise it stays literal.

Safety (this module is the injection firewall — the oracle re-verify is the final
net, but this guarantees nothing un-sanitized reaches executable code):
  • slots are TYPED (`number`); at bind time a new value MUST parse as a clean
    numeric literal or the whole bind is rejected → no raw user text is ever
    substituted into code (no code injection).
  • arity must match exactly (same slot count) or the bind is rejected.
  • any unfilled slot / ambiguity / non-numeric ⇒ return None ⇒ the caller falls
    through to the LLM-router path (never a wrong-but-runnable substitution).

Pure functions, no I/O, no torch — exhaustively offline-testable.
"""
from __future__ import annotations

import re
from typing import Optional

# A numeric literal/param: optional sign, digits, optional single decimal part.
# Deliberately NOT scientific/hex — a param we substitute into code must be a
# plain, obviously-safe number. Anything richer ⇒ not a templatizable param.
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# A value matched as a STANDALONE token in code: not flanked by word chars or a
# dot (so "8" in "math.factorial(8)" matches, but the "8" inside "v8" or "1.85"
# does not → no accidental partial-token substitution).
def _standalone(value: str) -> re.Pattern:
    return re.compile(r"(?<![\w.])" + re.escape(value) + r"(?![\w.])")


def extract_numeric_params(text: str) -> list[str]:
    """Ordered numeric tokens in a prompt — the compute params for E.1.

    "order my 8 climbing routes" → ["8"]; "17 times 23" → ["17", "23"]. Order is
    left-to-right (positional binding). Empty/None → []."""
    return _NUM_RE.findall(text or "")


def templatize(code: str, param_values: list[str]) -> Optional[dict]:
    """Build `{code_template, param_kinds, captured_params}` from the executed
    `code` + the numeric `param_values` known for THIS turn.

    A value → a `{pN}` slot ONLY when it appears EXACTLY ONCE as a standalone
    numeric token in the code (unambiguous). Values absent from the code, or
    appearing multiple times, are left literal (safe — that recipe then replays
    only for identical params, or falls through). Returns None when no param is
    templatizable (→ caller stores the literal recipe / uses the LLM fallback).

    Precise by construction: substitutes the KNOWN captured values, never guesses.
    """
    if not code or not param_values:
        return None
    template = code
    kinds: list[str] = []
    captured: list[str] = []
    slot = 0
    for val in param_values:
        if not _NUM_RE.fullmatch(val.strip()):
            continue
        pat = _standalone(val.strip())
        if len(pat.findall(template)) != 1:
            continue  # absent or ambiguous → leave literal (safe)
        template = pat.sub("{p%d}" % slot, template, count=1)
        kinds.append("number")
        captured.append(val.strip())
        slot += 1
    if slot == 0:
        return None
    return {"code_template": template, "param_kinds": kinds,
            "captured_params": captured}


def bind(template: Optional[dict], new_param_values: list[str]) -> Optional[str]:
    """Fill `template` with NEW numeric params → runnable code, or None.

    Returns None (→ caller falls through to the LLM router) on: missing/empty
    template, arity mismatch, any non-numeric new value (injection guard), or any
    unfilled slot remaining. Substitutes ONLY sanitized numeric literals — no raw
    text ever reaches the code string.
    """
    if not template:
        return None
    code_template = template.get("code_template", "")
    kinds = list(template.get("param_kinds", []))
    if not code_template or not kinds:
        return None
    if len(new_param_values) != len(kinds):
        return None  # arity must match exactly → else fall through
    code = code_template
    for i, (val, kind) in enumerate(zip(new_param_values, kinds)):
        if kind != "number":
            return None  # only numeric slots are supported (safe)
        v = str(val).strip()
        if not _NUM_RE.fullmatch(v):
            return None  # STRICT injection guard — must be a clean number
        code = code.replace("{p%d}" % i, v)
    if "{p" in code:
        return None  # any slot left unfilled → reject (never run a half-bound recipe)
    return code


def params_identical(captured_params: list[str], new_param_values: list[str]) -> bool:
    """True when the new prompt's numeric params equal the captured ones (a re-ask
    of the same computation). A templated recipe with identical params is a verbatim
    replay; this lets the caller skip even the bind step when nothing changed."""
    return [str(p).strip() for p in (captured_params or [])] == \
           [str(p).strip() for p in (new_param_values or [])]
