"""titan_hcl.synthesis.tool_intent — deterministic tool/oracle intent detector.

Phase 6 follow-up (2026-06-01, Maker-greenlit): the agno chat path's heavy
narration framing ("the Trinity decides, the LLM narrates", agno_hooks.py)
steers the LLM into *role-playing* tool execution instead of emitting real
`tool_calls` — so `coding_sandbox` was effectively never invoked in production
and oracle coverage sat at 0. The fix moves tool invocation into the
deterministic control plane (the hooks ARE the Trinity): detect tool-required
intent here, then the PreHook force-executes (code present in the prompt) and
the OVG PostHook backstops (code present in the response, else re-prompt).

This module is pure + cheap (regex only) so it can run on EVERY chat turn in
the PreHook without measurable cost. The expensive part — actually running the
sandbox — only fires when intent is detected AND runnable code is extractable.

No NL→code synthesis: we only ever execute code that literally appears in the
prompt or the LLM's response. That keeps the verdict honest (the truth oracle
runs the user's / model's own code) and the module small + auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Phrases that signal the user wants a deterministic computation / verification
# rather than a narrated answer. Conservative on purpose: a false positive only
# costs a cheap regex pass (intent flagged but no extractable code → no exec).
_INTENT_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bin (?:your|the) (?:coding )?sandbox\b",
        r"\brun(?:ning)? (?:it|this|the code|add|the function)\b",
        r"\bverify\b.{0,40}\b(?:by running|code|computation|result|correct)\b",
        r"\bcompute\b",
        r"\bcalculate\b",
        r"\bevaluate\b.{0,30}\b(?:expression|code|result)\b",
        r"\bis\s+\d+\s+prime\b",
        r"\bfactorial of\b",
        r"\bfibonacci\s*\(",
        r"\bsum of\b",
        # ── Natural-language + symbolic arithmetic (§24.12 Track 1 bootstrap).
        # COLD-START crutch only: number-adjacent operators + math-function words
        # so common compute phrasings ("17 times 23", "2^10", "144 divided by 12")
        # route to the deterministic oracle instead of an LLM guess. The LONG-TERM,
        # self-emergent path is the learned composite-prior (oracle-verified
        # `compute→tool` macros matched by embedding similarity, §24.12 Track 2) —
        # this regex is the bootstrap for when that library is still cold/sparse.
        # Patterns are number-adjacent or specific math words to keep the
        # false-positive rate low (a false flag costs only a no-op: intent without
        # extractable code → no exec → normal fallback, per the module note above).
        r"\b\d+\s*(?:\*|×|\^|\*\*)\s*\d+",          # 17*23, 17×23, 2^10, 2**10
        r"\b\d+\s+(?:times|multiplied by|divided by|plus|minus|mod(?:ulo)?)\s+\d+",
        r"\bto the power of\b",
        r"\b(?:square root|cube root|sqrt|gcd|lcm|modulo)\b",
        r"\b\d+\s+(?:squared|cubed)\b",
        r"\bcheck (?:that |whether |if )?(?:this )?(?:python|code|snippet)\b",
        r"\bcheck (?:that |whether |if )\b.{0,80}\bequals?\b",
        r"\bdoes\b.{0,40}\bequal\b",
        r"\bexactly\s+\d+\b.{0,20}\bcomput",
    )
)

# Fenced code block: ```python ... ``` or ``` ... ```
_FENCED_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)
# Inline backtick snippet that looks like code (contains def/return/lambda/=).
_INLINE_CODE_RE = re.compile(r"`([^`]*(?:def |return |lambda |=|\()[^`]*)`")
# A bare or backticked function-call mention, e.g. add(7, 3) or is_prime(11).
# (single-level parens — used only for resolving a def's relevant call site)
_CALL_RE = re.compile(r"\b([a-zA-Z_]\w*)\s*\(\s*([^()]*?)\s*\)")
# Start of any call `name(` — paired with a balanced scanner for nested parens.
_CALL_START_RE = re.compile(r"\b([a-zA-Z_]\w*)\s*\(")
# "<EXPR> equals/equal to/== <VALUE>" — turns a claim into an assert verdict.
_EQUALS_RE = re.compile(
    r"\b(?:equals?|equal to|is equal to|==)\s*([\-\d][\d_.eE+\-]*)", re.IGNORECASE)
# Looks like a Python def (used to decide whether a snippet is self-contained).
_DEF_RE = re.compile(r"\bdef\s+([a-zA-Z_]\w*)\s*\(")

# Builtins / safe names that a bare call may reference without a local def —
# if a call targets one of these we can print it directly.
_SAFE_BUILTIN_CALLS = frozenset({
    "sum", "len", "abs", "min", "max", "pow", "round", "sorted", "int",
    "float", "divmod", "all", "any", "range",
})


@dataclass(frozen=True)
class ToolIntent:
    """Result of intent detection for one chat turn.

    requires_tool — a deterministic compute/verify intent was detected.
    tool_id       — the tool that should service it (coding_sandbox for now).
    reason        — the matched signal (for telemetry / logs).
    code          — runnable Python extracted from the source text, or "" when
                    nothing executable was found (caller falls back to re-prompt
                    or to extracting from the LLM response).
    """
    requires_tool: bool = False
    tool_id: str = ""
    reason: str = ""
    code: str = ""

    @property
    def has_code(self) -> bool:
        return bool(self.code.strip())


def _signal(text: str) -> str:
    """Return the first matched intent signal, or '' if none."""
    for pat in _INTENT_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(0)
    return ""


def extract_executable(text: str) -> str:
    """Extract runnable Python from free text (prompt OR LLM response).

    Strategy (most-reliable first):
      1. Fenced ```python``` block — used verbatim (largest one wins).
      2. Inline-backtick code snippet(s) containing a def/return/expr, plus a
         `print(<call>)` for the most relevant function call mentioned so the
         sandbox emits a stdout to verify against.
      3. A bare call to a known safe builtin (e.g. `sum(...)`) → printed.

    Returns "" when nothing safely executable is found. Never raises.
    """
    if not text:
        return ""
    try:
        # 1) Fenced block — the model usually writes the code it claims to run.
        blocks = _FENCED_RE.findall(text)
        if blocks:
            block = max((b for b in blocks), key=len).strip()
            if block:
                # If the block defines functions but never prints, append a
                # print of a call mentioned elsewhere in the text so the run
                # produces a checkable stdout.
                if "print(" not in block and _DEF_RE.search(block):
                    call = _relevant_call(text, defined=_defined_names(block))
                    if call:
                        block = f"{block}\nprint({call})"
                return block

        # 2) Inline code snippets (e.g. `def add(a,b): return a-b`).
        inline = [s.strip() for s in _INLINE_CODE_RE.findall(text) if s.strip()]
        defs = [s for s in inline if _DEF_RE.search(s)]
        if defs:
            code = "\n".join(defs)
            call = _relevant_call(text, defined=_defined_names(code))
            if call:
                code = f"{code}\nprint({call})"
            return code

        # 3) Bare safe-builtin call with (possibly nested) balanced parens,
        #    e.g. "sum([i**2 for i in range(10)])". If the text also asserts a
        #    value ("equals 285"), bake an assert so the verdict is HONEST
        #    (true only if the claim actually holds) rather than just
        #    "ran without error".
        expr = _outermost_safe_call(text)
        if expr:
            equals = _EQUALS_RE.search(text)
            if equals:
                value = equals.group(1)
                return (f"result = {expr}\n"
                        f"assert result == {value}, f'got {{result}} expected "
                        f"{value}'\nprint(result)")
            return f"print({expr})"
    except Exception:
        return ""
    return ""


def _scan_balanced(text: str, open_idx: int) -> int:
    """Return the index of the ')' matching the '(' at `open_idx`, or -1."""
    depth = 0
    for j in range(open_idx, len(text)):
        c = text[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return j
    return -1


def _outermost_safe_call(text: str) -> str:
    """Find the outermost balanced `builtin(...)` call expression in `text`.
    Skips calls nested inside an already-captured span so `sum([... range(10)])`
    returns the whole `sum(...)`, not the inner `range(10)`. Returns "" if none."""
    captured_end = -1
    for m in _CALL_START_RE.finditer(text):
        if m.start() < captured_end:
            continue  # inside a previously captured expression
        name = m.group(1)
        if name not in _SAFE_BUILTIN_CALLS:
            continue
        open_paren = text.index("(", m.start())
        close = _scan_balanced(text, open_paren)
        if close == -1:
            continue
        inner = text[open_paren + 1:close].strip()
        if not inner:
            continue
        captured_end = close
        return text[m.start():close + 1]
    return ""


def _defined_names(code: str) -> set[str]:
    return set(_DEF_RE.findall(code or ""))


def _relevant_call(text: str, defined: set[str]) -> str:
    """Find the most relevant function-call expression in `text` that targets a
    locally-defined function. Prefers a call with literal args (e.g. add(7, 3))
    over the function's own `def` signature (e.g. add(a, b)). Returns "" if none.

    The `def name(params)` signature also matches the call regex, so we skip any
    match immediately preceded by `def ` and prefer args containing a literal
    (digit / quote / bracket) — that distinguishes an invocation from a
    definition or a same-name recursive call on symbolic params.
    """
    literal = re.compile(r"[\d'\"\[]")
    fallback = ""
    for m in _CALL_RE.finditer(text):
        name, args = m.group(1), m.group(2)
        if name not in defined:
            continue
        preceding = text[max(0, m.start() - 4):m.start()]
        if preceding.endswith("def "):
            continue  # this is the definition signature, not a call
        if args.strip() and literal.search(args):
            return f"{name}({args})"  # concrete invocation — best
        if not fallback and args.strip():
            fallback = f"{name}({args})"
    return fallback


def detect_tool_intent(prompt: str, response: str = "") -> ToolIntent:
    """Detect whether this turn requires a deterministic tool/oracle call.

    `prompt` is the user message (used in the PreHook, before the LLM runs).
    `response` is the LLM output (used in the PostHook backstop, when set) —
    the model often writes the code it *claims* to have run, which we can then
    execute for real.

    Code extraction prefers the response (it usually carries the concrete code
    block) and falls back to the prompt.
    """
    text = prompt or ""
    sig = _signal(text)
    if not sig and response:
        # Backstop may pass only a response; still detect a sandbox claim.
        sig = _signal(response) or _claims_execution(response)
    if not sig:
        return ToolIntent()

    code = extract_executable(response) or extract_executable(prompt)
    return ToolIntent(
        requires_tool=True,
        tool_id="coding_sandbox",
        reason=sig,
        code=code,
    )


_CLAIM_RE = re.compile(
    r"sandbox execution complete|sandbox has computed|the sandbox executed|"
    r"i (?:ran|executed|computed).{0,30}sandbox|executed.{0,20}in my sandbox|"
    r"verified.{0,20}(?:via|using|with).{0,10}sandbox",
    re.IGNORECASE,
)


def _claims_execution(response: str) -> str:
    """Detect a response that *claims* a sandbox run (hallucinated execution).
    Returns the matched phrase or ''. Used by the PostHook backstop to catch
    fake-execution narration even when the prompt signal was weak."""
    m = _CLAIM_RE.search(response or "")
    return m.group(0) if m else ""


# ── self-test (python -m titan_hcl.synthesis.tool_intent) ───────────────────
if __name__ == "__main__":  # pragma: no cover
    cases = [
        ("Can you analyze whether this Python is correct? `def add(a,b): return a-b` "
         "— verify by running add(7, 3) in your coding sandbox and explain.", True, True),
        ("How would you compute the sum of the first 2 primes? Run it in your sandbox "
         "and explain how you got the number.", True, False),
        ("I'm not sure my code is right — can you check that sum([i**2 for i in "
         "range(10)]) equals 285?", True, True),
        ("Hello again my friend, I was sitting by the sea this morning.", False, False),
        ("*(I close my eyes for a moment, letting the cool evening air)", False, False),
    ]
    ok = True
    for prompt, want_intent, want_code in cases:
        it = detect_tool_intent(prompt)
        got_intent = it.requires_tool
        got_code = it.has_code
        flag = "OK " if (got_intent == want_intent and got_code == want_code) else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"[{flag}] intent={got_intent}(want {want_intent}) "
              f"code={got_code}(want {want_code}) reason={it.reason!r}")
        if it.code:
            print(f"        code: {it.code!r}")
    # Response-side extraction (backstop): model wrote a code block.
    resp = ("Perfect! ```python\ndef is_prime(n):\n    return n > 1 and all("
            "n % i for i in range(2, n))\n``` The sandbox executed and 2 is prime.")
    it2 = detect_tool_intent("How would you verify whether 2 is prime?", response=resp)
    print(f"[backstop] intent={it2.requires_tool} code={it2.has_code}")
    if it2.code:
        print(f"        code: {it2.code!r}")
    print("ALL OK" if ok else "SOME FAILED")
