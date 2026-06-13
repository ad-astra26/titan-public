"""§7.E (E1.1/E1.3) — safe templated recipe replay (the E.1 compose-reuse heart).

Exhaustively covers the precision (capture-time templatization from KNOWN params)
and — critically — the SAFETY (typed numeric slots, strict sanitization → no code
injection, arity match, fall-through on any ambiguity). The oracle re-verify is the
final net live; this proves nothing un-sanitized reaches executable code.

Run: python -m pytest tests/test_recipe_template.py -v -p no:anchorpy
"""
from __future__ import annotations

from titan_hcl.synthesis.recipe_template import (
    bind, extract_numeric_params, params_identical, templatize,
)


# ── extract_numeric_params ──────────────────────────────────────────────────
def test_extract_basic():
    assert extract_numeric_params("how many ways to order my 8 climbing routes") == ["8"]
    assert extract_numeric_params("what is 17 times 23") == ["17", "23"]
    assert extract_numeric_params("2.5 plus 3.5") == ["2.5", "3.5"]
    assert extract_numeric_params("no numbers here") == []
    assert extract_numeric_params("") == []
    assert extract_numeric_params(None) == []


# ── templatize (precision) ──────────────────────────────────────────────────
def test_templatize_single_param():
    t = templatize("math.factorial(8)", ["8"])
    assert t is not None
    assert t["code_template"] == "math.factorial({p0})"
    assert t["param_kinds"] == ["number"]
    assert t["captured_params"] == ["8"]


def test_templatize_two_params_positional():
    t = templatize("17*23", ["17", "23"])
    assert t["code_template"] == "{p0}*{p1}"
    assert t["param_kinds"] == ["number", "number"]


def test_templatize_value_absent_from_code_left_literal():
    # the prompt had "8" but the code uses a different literal → not templatizable
    t = templatize("math.factorial(5)", ["8"])
    assert t is None


def test_templatize_ambiguous_value_left_literal():
    # "5" appears twice → ambiguous → that param stays literal; nothing templatized
    t = templatize("pow(5, 5)", ["5"])
    assert t is None


def test_templatize_mixed_one_ok_one_ambiguous():
    # 8 is unambiguous (→ slot), 2 appears twice (→ left literal)
    t = templatize("comb(8, 2) + 2", ["8", "2"])
    assert t is not None
    assert t["code_template"] == "comb({p0}, 2) + 2"
    assert t["param_kinds"] == ["number"]
    assert t["captured_params"] == ["8"]


def test_templatize_no_partial_token_substitution():
    # "8" must NOT match inside "v8" or "1.85"
    t = templatize("v8 = 1.85", ["8"])
    assert t is None  # no standalone 8 → nothing templatized


def test_templatize_empty_inputs():
    assert templatize("", ["8"]) is None
    assert templatize("math.factorial(8)", []) is None


# ── bind (precision + SAFETY) ───────────────────────────────────────────────
def test_bind_substitutes_new_numeric_param():
    t = templatize("math.factorial(8)", ["8"])
    assert bind(t, ["10"]) == "math.factorial(10)"


def test_bind_two_params():
    t = templatize("17*23", ["17", "23"])
    assert bind(t, ["12", "9"]) == "12*9"


def test_bind_decimal_param():
    t = templatize("area(2.5)", ["2.5"])
    assert bind(t, ["7.25"]) == "area(7.25)"


def test_bind_arity_mismatch_rejected():
    t = templatize("17*23", ["17", "23"])
    assert bind(t, ["12"]) is None          # too few
    assert bind(t, ["12", "9", "4"]) is None  # too many


def test_bind_rejects_non_numeric_INJECTION_GUARD():
    # the core safety property — raw text is NEVER substituted into code.
    t = templatize("math.factorial(8)", ["8"])
    assert bind(t, ["__import__('os').system('rm -rf /')"]) is None
    assert bind(t, ["8); os.system('x'); ("]) is None
    assert bind(t, ["1e9"]) is None          # scientific notation not a clean literal
    assert bind(t, ["0x10"]) is None         # hex not a clean literal
    assert bind(t, ["eight"]) is None        # word not numeric
    assert bind(t, [""]) is None


def test_bind_negative_param():
    t = templatize("abs(-5)", ["-5"])
    assert t is not None
    assert bind(t, ["-12"]) == "abs(-12)"


def test_bind_none_and_empty_template():
    assert bind(None, ["10"]) is None
    assert bind({}, ["10"]) is None
    assert bind({"code_template": "", "param_kinds": []}, []) is None


def test_bind_unknown_kind_rejected():
    # a non-number slot kind is never substituted (defensive — only number ships)
    tmpl = {"code_template": "f({p0})", "param_kinds": ["string"], "captured_params": ["x"]}
    assert bind(tmpl, ["y"]) is None


# ── params_identical (verbatim-replay shortcut) ─────────────────────────────
def test_params_identical():
    assert params_identical(["8"], ["8"]) is True
    assert params_identical(["8"], ["10"]) is False
    assert params_identical(["17", "23"], ["17", "23"]) is True
    assert params_identical([], []) is True


# ── end-to-end round-trip ───────────────────────────────────────────────────
def test_capture_then_replay_new_param_roundtrip():
    # capture from the §1.3 worked example, replay for a different count
    prompt_old = "how many ways can I order my 8 climbing routes?"
    code_old = "math.factorial(8)"
    t = templatize(code_old, extract_numeric_params(prompt_old))
    prompt_new = "how many orderings of 10 routes?"
    code_new = bind(t, extract_numeric_params(prompt_new))
    assert code_new == "math.factorial(10)"
