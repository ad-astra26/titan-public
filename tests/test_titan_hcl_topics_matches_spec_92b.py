"""Lockstep test — `TITAN_HCL_BROADCAST_TOPICS` MUST match SPEC §9.B exactly.

D-SPEC-42 (SPEC v1.4.0, 2026-05-12) closed the architectural regression caused
by rFP_worker_broadcast_topics_completion §4.C retirement: pre-fix, 6 separate
BusSocketClient connections subscribed-all and the
`_HIGH_RATE_BROADCAST_TYPES` stopgap masked the SPEC violation. Post-fix,
titan_HCL opens ONE BusSocketClient connection per SPEC §9.B
(`titan_HCL ... Bus subscriptions:`), subscribing only to the SPEC-enumerated
broadcast topic list. The other 5 kernel-internal subscriber names + 11 proxy
reply queues live as ADDITIVE ALIASES on titan_HCL's single connection per
SPEC §8.2 v1.3.0 multi-name semantics.

This test asserts: `titan_hcl.core.kernel.TITAN_HCL_BROADCAST_TOPICS` is
the SAME SET as the bus topics declared in SPEC §9.B titan_HCL block. Any
drift between SPEC and code is a CI failure — caller must update BOTH the
SPEC §9.B block AND the constant in the same commit.

Per Rule 0 (`feedback_specs_100pct_precedence_first_rule.md`, codified
2026-05-12): SPEC has 100% precedence; if SPEC and code disagree, SPEC wins.
This test surfaces such disagreement at commit time.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from titan_hcl.core.kernel import TITAN_HCL_BROADCAST_TOPICS

SPEC_PATH = Path(__file__).resolve().parent.parent / "titan-docs" / "specs" / "SPEC_titan_architecture.md"


def _parse_titan_hcl_subscriptions_from_spec() -> set[str]:
    """Parse SPEC §9.B titan_HCL block and extract the REQUIRED + OPTIONAL
    bus subscription topic names.

    Block format (verified against SPEC v1.4.0 lines 1149-1173):

        #### titan_HCL (Python parent process)

        ```
        titan_HCL  (entry: ...)
        ├── Parent supervisor: titan-kernel-rs
        ├── Children supervised: ...
        ├── Bus subscriptions:
        │   ├── REQUIRED: KERNEL_EPOCH_TICK              # comment
        │   ├── REQUIRED: KERNEL_SHUTDOWN_ANNOUNCE       # comment
        │   ├── REQUIRED: BODY_STATE / MIND_STATE / SPIRIT_STATE  # comment
        │   ├── REQUIRED: SWAP_HANDOFF / SWAP_HANDOFF_CANCELED   # comment
        │   ├── REQUIRED: ADOPTION_ACK
        │   └── OPTIONAL: SPHERE_PULSE / SPHERE_EPOCH_TICK / ...
        ├── Bus publications: ...
        ```

    Topics may appear singly or grouped with `/` separators. We extract every
    UPPERCASE_WITH_UNDERSCORES identifier on REQUIRED / OPTIONAL lines.
    """
    if not SPEC_PATH.exists():
        pytest.skip(f"SPEC not found at {SPEC_PATH}")

    text = SPEC_PATH.read_text(encoding="utf-8")
    # Locate the titan_HCL block.
    start = text.find("#### titan_HCL (Python parent process)")
    if start < 0:
        pytest.fail(
            "SPEC §9.B titan_HCL block not found. The lockstep test relies on "
            "the section heading '#### titan_HCL (Python parent process)'. "
            "If the heading was renamed, update this test alongside SPEC."
        )
    # The block ends at the next `####` heading or end of file.
    next_block = text.find("\n#### ", start + 1)
    block = text[start : next_block if next_block > 0 else len(text)]

    # Find the Bus subscriptions sub-section — everything between
    # "├── Bus subscriptions:" and the next top-level "├── " or "└── " line
    # that is NOT inside the nested `│   ├── ` indent.
    subs_match = re.search(
        r"├── Bus subscriptions:\n(?P<body>(?:│\s+(?:├──|└──).*\n)+)",
        block,
    )
    if subs_match is None:
        pytest.fail(
            "Couldn't locate 'Bus subscriptions:' sub-block within SPEC §9.B "
            "titan_HCL block. Update this test if the block format changed."
        )
    body = subs_match.group("body")

    # Strip comments (after `#`), extract the colon-prefixed payload, split on `/`.
    topics: set[str] = set()
    line_re = re.compile(r"(?:REQUIRED|OPTIONAL):\s*(?P<payload>[^#\n]+)")
    for m in line_re.finditer(body):
        payload = m.group("payload").strip()
        for part in payload.split("/"):
            ident = part.strip()
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", ident):
                topics.add(ident)
    if not topics:
        pytest.fail(
            "No topics parsed from SPEC §9.B titan_HCL Bus subscriptions block."
        )
    return topics


def test_titan_hcl_broadcast_topics_matches_spec_92b():
    """`TITAN_HCL_BROADCAST_TOPICS` MUST equal the SPEC §9.B titan_HCL
    block's REQUIRED ∪ OPTIONAL bus subscriptions.

    Adding a new topic to SPEC §9.B → must also add to the constant.
    Removing a topic from SPEC §9.B → must also remove from the constant.
    Any drift → this test fails and prevents merge.
    """
    spec_topics = _parse_titan_hcl_subscriptions_from_spec()
    code_topics = set(TITAN_HCL_BROADCAST_TOPICS)

    missing_in_code = spec_topics - code_topics
    extra_in_code = code_topics - spec_topics

    assert not missing_in_code and not extra_in_code, (
        f"SPEC §9.B ↔ TITAN_HCL_BROADCAST_TOPICS drift detected.\n"
        f"  SPEC §9.B requires: {sorted(spec_topics)}\n"
        f"  Code declares:      {sorted(code_topics)}\n"
        f"  Missing from code:  {sorted(missing_in_code) or '(none)'}\n"
        f"  Extra in code:      {sorted(extra_in_code) or '(none)'}\n"
        f"\n"
        f"Per Rule 0 (SPEC has 100% precedence): "
        f"if SPEC §9.B is wrong, fix SPEC first (greenlight required) and "
        f"bump spec_version per §2.6. If code is wrong, fix "
        f"`titan_hcl.core.kernel.TITAN_HCL_BROADCAST_TOPICS` to match SPEC. "
        f"Both must update in the same commit."
    )


def test_titan_hcl_broadcast_topics_is_a_tuple_of_strings():
    """Sanity — guard against future refactor that breaks the type contract.
    The constant is read at kernel boot to pass into `BusSocketClient(topics=...)`
    which expects an iterable of strings.
    """
    assert isinstance(TITAN_HCL_BROADCAST_TOPICS, tuple), (
        "TITAN_HCL_BROADCAST_TOPICS must be a tuple (immutable + hashable)"
    )
    for topic in TITAN_HCL_BROADCAST_TOPICS:
        assert isinstance(topic, str), (
            f"TITAN_HCL_BROADCAST_TOPICS entry {topic!r} must be a string"
        )
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", topic), (
            f"TITAN_HCL_BROADCAST_TOPICS entry {topic!r} must be SCREAMING_SNAKE_CASE"
        )
