#!/usr/bin/env python3
"""register_unregistered_bus_literals — mechanical registration of bus literals.

Scope: takes the `bus_literal_msg_type` findings from the patched
arch_map_dead_wiring scanner and produces a single edit to `titan_plugin/bus.py`
that registers every unregistered literal as a constant.

What this script DOES:
  - Read scanner findings (from arch_map_dead_wiring.py --all --json)
  - Group unregistered literals by domain prefix (TIMECHAIN_*, CGN_*, etc.)
  - Append a new section to bus.py with all constants registered

What this script DOES NOT do (manual follow-up):
  - Replace string literals at the producer/consumer call sites with the
    constants. That requires per-file judgment (each module has its own
    bus-import style: `from titan_plugin.bus import X`, aliased imports,
    relative imports). Surgical Edit's are safer than auto-rewrite.
  - Add spec entries to bus_specs.py with non-default priority/coalesce/ttl.
    Most can stay at the safe DEFAULT_SPEC (P2, no coalesce). Specs only
    need to be added for messages that genuinely need P0 (kernel-critical)
    or coalesce semantics — that's a judgment call per-message.

Usage:
    python scripts/register_unregistered_bus_literals.py --dry-run
    python scripts/register_unregistered_bus_literals.py --apply
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BUS_PY = REPO_ROOT / "titan_plugin" / "bus.py"
SCANNER = REPO_ROOT / "scripts" / "arch_map_dead_wiring.py"

SECTION_HEADER = (
    "# ── Unregistered literals — registered 2026-04-28 audit ─────────────\n"
    "# These message types were used as string literals across the codebase\n"
    "# without being registered as constants here. The arch_map dead-wiring\n"
    "# scanner v2.4 (commit 4493b41e) caught 77 such literals on the\n"
    "# 2026-04-28 Phase C contract audit; this section closes the drift.\n"
    "#\n"
    "# The scanner now emits HIGH-severity `bus_literal_msg_type` findings\n"
    "# for any literal looking like a bus msg type (UPPER_SNAKE_CASE) that\n"
    "# isn't in this file. Future literal drift fails CI.\n"
    "#\n"
    "# Spec entries in titan_plugin/bus_specs.py are optional: messages here\n"
    "# default to P2 + no-coalesce (the safe default per Phase B.2 §D6).\n"
    "# Add a spec entry only if a message genuinely needs P0/P1/P3 or\n"
    "# coalesce semantics — judgment per-message.\n"
)


# Domain grouping — first segment of the constant name maps to a sub-section
# label. Keeps bus.py readable instead of one flat block of 77 names.
DOMAIN_LABELS: dict[str, str] = {
    "BACKUP":     "Backup / save lifecycle",
    "BUS":        "Bus protocol — supervision transfer",
    "CGN":        "CGN protocol",
    "CONTRACT":   "TimeChain contracts",
    "EVENTS":     "Events teacher",
    "EXPERIENCE": "Experience playground",
    "GREAT":      "GREAT pulse / kin",
    "KNOWLEDGE":  "Knowledge graph queries",
    "LLM":        "LLM teacher request/response",
    "MAKER":      "Maker dialogue + narration + proposals",
    "MEDITATION": "Meditation lifecycle",
    "MEMORY":     "Memory ops",
    "META":       "Meta-reasoning rewards + signals",
    "OUTER":      "Outer trinity ready",
    "OUTPUT":     "Output verifier",
    "QUERY":      "Query / response infra",
    "REFLEX":     "Reflex worker ready",
    "SAVE":       "Save lifecycle",
    "SEARCH":     "Search pipeline",
    "SELF":       "Self-exploration / self-prediction",
    "SILENT":     "Silent-swallow report",
    "SOCIAL":     "Social perception",
    "SYSTEM":     "System upgrade thoughts",
    "TIMECHAIN":  "TimeChain operations",
    "WARNING":    "Warning monitor pulses",
    "X":          "X / social_x dispatch",
}


def run_scanner() -> list[dict]:
    """Run arch_map dead_wiring with --all and return the findings list."""
    proc = subprocess.run(
        [sys.executable, str(SCANNER), "--all", "--json"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180,
    )
    # Scanner prints a non-JSON banner before the JSON document — find first {
    out = proc.stdout
    start = out.find("{")
    if start == -1:
        raise RuntimeError(
            f"scanner produced no JSON; stdout={out[:300]!r} stderr={proc.stderr[:300]!r}")
    data = json.loads(out[start:])
    return data.get("findings", [])


def collect_unregistered(findings: list[dict]) -> dict[str, list[str]]:
    """Group unregistered-literal findings by domain prefix.

    Returns: {domain_label: [constant_name, ...]} with names sorted alpha.
    """
    by_domain: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for f in findings:
        if f.get("kind") != "bus_literal_msg_type":
            continue
        # Title format: "msg 'NAME' — string literal..."
        title = f.get("title", "")
        if "'" not in title:
            continue
        name = title.split("'")[1]
        if name in seen:
            continue
        seen.add(name)
        domain = name.split("_", 1)[0]
        label = DOMAIN_LABELS.get(domain, f"{domain} (uncategorized)")
        by_domain[label].append(name)
    for k in by_domain:
        by_domain[k].sort()
    return dict(sorted(by_domain.items()))


def render_section(by_domain: dict[str, list[str]]) -> str:
    """Render the new bus.py section as a single string."""
    out_lines: list[str] = []
    out_lines.append("\n")
    out_lines.append(SECTION_HEADER)
    out_lines.append("\n")
    for label, names in by_domain.items():
        out_lines.append(f"# {label}\n")
        # Compute padding for alignment within this domain (max name length).
        if not names:
            continue
        pad = max(len(n) for n in names) + 1
        for n in names:
            out_lines.append(f'{n:<{pad}}= "{n}"\n')
        out_lines.append("\n")
    return "".join(out_lines)


def apply_to_bus_py(section: str) -> None:
    """Append the new section to bus.py."""
    src = BUS_PY.read_text()
    if "Unregistered literals — registered 2026-04-28 audit" in src:
        print(f"refusing to append: bus.py already has the section",
              file=sys.stderr)
        sys.exit(2)
    BUS_PY.write_text(src.rstrip() + "\n" + section)
    print(f"appended {section.count('=')} constants to {BUS_PY}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true",
                   help="actually edit titan_plugin/bus.py (default: dry-run)")
    args = p.parse_args()

    print("running arch_map dead_wiring --all --json …", file=sys.stderr)
    findings = run_scanner()
    by_domain = collect_unregistered(findings)
    total = sum(len(v) for v in by_domain.values())

    section = render_section(by_domain)

    print(f"\n{total} unregistered literals across {len(by_domain)} domain(s):\n")
    for label, names in by_domain.items():
        print(f"  {label}: {len(names)} ({', '.join(names[:3])}"
              + ("…" if len(names) > 3 else "") + ")")

    print("\n--- proposed bus.py append ---\n")
    print(section)
    print("--- end proposed append ---\n")

    if args.apply:
        apply_to_bus_py(section)
        print("done. next steps:")
        print("  1. re-run arch_map dead_wiring --all to confirm 0 literal findings")
        print("  2. manually replace literals in producer/consumer files using")
        print("     the scanner's reported sites")
        print("  3. optionally add bus_specs.py entries for any P0/P1/P3 needs")
    else:
        print("dry-run only — no files changed. re-run with --apply to commit.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
