#!/usr/bin/env python3
"""Trinity SPEC-conformance checker (Layer 2 of the conformance gate).

Single machine-checked answer to "is the trinity 100% per spec?". Parses the
clause→test manifest (titan-docs/specs/TRINITY_CONFORMANCE.md), runs the Rust
conformance suite (titan-trinity-daemon/tests/trinity_conformance.rs), and:

  * BLOCKS (exit 2) if any LOCKED clause's test is MISSING or RED, if any clause
    has no mapped test, or if a manifest test name doesn't exist in the suite.
  * REPORTS PENDING clauses loudly (so non-conformance is never invisible) but
    does NOT block routine commits on them — UNLESS --strict (used to gate
    "P0 done", which requires every clause LOCKED + green).

Exit codes: 0 = conformant (gate passes), 2 = blocked (LOCKED regression /
unmapped / missing test), 3 = --strict and PENDING clauses remain.

Usage:
    python scripts/trinity_conformance.py [--strict] [--no-run]
    python scripts/arch_map.py trinity-conformance [--strict]
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "titan-docs" / "specs" / "TRINITY_CONFORMANCE.md"
RUST_DIR = REPO / "titan-rust"
RUST_TEST = "trinity_conformance"  # tests/trinity_conformance.rs
RUST_CRATE = "titan-trinity-daemon"


def _load_manifest() -> list[dict]:
    """Parse the ```yaml clause block out of the manifest (no PyYAML dependency)."""
    text = MANIFEST.read_text(encoding="utf-8")
    m = re.search(r"```yaml\n(.*?)```", text, re.DOTALL)
    if not m:
        sys.exit(f"FATAL: no ```yaml clause block in {MANIFEST}")
    body = m.group(1)
    clauses: list[dict] = []
    cur: dict | None = None
    for raw in body.splitlines():
        line = raw.rstrip()
        if re.match(r"\s*#", line) or not line.strip():
            continue
        mid = re.match(r"\s*-\s+id:\s*(\S+)", line)
        if mid:
            if cur:
                clauses.append(cur)
            cur = {"id": mid.group(1)}
            continue
        mkv = re.match(r"\s+(\w+):\s*(.*)", line)
        if mkv and cur is not None:
            key, val = mkv.group(1), mkv.group(2).strip()
            # strip trailing inline comment + surrounding quotes
            val = re.sub(r"\s+#.*$", "", val).strip().strip('"')
            cur[key] = val
    if cur:
        clauses.append(cur)
    return clauses


# Rust test targets the gate runs. The §G5.2 kernel conformance suite lives in
# titan-trinity-daemon; the §G11 pulse clauses are enforced by the SHIPPED
# resonance + boot unit tests in titan-unified-spirit-rs (reused, not duplicated).
RUST_TARGETS = [
    ["cargo", "test", "-p", "titan-trinity-daemon", "--test", "trinity_conformance"],
    ["cargo", "test", "-p", "titan-unified-spirit-rs", "--lib"],
]


def _run_rust_suite() -> dict[str, str]:
    """Run the Rust conformance targets; return {full_test_path: 'ok'|'FAILED'|'ignored'}."""
    results: dict[str, str] = {}
    for cmd in RUST_TARGETS:
        try:
            proc = subprocess.run(
                cmd + ["--", "--format", "pretty"],
                cwd=RUST_DIR, capture_output=True, text=True, timeout=600,
            )
        except FileNotFoundError:
            sys.exit("FATAL: cargo not found — cannot run the Rust conformance suite")
        except subprocess.TimeoutExpired:
            sys.exit(f"FATAL: conformance target timed out: {' '.join(cmd)}")
        out = proc.stdout + proc.stderr
        if ("error[" in out) or ("error:" in out and "test result" not in out):
            print(out[-3000:], file=sys.stderr)
            sys.exit(f"FATAL: conformance target failed to COMPILE: {' '.join(cmd)}")
        for line in out.splitlines():
            m = re.match(r"test\s+(\S+)\s+\.\.\.\s+(ok|FAILED|ignored)", line)
            if m:
                results[m.group(1)] = m.group(2)
    return results


def _lookup(results: dict[str, str], test: str) -> str:
    """Match a manifest test name against full Rust test paths by exact-or-suffix."""
    if test in results:
        return results[test]
    for full, res in results.items():
        if full == test or full.endswith("::" + test):
            return res
    return "MISSING"


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    no_run = "--no-run" in argv

    clauses = _load_manifest()
    rust = {} if no_run else _run_rust_suite()

    blocked: list[str] = []       # LOCKED regressions / unmapped / missing → exit 2
    pending: list[str] = []       # known gaps (reported; block only under --strict)
    ok: list[str] = []

    rows = []
    for c in clauses:
        cid = c.get("id", "?")
        status = c.get("status", "PENDING")
        test = c.get("test")
        is_py = bool(test) and test.startswith("test_")  # Python pytest clause (e.g. MSL)
        if not test:
            blocked.append(f"{cid}: NO test mapped (every clause must map to a test)")
            rows.append((cid, status, "—", "✗ UNMAPPED"))
            continue
        # Python-runner clauses are not executed here (run via pytest); report by status.
        if is_py:
            res = "py:pending" if status == "PENDING" else "py:assumed"
        else:
            res = _lookup(rust, test)

        if status == "LOCKED":
            if res == "ok":
                ok.append(cid)
                rows.append((cid, status, test, "✓ green"))
            elif res == "MISSING":
                blocked.append(f"{cid}: LOCKED but test '{test}' NOT FOUND in suite")
                rows.append((cid, status, test, "✗ MISSING"))
            else:
                blocked.append(f"{cid}: LOCKED but test '{test}' is {res} (REGRESSION)")
                rows.append((cid, status, test, f"✗ {res}"))
        else:  # PENDING
            pending.append(cid)
            mark = "⚠ red" if res in ("FAILED", "MISSING", "py:pending") else f"⚠ {res}"
            rows.append((cid, status, test, f"{mark} (PENDING)"))

    # ── report ───────────────────────────────────────────────────────────────
    print("\n=== Trinity SPEC-Conformance ===")
    w = max(len(r[0]) for r in rows)
    for cid, status, test, mark in rows:
        print(f"  {cid:<{w}}  {status:<11}  {mark:<16}  {test}")
    print(
        f"\n  LOCKED green: {len(ok)}   PENDING (gaps): {len(pending)}   "
        f"BLOCKING: {len(blocked)}   total clauses: {len(clauses)}"
    )

    if blocked:
        print("\n🚫 BLOCKED — trinity is not committable:")
        for b in blocked:
            print(f"   • {b}")
        return 2
    if pending:
        print("\n⚠️  TRINITY NOT YET 100% PER SPEC — PENDING clauses (tracked, not yet conforming):")
        for p in pending:
            print(f"   • {p}")
        if strict:
            print("\n🚫 --strict: PENDING clauses remain → P0 NOT done.")
            return 3
        print("   (routine commits allowed; promote each to LOCKED with a green test as it ships)")
    else:
        print("\n✅ All clauses LOCKED + green — trinity is 100% per the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
