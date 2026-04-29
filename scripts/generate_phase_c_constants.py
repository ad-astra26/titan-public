#!/usr/bin/env python3
"""
generate_phase_c_constants.py — regenerator for Titan SPEC constants.

Reads `titan-docs/SPEC_titan_architecture_constants.toml` (single source of truth
per SPEC §19) and emits two byte-identical generated files:

  - titan_plugin/_phase_c_constants.py   (Python `Final[T]` annotations)
  - titan-rust/crates/titan-core/src/constants.rs  (Rust `pub const NAME: T = V;`)

Both files have an AUTO-GENERATED header — hand-editing them is a SPEC violation
flagged by `arch_map phase-c verify` (per SPEC §19).

Usage:
    python scripts/generate_phase_c_constants.py [--check]

    --check   Compare generated output against committed files; exit non-zero
              on any difference (CI parity test mode).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent
TOML_PATH = REPO_ROOT / "titan-docs" / "SPEC_titan_architecture_constants.toml"
PYTHON_OUT = REPO_ROOT / "titan_plugin" / "_phase_c_constants.py"
RUST_DIR = REPO_ROOT / "titan-rust" / "crates" / "titan-core" / "src"
RUST_OUT = RUST_DIR / "constants.rs"


# ── Type mapping ────────────────────────────────────────────────────────

PYTHON_TYPE_MAP = {
    "uint": "int",
    "int": "int",
    "float": "float",
    "string": "str",
    "bool": "bool",
}

RUST_TYPE_MAP = {
    "uint": "u64",
    "int": "i64",
    "float": "f64",
    "string": "&str",
    "bool": "bool",
}


def _python_value(value, ctype: str, unit: str) -> str:
    """Format a value as a Python literal."""
    if ctype == "string":
        # bytes_utf8 unit signals "this is a bytes constant in Python"
        if unit == "bytes_utf8":
            return f'b"{value}"'
        return f'"{value}"'
    if ctype == "bool":
        return "True" if value else "False"
    return repr(value)


def _python_type(ctype: str, unit: str) -> str:
    """Pick the Python annotation."""
    if ctype == "string" and unit == "bytes_utf8":
        return "bytes"
    return PYTHON_TYPE_MAP[ctype]


def _rust_value(value, ctype: str, unit: str) -> str:
    """Format a value as a Rust literal."""
    if ctype == "string":
        if unit == "bytes_utf8":
            # bytes constant — emit as b"..." byte string literal
            return f'b"{value}"'
        return f'"{value}"'
    if ctype == "bool":
        return "true" if value else "false"
    if ctype == "float":
        # Always include decimal point for f64
        s = repr(float(value))
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s
    # uint / int
    return str(value)


def _rust_type(ctype: str, unit: str) -> str:
    """Pick the Rust type."""
    if ctype == "string" and unit == "bytes_utf8":
        # &[u8] for byte literals — but byte string literals have type &[u8; N]
        # so we use a more permissive annotation
        return "&[u8]"
    return RUST_TYPE_MAP[ctype]


# ── Generators ──────────────────────────────────────────────────────────

PY_HEADER = '''"""
{name} — AUTO-GENERATED from {source}.

DO NOT EDIT BY HAND. Edit the TOML, then run:
    python scripts/generate_phase_c_constants.py

SPEC version: {spec_version}
Source SHA-256: {toml_sha256}

Per SPEC §19 + §2.6: hand-editing this file is a SPEC violation flagged by
`arch_map phase-c verify`.
"""
from __future__ import annotations
from typing import Final

# ── SPEC version metadata ──────────────────────────────────────────────
SPEC_VERSION: Final[str] = "{spec_version}"
SPEC_SOURCE_SHA256: Final[str] = "{toml_sha256}"
'''

def _rustfmt_in_memory(content: str) -> str:
    """Pipe `content` through rustfmt and return the formatted output.

    Best-effort — falls through to the input unchanged if rustfmt is missing
    (e.g. on T2/T3 deploy hosts) or fails. Pre-commit hook on T1 (where
    rustfmt is always installed) is the authoritative drift-catcher.
    """
    import shutil as _shutil
    import subprocess as _sub
    if not _shutil.which("rustfmt"):
        return content
    try:
        result = _sub.run(
            ["rustfmt", "--edition", "2021"],
            input=content,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except (OSError, _sub.SubprocessError):
        pass
    return content


RUST_HEADER = '''//! {name} — AUTO-GENERATED from {source}.
//!
//! DO NOT EDIT BY HAND. Edit the TOML, then run:
//!     python scripts/generate_phase_c_constants.py
//!
//! SPEC version: {spec_version}
//! Source SHA-256: {toml_sha256}
//!
//! Per SPEC §19 + §2.6: hand-editing this file is a SPEC violation flagged by
//! `arch_map phase-c verify`.
#![allow(dead_code)]

// ── SPEC version metadata ──────────────────────────────────────────────
pub const SPEC_VERSION: &str = "{spec_version}";
pub const SPEC_SOURCE_SHA256: &str = "{toml_sha256}";
'''


def _read_toml() -> tuple[dict, str]:
    """Load the TOML and compute its SHA-256 for drift detection."""
    raw = TOML_PATH.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    return tomllib.loads(raw.decode("utf-8")), sha


def _group_by_domain(constants: dict) -> dict[str, list[tuple[str, dict]]]:
    """Group constants by their domain field, preserving TOML order within each."""
    groups: dict[str, list[tuple[str, dict]]] = {}
    for name, spec in constants.items():
        domain = spec.get("domain", "UNKNOWN")
        groups.setdefault(domain, []).append((name, spec))
    return groups


def render_python(toml: dict, toml_sha: str) -> str:
    """Render the Python constants module."""
    spec_version = toml.get("spec_version", "0.0.0")
    constants = toml.get("constants", {})
    domains = toml.get("domains", {})
    grouped = _group_by_domain(constants)

    lines: list[str] = []
    lines.append(
        PY_HEADER.format(
            name="_phase_c_constants.py",
            source="titan-docs/SPEC_titan_architecture_constants.toml",
            spec_version=spec_version,
            toml_sha256=toml_sha,
        )
    )

    # Emit constants grouped by domain in domain registry order, then any leftover
    domain_order = list(domains.keys())
    seen_domains: set[str] = set()
    for domain in domain_order:
        if domain not in grouped:
            continue
        seen_domains.add(domain)
        lines.append("")
        lines.append(f"# ── {domain} " + "─" * (70 - len(domain)))
        desc = domains[domain].get("description", "")
        if desc:
            lines.append(f"# {desc}")
        lines.append("")
        for name, spec in grouped[domain]:
            ctype = spec["type"]
            unit = spec.get("unit", "")
            value = _python_value(spec["value"], ctype, unit)
            ann = _python_type(ctype, unit)
            description = spec.get("description", "").strip()
            if description:
                lines.append(f"# {description}")
            lines.append(f"{name}: Final[{ann}] = {value}")
        lines.append("")

    # Any constants whose domain isn't registered (shouldn't happen if TOML is clean)
    for domain, items in grouped.items():
        if domain in seen_domains:
            continue
        lines.append(f"# ── {domain} (UNREGISTERED DOMAIN — fix TOML) ──")
        for name, spec in items:
            ctype = spec["type"]
            unit = spec.get("unit", "")
            value = _python_value(spec["value"], ctype, unit)
            ann = _python_type(ctype, unit)
            lines.append(f"{name}: Final[{ann}] = {value}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_rust(toml: dict, toml_sha: str) -> str:
    """Render the Rust constants module."""
    spec_version = toml.get("spec_version", "0.0.0")
    constants = toml.get("constants", {})
    domains = toml.get("domains", {})
    grouped = _group_by_domain(constants)

    lines: list[str] = []
    lines.append(
        RUST_HEADER.format(
            name="constants.rs",
            source="titan-docs/SPEC_titan_architecture_constants.toml",
            spec_version=spec_version,
            toml_sha256=toml_sha,
        )
    )

    domain_order = list(domains.keys())
    seen_domains: set[str] = set()
    for domain in domain_order:
        if domain not in grouped:
            continue
        seen_domains.add(domain)
        lines.append("")
        lines.append(f"// ── {domain} " + "─" * (70 - len(domain)))
        desc = domains[domain].get("description", "")
        if desc:
            lines.append(f"// {desc}")
        lines.append("")
        for name, spec in grouped[domain]:
            ctype = spec["type"]
            unit = spec.get("unit", "")
            value = _rust_value(spec["value"], ctype, unit)
            rtype = _rust_type(ctype, unit)
            description = spec.get("description", "").strip()
            if description:
                lines.append(f"/// {description}")
            lines.append(f"pub const {name}: {rtype} = {value};")
        lines.append("")

    for domain, items in grouped.items():
        if domain in seen_domains:
            continue
        lines.append(f"// ── {domain} (UNREGISTERED DOMAIN — fix TOML) ──")
        for name, spec in items:
            ctype = spec["type"]
            unit = spec.get("unit", "")
            value = _rust_value(spec["value"], ctype, unit)
            rtype = _rust_type(ctype, unit)
            lines.append(f"pub const {name}: {rtype} = {value};")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare generated output to committed files; non-zero on diff (CI mode)",
    )
    args = parser.parse_args()

    if not TOML_PATH.exists():
        print(f"ERROR: TOML not found at {TOML_PATH}", file=sys.stderr)
        return 2

    toml, toml_sha = _read_toml()
    py_content = render_python(toml, toml_sha)
    rust_content = render_rust(toml, toml_sha)

    # Apply rustfmt to the in-memory Rust content so the output is byte-
    # identical to what `cargo fmt --check` expects. Without this step,
    # `cargo fmt` running anywhere in the workspace silently reformats
    # `constants.rs` (long-string line wrapping, double-blank-line collapse),
    # which then trips the GENERATOR_PARITY check on the next verify.
    # Codified 2026-04-29 during C-S2 chunk C2-4.
    rust_content = _rustfmt_in_memory(rust_content)

    if args.check:
        # CI mode — diff against committed
        problems: list[str] = []
        for path, content in [(PYTHON_OUT, py_content), (RUST_OUT, rust_content)]:
            if not path.exists():
                problems.append(f"MISSING: {path.relative_to(REPO_ROOT)}")
                continue
            existing = path.read_text(encoding="utf-8")
            if existing != content:
                problems.append(f"DRIFT: {path.relative_to(REPO_ROOT)} (regenerate from TOML)")
        if problems:
            for p in problems:
                print(f"  ✗ {p}", file=sys.stderr)
            print(
                "\nFIX: run `python scripts/generate_phase_c_constants.py` then commit.",
                file=sys.stderr,
            )
            return 1
        print(f"✓ Generated files in sync with TOML (SHA-256 {toml_sha[:12]}...)")
        return 0

    # Write mode
    PYTHON_OUT.parent.mkdir(parents=True, exist_ok=True)
    RUST_DIR.mkdir(parents=True, exist_ok=True)

    PYTHON_OUT.write_text(py_content, encoding="utf-8")
    RUST_OUT.write_text(rust_content, encoding="utf-8")

    n_constants = len(toml.get("constants", {}))
    n_domains = len(toml.get("domains", {}))
    n_truths = len(toml.get("ground_truths", {}))
    print(
        f"✓ Generated {n_constants} constants across {n_domains} domains "
        f"+ {n_truths} ground truths"
    )
    print(f"  → {PYTHON_OUT.relative_to(REPO_ROOT)}")
    print(f"  → {RUST_OUT.relative_to(REPO_ROOT)}")
    print(f"  SPEC v{toml.get('spec_version')} | TOML SHA-256: {toml_sha[:12]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
