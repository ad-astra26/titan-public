"""Comment-aware TOML model for `setup_titan config` (W1.f).

The whole point of W1.f (RFP): the config explorer is generated from each key's
OWN inline comments — there is no parallel doc to drift. So this module reads a
TOML file as lines, pairing every `key = value` with the contiguous comment
block immediately above it (plus any trailing inline comment) as that key's
help, and edits values IN PLACE on a single line so all comments + formatting
survive.

Stdlib-only (no tomlkit / tomllib needed): reads are line-based so comments are
preserved; edits replace only the value span of one line. Multi-line arrays are
surfaced read-only (edit-in-file) — scalars + single-line values are editable,
which covers the knobs a tester actually turns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# `[section]` or `[section.sub]` or `[[array_of_tables]]`
_SECTION_RE = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(#.*)?$")
# `  key = value   # inline`  (key chars: alnum, _, -, .)
_KEY_RE = re.compile(r"^(\s*)([A-Za-z0-9_.\-]+)\s*=\s*(.*)$")


@dataclass
class Entry:
    file: Path
    section: str          # "" for top-of-file keys
    key: str
    raw_value: str        # the value text exactly as it appears (no inline comment)
    help: str             # joined preceding comment block + inline comment
    lineno: int           # 1-based line of the `key =` statement
    editable: bool        # False for multi-line arrays/tables

    @property
    def dotted(self) -> str:
        return f"{self.section}.{self.key}" if self.section else self.key


def _split_inline_comment(rest: str) -> tuple[str, str]:
    """Split `value  # comment` into (value, comment), respecting quotes."""
    in_single = in_double = False
    for i, ch in enumerate(rest):
        if ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "'" and not in_double:
            in_single = not in_single
        elif ch == "#" and not in_single and not in_double:
            return rest[:i].rstrip(), rest[i + 1:].strip()
    return rest.rstrip(), ""


def _is_unclosed_array(value: str) -> bool:
    """True if the value opens an array/inline-table that isn't closed on this line."""
    return value.count("[") > value.count("]") or value.count("{") > value.count("}")


def parse_toml_with_comments(path: Path) -> list[Entry]:
    """Return one Entry per key, each carrying its own comment as help."""
    entries: list[Entry] = []
    section = ""
    comment_buf: list[str] = []
    lines = path.read_text().splitlines()

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        if not stripped:                       # blank line resets the help buffer
            comment_buf = []
            i += 1
            continue

        if stripped.startswith("#"):           # accumulate a help comment
            comment_buf.append(stripped.lstrip("#").strip())
            i += 1
            continue

        sm = _SECTION_RE.match(raw)
        if sm and stripped.startswith("["):
            section = sm.group(1).strip()
            comment_buf = []
            i += 1
            continue

        km = _KEY_RE.match(raw)
        if km:
            _indent, key, rest = km.groups()
            value, inline = _split_inline_comment(rest)
            editable = not _is_unclosed_array(value)
            help_parts = [c for c in comment_buf if c]
            if inline:
                help_parts.append(inline)
            entries.append(Entry(
                file=path, section=section, key=key, raw_value=value,
                help=" · ".join(help_parts), lineno=i + 1, editable=editable,
            ))
            comment_buf = []
            # skip the body of a multi-line array so its inner lines aren't
            # mistaken for keys
            if not editable:
                depth = value.count("[") - value.count("]") + value.count("{") - value.count("}")
                while depth > 0 and i + 1 < len(lines):
                    i += 1
                    depth += lines[i].count("[") - lines[i].count("]")
                    depth += lines[i].count("{") - lines[i].count("}")
            i += 1
            continue

        comment_buf = []
        i += 1

    return entries


def coerce_like(old_raw: str, user_input: str) -> str:
    """Render `user_input` in the same TOML shape as `old_raw`.

    - old was a quoted string  → quote the input (stripping any quotes the user added)
    - old was a bool           → require true/false
    - otherwise (number/bare)  → pass through verbatim
    """
    user_input = user_input.strip()
    old = old_raw.strip()
    if old[:1] in ('"', "'"):
        return '"' + user_input.strip("\"'") + '"'
    if old in ("true", "false"):
        low = user_input.lower()
        if low not in ("true", "false"):
            raise ValueError(f"expected true/false, got {user_input!r}")
        return low
    return user_input


def set_value(path: Path, lineno: int, new_raw: str) -> None:
    """Replace the value span on `lineno` (1-based), preserving key + inline comment."""
    lines = path.read_text().splitlines()
    idx = lineno - 1
    if not (0 <= idx < len(lines)):
        raise IndexError(f"line {lineno} out of range for {path}")
    km = _KEY_RE.match(lines[idx])
    if not km:
        raise ValueError(f"line {lineno} of {path} is not a `key = value` line: {lines[idx]!r}")
    indent, key, rest = km.groups()
    _old_value, inline = _split_inline_comment(rest)
    rebuilt = f"{indent}{key} = {new_raw}"
    if inline:
        rebuilt += f"  # {inline}"
    lines[idx] = rebuilt
    path.write_text("\n".join(lines) + "\n")


def find_entry(entries: list[Entry], dotted: str) -> list[Entry]:
    """All entries whose dotted path (or bare key) matches `dotted`."""
    return [e for e in entries if e.dotted == dotted or e.key == dotted]


def set_by_dotted(path: Path, dotted: str, value: str) -> bool:
    """Set an editable key identified by its dotted path (`section.key`).

    Coerces `value` into the existing key's TOML shape (quoted string, bool,
    number) so the file stays valid. Returns True if a unique editable entry
    matched and was written, False otherwise (caller decides how to surface a
    miss — e.g. the key may not exist in this config version).
    """
    matches = [e for e in find_entry(parse_toml_with_comments(path), dotted) if e.editable]
    if not matches:
        return False
    e = matches[0]
    set_value(path, e.lineno, coerce_like(e.raw_value, value))
    return True
