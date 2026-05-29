"""`setup_titan config` — browse + edit config.toml + titan_params.toml (W1.f).

CLI-first, comment-driven (the help text IS each key's own inline comment — no
parallel doc, no drift). Consistent with the project's W1.b.2 decision: the CLI
is the canonical engine; a Textual presentation layer is deferred to post-v0.0.1
(the brand UI primitives already live in ui.py).

Surfaces:
    config                      interactive browse/edit loop
    config --list               dump every section.key = value + help
    config --get <sec.key>      print one key's value + help
    config --set <sec.key>=val  edit one key non-interactively (scriptable)

Both files are addressed by the same dotted `section.key`. Edits go through
config_model.set_value (single-line, comment-preserving). Multi-line arrays are
shown read-only with a pointer to edit them in the file directly.
"""
from __future__ import annotations

from pathlib import Path

from .config_model import Entry, coerce_like, find_entry, parse_toml_with_comments, set_value
from .ui import ANSI, GROWTH, HAZE, METAL, PULSE, cprint, section


def _config_files(install_root: Path) -> list[Path]:
    return [
        install_root / "titan_hcl" / "config.toml",
        install_root / "titan_hcl" / "titan_params.toml",
    ]


def _load_all(install_root: Path) -> list[Entry]:
    entries: list[Entry] = []
    for p in _config_files(install_root):
        if p.exists():
            entries.extend(parse_toml_with_comments(p))
    return entries


def _print_entry(e: Entry, *, verbose: bool = False) -> None:
    lock = "" if e.editable else f" {METAL}(multi-line — edit in file){ANSI.RESET}"
    print(f"  {PULSE}{e.dotted}{ANSI.RESET} = {GROWTH}{e.raw_value}{ANSI.RESET}{lock}")
    if verbose and e.help:
        print(f"      {METAL}{e.help}{ANSI.RESET}")


def cmd_list(install_root: Path) -> int:
    entries = _load_all(install_root)
    cur = None
    for e in entries:
        if e.section != cur:
            cur = e.section
            section(cur or "(top-level)")
        _print_entry(e, verbose=True)
    cprint(f"\n  {len(entries)} keys across {len(_config_files(install_root))} files.", role="text_strong")
    return 0


def cmd_get(install_root: Path, dotted: str) -> int:
    matches = find_entry(_load_all(install_root), dotted)
    if not matches:
        cprint(f"  no key matches {dotted!r}.", role="error")
        return 2
    for e in matches:
        section(f"{e.dotted}  [{e.file.name}]")
        print(f"  value: {GROWTH}{e.raw_value}{ANSI.RESET}")
        if e.help:
            print(f"  help:  {METAL}{e.help}{ANSI.RESET}")
        if not e.editable:
            cprint("  (multi-line value — edit directly in the file)", role="warning")
    return 0


def cmd_set(install_root: Path, assignment: str) -> int:
    if "=" not in assignment:
        cprint("  --set expects section.key=value", role="error")
        return 2
    dotted, _, value = assignment.partition("=")
    dotted, value = dotted.strip(), value.strip()
    matches = find_entry(_load_all(install_root), dotted)
    if not matches:
        cprint(f"  no key matches {dotted!r}.", role="error")
        return 2
    if len(matches) > 1:
        cprint(f"  {dotted!r} is ambiguous across files — qualify with the full section path:",
               role="error")
        for e in matches:
            cprint(f"    {e.dotted}  ({e.file.name})", role="text_muted")
        return 2
    e = matches[0]
    if not e.editable:
        cprint(f"  {e.dotted} is a multi-line value — edit it directly in {e.file}.", role="error")
        return 2
    try:
        new_raw = coerce_like(e.raw_value, value)
    except ValueError as exc:
        cprint(f"  {exc}", role="error")
        return 2
    set_value(e.file, e.lineno, new_raw)
    cprint(f"  ✓ {e.dotted}: {e.raw_value} → {new_raw}  ({e.file.name})", role="success")
    return 0


def cmd_interactive(install_root: Path) -> int:
    entries = _load_all(install_root)
    sections: list[str] = []
    for e in entries:
        if e.section not in sections:
            sections.append(e.section)

    section("Titan config explorer")
    cprint("  Browse sections, view each key's own help, edit scalar values.", role="text_muted")
    cprint("  (Edits write back in place — comments + formatting preserved.)\n", role="text_muted")

    while True:
        for i, s in enumerate(sections):
            print(f"  {HAZE}{i:>3}{ANSI.RESET}  {s or '(top-level)'}")
        try:
            pick = input("\n  section # (or 'q' to quit): ").strip()
        except EOFError:
            return 0
        if pick.lower() in ("q", "quit", ""):
            return 0
        if not pick.isdigit() or int(pick) >= len(sections):
            cprint("  invalid selection.", role="warning")
            continue
        sec = sections[int(pick)]
        sec_entries = [e for e in entries if e.section == sec]
        section(sec or "(top-level)")
        for j, e in enumerate(sec_entries):
            print(f"  {HAZE}{j:>3}{ANSI.RESET}  ", end="")
            _print_entry(e, verbose=True)
        try:
            kpick = input("\n  key # to edit (Enter to go back): ").strip()
        except EOFError:
            return 0
        if not kpick:
            entries = _load_all(install_root)   # reload in case of prior edits
            continue
        if not kpick.isdigit() or int(kpick) >= len(sec_entries):
            cprint("  invalid selection.", role="warning")
            continue
        e = sec_entries[int(kpick)]
        if not e.editable:
            cprint(f"  {e.dotted} is multi-line — edit it directly in {e.file}.", role="warning")
            continue
        cprint(f"\n  {e.dotted}", role="accent", bold=True)
        if e.help:
            print(f"  {METAL}{e.help}{ANSI.RESET}")
        try:
            newval = input(f"  new value [{e.raw_value}] (Enter to keep): ").strip()
        except EOFError:
            return 0
        if not newval:
            continue
        try:
            new_raw = coerce_like(e.raw_value, newval)
            set_value(e.file, e.lineno, new_raw)
        except (ValueError, IndexError) as exc:
            cprint(f"  edit failed: {exc}", role="error")
            continue
        cprint(f"  ✓ saved: {e.raw_value} → {new_raw}", role="success")
        entries = _load_all(install_root)       # reload to reflect the edit


def run_config(install_root: Path, *, list_all: bool, get: str | None, set_kv: str | None) -> int:
    if list_all:
        return cmd_list(install_root)
    if get:
        return cmd_get(install_root, get)
    if set_kv:
        return cmd_set(install_root, set_kv)
    return cmd_interactive(install_root)
