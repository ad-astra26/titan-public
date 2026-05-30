"""Prompt seam — one input abstraction shared by the CLI and the Textual TUI.

The phase bodies (`phases`, `inference`, `comms`) never call ``input()`` directly;
they go through a :class:`Prompter`. Two implementations:

* :class:`StdinPrompter` — the historical CLI behaviour (``input()`` against the
  real stdin, brand-coloured hints). The default everywhere, so the CLI path is
  byte-for-byte what it always was.
* :class:`ScriptedPrompter` — replays answers pre-collected by the Textual TUI's
  branded question screens (and by unit tests). Keyed by a stable call-site key,
  so a branching phase flow only ever asks for the keys it actually reaches.

Both front-ends therefore drive the *identical* phase logic — the TUI is a UI
wrapper over the same engine, never a second code path (RFP decision #11).

Outputs (``cprint``) are NOT routed through here: the install walker runs on the
real terminal in both modes, so phase output prints normally.
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Prompter(Protocol):
    """The four input shapes the wizard needs."""

    def line(self, key: str, question: str, *, default: str | None = None) -> str:
        """Free single-line answer (optionally with a default)."""

    def confirm(self, key: str, question: str, *, default_yes: bool) -> bool:
        """Yes/no."""

    def choice(self, key: str, question: str, *, options: list[str], default: str) -> str:
        """Pick one token from ``options`` (returns the chosen token)."""

    def until(self, key: str, question: str, *, validate: Callable[[str], bool],
              hint: str, secret: bool = False) -> str:
        """Loop until ``validate(answer)`` is True (keys, tokens, URLs)."""


# ── CLI (stdin) ──────────────────────────────────────────────────────────────


class StdinPrompter:
    """Historical CLI behaviour — ``input()`` against the real stdin.

    ``key`` is ignored (the question text is the user-visible prompt). Behaviour
    is intentionally identical to the inline ``input()`` calls these methods
    replaced, including the EOFError → SystemExit guard for closed stdin.
    """

    @staticmethod
    def _ask(prompt: str) -> str:
        try:
            return input(prompt).strip()
        except EOFError:
            raise SystemExit(f"setup_titan: stdin closed during prompt: {prompt.strip()!r}")

    def line(self, key: str, question: str, *, default: str | None = None) -> str:
        suffix = f" [{default}]" if default else ""
        return self._ask(f"  {question}{suffix}: ") or (default or "")

    def confirm(self, key: str, question: str, *, default_yes: bool) -> bool:
        suffix = " [Y/n]" if default_yes else " [y/N]"
        ans = self._ask(f"  {question}{suffix}: ").lower()
        if not ans:
            return default_yes
        return ans in ("y", "yes")

    def choice(self, key: str, question: str, *, options: list[str], default: str) -> str:
        ans = self._ask(f"  {question}: ")
        return ans or default

    def until(self, key: str, question: str, *, validate: Callable[[str], bool],
              hint: str, secret: bool = False) -> str:
        while True:
            ans = self._ask(f"  {question}: ")
            if validate(ans):
                return ans
            print(f"    {hint}")


# ── TUI / tests (pre-collected answers) ───────────────────────────────────────


class ScriptedPrompter:
    """Replays answers collected up-front (Textual TUI) or fixed (unit tests).

    ``answers`` maps a call-site ``key`` → the answer (str for line/choice/until,
    bool for confirm). A requested key that is absent raises ``KeyError`` — that
    is a wiring bug (a reachable prompt with no collected answer), surfaced loudly
    rather than silently defaulted. ``until`` still runs ``validate`` so a bad
    pre-collected value fails fast instead of reaching a phase body.
    """

    def __init__(self, answers: dict[str, str | bool]):
        self._answers = dict(answers)

    def _get(self, key: str):
        if key not in self._answers:
            raise KeyError(f"ScriptedPrompter: no pre-collected answer for {key!r}")
        return self._answers[key]

    def line(self, key: str, question: str, *, default: str | None = None) -> str:
        val = self._get(key)
        return str(val) if str(val) else (default or "")

    def confirm(self, key: str, question: str, *, default_yes: bool) -> bool:
        return bool(self._get(key))

    def choice(self, key: str, question: str, *, options: list[str], default: str) -> str:
        val = str(self._get(key))
        return val or default

    def until(self, key: str, question: str, *, validate: Callable[[str], bool],
              hint: str, secret: bool = False) -> str:
        val = str(self._get(key))
        if not validate(val):
            raise ValueError(f"ScriptedPrompter: answer for {key!r} fails validation ({hint})")
        return val
