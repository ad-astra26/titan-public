"""Titan brand UI primitives — palette, Textual CSS, base App.

Canonical brand source: `titan-console/src/index.css` (the TC² SPA brand tokens).
Semantic names are used everywhere — never raw hex outside this file.

Roles:
    bg / card       surface
    metal           muted text / borders (low-emphasis)
    haze            gold accent — primary highlight, current focus
    pulse           purple — secondary accent, action / Maker
    growth          teal — success, "alive", positive state
    danger          red — only for genuine errors

Glows mirror the TC² console's haze-glow / pulse-glow / growth-glow shadows.
"""
from __future__ import annotations

# ── canonical brand palette (mirrors titan-console/src/index.css) ──
BRAND = {
    "bg":      "#0B0E14",  # --titan-bg
    "card":    "#1A1D23",  # --titan-card
    "metal":   "#8E9AAF",  # --titan-metal — muted text + borders
    "haze":    "#E5C79E",  # --titan-haze — gold primary accent
    "pulse":   "#9945FF",  # --titan-pulse — purple secondary accent
    "growth":  "#77CCCC",  # --titan-growth — teal success/alive
    "danger":  "#ff6b6b",  # error-only red (not yet in Observatory vars)
}

# Role aliases (so semantic intent, not hue, drives UI choices)
ROLE = {
    "surface":     BRAND["bg"],
    "card":        BRAND["card"],
    "text_muted":  BRAND["metal"],
    "text_strong": BRAND["haze"],
    "accent":      BRAND["pulse"],
    "success":     BRAND["growth"],
    "warning":     BRAND["haze"],   # gold doubles as warning per brand intent
    "error":       BRAND["danger"],
}


# ── Textual stylesheet — uses brand hexes only via this constant ──
TITAN_CSS = f"""
Screen {{
    background: {BRAND["bg"]};
    color: {BRAND["metal"]};
}}

.brand-title {{
    color: {BRAND["haze"]};
    text-style: bold;
    content-align: center middle;
    padding: 1;
}}

.brand-subtitle {{
    color: {BRAND["metal"]};
    content-align: center middle;
}}

.brand-card {{
    background: {BRAND["card"]};
    border: solid {BRAND["pulse"]};
    padding: 1 2;
    margin: 1 2;
}}

.brand-card-accent {{
    background: {BRAND["card"]};
    border: solid {BRAND["growth"]};
    padding: 1 2;
    margin: 1 2;
}}

.brand-haze    {{ color: {BRAND["haze"]}; }}
.brand-pulse   {{ color: {BRAND["pulse"]}; }}
.brand-growth  {{ color: {BRAND["growth"]}; }}
.brand-metal   {{ color: {BRAND["metal"]}; }}
.brand-danger  {{ color: {BRAND["danger"]}; text-style: bold; }}

Button.-primary {{
    background: {BRAND["pulse"]};
    color: white;
}}
Button.-success {{
    background: {BRAND["growth"]};
    color: {BRAND["bg"]};
}}

Input {{
    border: solid {BRAND["pulse"]};
}}
Input:focus {{
    border: solid {BRAND["haze"]};
}}

Footer {{
    background: {BRAND["card"]};
    color: {BRAND["metal"]};
}}
Header {{
    background: {BRAND["card"]};
    color: {BRAND["haze"]};
}}
"""


# ── ANSI helpers (for non-Textual contexts — bootstrap, error paths, --dry-run) ──
class ANSI:
    """ANSI escape sequences using the brand palette via truecolor."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    @staticmethod
    def _fg(hex_color: str) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"\033[38;2;{r};{g};{b}m"

HAZE   = ANSI._fg(BRAND["haze"])
PULSE  = ANSI._fg(BRAND["pulse"])
GROWTH = ANSI._fg(BRAND["growth"])
METAL  = ANSI._fg(BRAND["metal"])
DANGER = ANSI._fg(BRAND["danger"])


def cprint(msg: str, role: str = "text_muted", bold: bool = False) -> None:
    """Print one line in a brand role color, ANSI-only (no Textual)."""
    color_hex = ROLE.get(role, BRAND["metal"])
    prefix = ANSI._fg(color_hex) + (ANSI.BOLD if bold else "")
    print(f"{prefix}{msg}{ANSI.RESET}")


def section(title: str) -> None:
    """Print a brand-styled section header (ANSI). Use in plain-text contexts."""
    print()
    cprint(f"── {title} " + "─" * max(0, 60 - len(title)), role="text_strong", bold=True)
