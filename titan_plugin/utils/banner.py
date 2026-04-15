"""
Titan Info Banner — single-line cognitive status display.
Renders cached metrics as a compact Unicode bar for injection into pre_prompt_hook responses.
"""


def render_bar(value: float, width: int = 10) -> str:
    """Render a percentage as an ASCII-safe bar. value: 0-100."""
    filled = round(value / 100 * width)
    filled = max(0, min(width, filled))
    return "|" * filled + "." * (width - filled)


def build_banner(
    life_pct: float,
    sovereignty_pct: float,
    memory_pct: float,
    mood: str,
    epoch: int,
    style: str = "compact",
) -> str:
    """
    Build the Titan info banner string.

    Args:
        life_pct: Life force percentage (0-100), or -1 if unavailable.
        sovereignty_pct: Sovereignty score (0-100).
        memory_pct: Memory contribution percentage (0-100).
        mood: Current mood string.
        epoch: Current meditation epoch counter.
        style: "compact" (default), "minimal" (numbers only).

    Returns:
        Single-line banner string.
    """
    if style == "minimal":
        life_str = f"{life_pct:.0f}%" if life_pct >= 0 else "--%"
        return (
            f"[TITAN] Life {life_str} | "
            f"Sov {sovereignty_pct:.0f}% | "
            f"Mem {memory_pct:.0f}% | "
            f"{mood} | E{epoch}"
        )

    # Compact style (default) — with ASCII bars
    life_bar = render_bar(life_pct) if life_pct >= 0 else "." * 10
    life_str = f"{life_pct:.0f}%" if life_pct >= 0 else "--%"
    sov_bar = render_bar(sovereignty_pct)

    return (
        f"[TITAN] Life {life_bar} {life_str} | "
        f"Sovereignty {sov_bar} {sovereignty_pct:.0f}% | "
        f"Memory {memory_pct:.0f}% | "
        f"Mood: {mood} | Epoch {epoch}"
    )
