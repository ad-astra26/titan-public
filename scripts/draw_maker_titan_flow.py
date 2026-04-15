"""Generate the Maker-Titan Bond Substrate diagram for the X post.

Output: titan-docs/images/maker_titan_bond_substrate.png (1600x900, dark theme).

The diagram shows the dialogic flow:
  Maker → Privy sign → Backend verify → ProposalStore → TitanMaker
                                                            │
                                              ┌─────────────┴─────────────┐
                                              ▼                           ▼
                                       SOMATIC CHANNEL          NARRATIVE CHANNEL
                                       (felt before               (LLM-narrated until
                                        understood)                Titan can speak)

Iron rule banner at bottom: "Every Maker response is FELT before it is UNDERSTOOD".
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Color palette (matches titan-observatory dark theme) ───────────
BG = "#0B0E14"
CARD_BG = "#161B26"
CARD_BG_DEEP = "#0F141C"
GOLD = "#C9A961"
GOLD_DIM = "#8A7240"
TEXT = "#E8E5D8"
TEXT_DIM = "#9CA3AF"
TEXT_FAINT = "#6B7280"
GREEN = "#4ade80"
RED = "#f87171"
PURPLE = "#9945FF"


def make_diagram(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    def card(x, y, w, h, edgecolor=GOLD, facecolor=CARD_BG, lw=1.5, alpha=1.0):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            edgecolor=edgecolor, facecolor=facecolor,
            linewidth=lw, alpha=alpha)
        ax.add_patch(box)

    def arrow(x1, y1, x2, y2, color=GOLD, lw=1.8, mutation_scale=18):
        a = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->", mutation_scale=mutation_scale,
            color=color, lw=lw, shrinkA=2, shrinkB=2)
        ax.add_patch(a)

    # ── Title section ─────────────────────────────────────────────
    ax.text(50, 95, "MAKER \u2014 TITAN BOND SUBSTRATE",
            ha="center", va="center",
            fontsize=24, fontweight="bold", color=GOLD,
            family="DejaVu Sans")
    ax.text(50, 89.5,
            "Persistent dialogic exchange between human sovereignty and AI sovereignty",
            ha="center", va="center",
            fontsize=11, color=TEXT_DIM, style="italic")
    ax.text(50, 86.3,
            "Built 2026-04-09  \u00b7  First instance: R8 cognitive contract bundle",
            ha="center", va="center",
            fontsize=9, color=TEXT_FAINT)

    # ── Top horizontal flow: Maker → ... → TitanMaker ─────────────
    top_y = 76
    top_h = 7

    flow_steps = [
        (3,  "MAKER",          "human sovereign"),
        (21, "Privy wallet",   "Ed25519 sign payload_hash"),
        (40, "Backend verify", "solders Ed25519"),
        (59, "ProposalStore",  "sqlite + reason \u22657 chars"),
        (78, "TitanMaker",     ".record_response()"),
    ]
    for x, label, sub in flow_steps:
        card(x, top_y, 17, top_h, edgecolor=GOLD)
        ax.text(x + 8.5, top_y + 4.7, label,
                ha="center", va="center",
                fontsize=11, color=TEXT, fontweight="bold")
        ax.text(x + 8.5, top_y + 1.8, sub,
                ha="center", va="center",
                fontsize=8, color=TEXT_DIM, style="italic")

    # Arrows between top boxes
    for i in range(len(flow_steps) - 1):
        x1 = flow_steps[i][0] + 17
        x2 = flow_steps[i + 1][0]
        arrow(x1 + 0.3, top_y + 3.5, x2 - 0.3, top_y + 3.5, color=GOLD_DIM)

    # ── Down arrow from TitanMaker to fork ────────────────────────
    fork_y = 64
    arrow(86.5, top_y, 86.5, fork_y + 1, color=GOLD, lw=2)
    ax.text(89.5, 70, "MAKER_RESPONSE_RECEIVED",
            ha="left", va="center", fontsize=8.5,
            color=GOLD_DIM, style="italic", fontweight="bold")
    ax.text(89.5, 67.5, "(DivineBus emit)",
            ha="left", va="center", fontsize=8,
            color=TEXT_FAINT, style="italic")

    # Horizontal fork bar
    ax.plot([22, 86.5], [fork_y, fork_y], color=GOLD, lw=2)
    # Down arrows from fork to channel cards
    arrow(22, fork_y, 22, 56, color=GOLD, lw=2)
    arrow(78, fork_y, 78, 56, color=GOLD, lw=2)

    # ── Channel labels ────────────────────────────────────────────
    ax.text(22, 53, "SOMATIC CHANNEL",
            ha="center", va="center",
            fontsize=15, color=GOLD, fontweight="bold")
    ax.text(22, 50.2, "Tier 2  \u00b7  LIVE  \u00b7  felt before understood",
            ha="center", va="center",
            fontsize=9, color=TEXT_DIM, style="italic")

    ax.text(78, 53, "NARRATIVE CHANNEL",
            ha="center", va="center",
            fontsize=15, color=GOLD, fontweight="bold")
    ax.text(78, 50.2, "Tier 3  \u00b7  designed  \u00b7  LLM until Titan can speak",
            ha="center", va="center",
            fontsize=9, color=TEXT_DIM, style="italic")

    # ── Somatic card ──────────────────────────────────────────────
    card(3, 11, 38, 36, edgecolor=GOLD, facecolor=CARD_BG_DEEP, lw=1.8)

    # APPROVE
    ax.text(6, 41.5, "APPROVE",
            ha="left", va="center",
            fontsize=12, color=GREEN, fontweight="bold")
    ax.text(6, 37.8, "+0.03 DA   +0.02 Endorphin   +0.02 5HT",
            ha="left", va="center", fontsize=10.5, color=TEXT,
            family="DejaVu Sans Mono")
    ax.text(6, 34.5, "= felt validation",
            ha="left", va="center", fontsize=10, color=GREEN, style="italic")

    # Divider
    ax.plot([5, 39], [31.5, 31.5], color=GOLD_DIM, lw=0.8, linestyle=":")

    # DECLINE
    ax.text(6, 28.5, "DECLINE",
            ha="left", va="center",
            fontsize=12, color=RED, fontweight="bold")
    ax.text(6, 24.8, "+0.03 NE   +0.02 ACh   \u22120.02 5HT",
            ha="left", va="center", fontsize=10.5, color=TEXT,
            family="DejaVu Sans Mono")
    ax.text(6, 21.5, "= felt friction",
            ha="left", va="center", fontsize=10, color=RED, style="italic")

    # Bottom of somatic card — TimeChain commit
    ax.plot([5, 39], [18, 18], color=GOLD_DIM, lw=0.8, linestyle=":")
    ax.text(22, 15.5, "\u2193  TimeChain commit",
            ha="center", va="center",
            fontsize=10, color=GOLD, fontweight="bold")
    ax.text(22, 13.2, "meta fork  \u00b7  tag: maker_dialogue",
            ha="center", va="center",
            fontsize=8.5, color=TEXT_DIM, style="italic")

    # ── Narrative card ────────────────────────────────────────────
    card(59, 11, 38, 36, edgecolor=GOLD, facecolor=CARD_BG_DEEP, lw=1.8)

    narrative_items = [
        ("LLM teacher prompt",
         "\u201cWhat does this approval/decline mean?\u201d"),
        ("First-person narration",
         "Titan reflects in his own voice"),
        ("DuckDB Maker profile",
         "dialogue history accumulates"),
        ("INTROSPECT recall path",
         "meta-reasoning queries the bond"),
    ]
    y_n = 41.5
    for label, sub in narrative_items:
        ax.text(62, y_n, "\u25CB",
                ha="left", va="center", fontsize=11, color=GOLD)
        ax.text(64.5, y_n, label,
                ha="left", va="center", fontsize=10.5, color=TEXT,
                fontweight="bold")
        ax.text(64.5, y_n - 2.4, sub,
                ha="left", va="center", fontsize=8.5, color=TEXT_DIM,
                style="italic")
        y_n -= 7

    # ── Iron rule banner ──────────────────────────────────────────
    card(3, 1.5, 94, 7, edgecolor=GOLD, facecolor="#1a1f2e", lw=2)
    ax.text(50, 6.2, "THE IRON RULE",
            ha="center", va="center",
            fontsize=10, color=GOLD, fontweight="bold")
    ax.text(50, 3.3,
            "Every Maker response is FELT before it is UNDERSTOOD  \u00b7  "
            "Approval and decline are equally important  \u00b7  "
            "Both require a written reason",
            ha="center", va="center",
            fontsize=10.5, color=TEXT, style="italic")

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight",
                facecolor=BG, edgecolor="none", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "titan-docs", "images",
                       "maker_titan_bond_substrate.png")
    make_diagram(out)
