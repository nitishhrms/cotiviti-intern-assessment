"""
Render supporting figures for the written report as PNGs:
  - trends_timeline.png : Past / Present / Future evolution of clinical NLP
  - strategy_actions.png: the three recommended strategic actions for Cotiviti

Style matches make_hld.py (Cotiviti-inspired palette).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

PURPLE = "#4B2E83"
MAGENTA = "#C4007A"
TEAL = "#009E8F"
ORANGE = "#E07A00"
BAND = "#F1EDFA"
DARK = "#222222"
GREY = "#5A5A5A"
HERE = os.path.dirname(__file__)


def _card(ax, x, y, w, h, fc=BAND, ec="none"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.0, zorder=2))


# ==========================================================================
# Figure 1 — Trends timeline
# ==========================================================================
def trends_timeline():
    fig, ax = plt.subplots(figsize=(11.5, 3.5), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    eras = [
        ("PAST", "2000 - 2017", "Rules & Terminologies", PURPLE,
         ["cTAKES, MetaMap", "UMLS / SNOMED CT / ICD", "Accurate but brittle"]),
        ("PRESENT", "2018 - 2023", "Clinical Transformers", MAGENTA,
         ["BioBERT, ClinicalBERT,", "GatorTron; DenseNet vision", "Learns from large data"]),
        ("FUTURE", "2024 onward", "Generative & Multimodal", TEAL,
         ["Clinical LLMs & LMMs", "Image + text reasoning", "RAG grounding for safety"]),
    ]
    x = 2.0
    cw = 30.0
    gap = 3.0
    for kicker, yrs, title, col, items in eras:
        # header
        ax.add_patch(FancyBboxPatch((x, 30), cw, 7.5, boxstyle="round,pad=0.2,rounding_size=1.0",
                                    fc=col, ec="none", zorder=3))
        ax.text(x + cw / 2, 35.4, kicker + "   " + yrs, ha="center", va="center",
                color="white", fontsize=10.5, fontweight="bold", zorder=4)
        ax.text(x + cw / 2, 31.9, title, ha="center", va="center",
                color="white", fontsize=11.5, fontweight="bold", zorder=4)
        # body card
        _card(ax, x, 4, cw, 24)
        for i, it in enumerate(items):
            ax.text(x + 2.2, 23 - i * 6.0, "• " + it, ha="left", va="center",
                    color=DARK, fontsize=10.2, zorder=4)
        x += cw + gap

    # flow arrows between cards
    for ax_x in (33.0, 66.0):
        ax.add_patch(FancyArrowPatch((ax_x, 16), (ax_x + 3.0, 16), arrowstyle="-|>",
                                     mutation_scale=16, color="#9A8AC0", lw=2.4, zorder=5))
    out = os.path.join(HERE, "trends_timeline.png")
    plt.savefig(out, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    print("Saved:", out)


# ==========================================================================
# Figure 2 — Strategic actions
# ==========================================================================
def strategy_actions():
    fig, ax = plt.subplots(figsize=(11.5, 3.0), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

    actions = [
        ("01", "Grounded NLP Layer", PURPLE,
         ["Medical encoders + RAG over", "coding policies & guidelines;", "every output cites a source"]),
        ("02", "Multimodal Review", MAGENTA,
         ["OCR + NLP + Computer Vision", "reconcile imaging and notes", "against submitted claims"]),
        ("03", "Human-in-the-Loop", TEAL,
         ["Version, monitor drift, and", "route low-confidence cases", "to expert reviewers"]),
    ]
    x = 2.0
    cw = 30.0
    gap = 3.0
    for num, title, col, items in actions:
        _card(ax, x, 2, cw, 30, fc="#FFFFFF", ec="#E2DDEE")
        ax.add_patch(FancyBboxPatch((x, 2), 0.9, 30, boxstyle="square,pad=0",
                                    fc=col, ec="none", zorder=3))
        cx = x + cw / 2
        ax.add_patch(Circle((cx, 26.5), 2.6, fc=col, ec="none", zorder=4))
        ax.text(cx, 26.5, num, ha="center", va="center", color="white",
                fontsize=11, fontweight="bold", zorder=5)
        ax.text(cx, 20.8, title, ha="center", va="center", color=col,
                fontsize=10.8, fontweight="bold", zorder=5)
        for i, it in enumerate(items):
            ax.text(x + 3.0, 15.0 - i * 4.6, it, ha="left", va="center",
                    color=DARK, fontsize=9.6, zorder=5)
        x += cw + gap
    out = os.path.join(HERE, "strategy_actions.png")
    plt.savefig(out, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    print("Saved:", out)


if __name__ == "__main__":
    trends_timeline()
    strategy_actions()
