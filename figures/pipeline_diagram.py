"""
figures/pipeline_diagram.py
----------------------------
Regenerates Fig. 1 from the paper — ImFREQ-Lite pipeline block diagram.

Run:
    python figures/pipeline_diagram.py

Output:
    figures/fig1_pipeline.png  (260 DPI, publication quality)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Color palette (IEEE-friendly, monochrome-safe) ────────────────────────────
C_IO    = "#EAF4FB"   # input/output boxes
C_STAGE = "#D6EAF8"   # processing stage boxes
C_EDGE  = "#1A5276"   # border color
C_ARROW = "#1A5276"   # arrow color
C_LABEL = "#1A252F"   # text
C_SUB   = "#626567"   # subtitle
C_SNUM  = "#2E86C1"   # stage number


def draw_box(ax, x, y, w, h, title, sub="", io=False):
    fc = C_IO if io else C_STAGE
    lw = 1.8 if io else 1.5
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.10",
        facecolor=fc, edgecolor=C_EDGE,
        linewidth=lw, zorder=3,
    )
    ax.add_patch(rect)
    cx, cy = x + w / 2, y + h / 2
    dy = 0.14 if sub else 0
    ax.text(cx, cy + dy, title,
            ha="center", va="center",
            fontsize=8.8, fontweight="bold",
            color=C_LABEL, zorder=4)
    if sub:
        ax.text(cx, cy - 0.18, sub,
                ha="center", va="center",
                fontsize=7.2, style="italic",
                color=C_SUB, zorder=4)


def draw_arrow(ax, x1, x2, y=1.90):
    ax.annotate(
        "", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(
            arrowstyle="-|>", color=C_ARROW,
            lw=1.8, mutation_scale=15,
        ),
        zorder=5,
    )


def stage_label(ax, cx, y_top, n):
    ax.text(cx, y_top + 0.10, f"S{n}",
            ha="center", va="bottom",
            fontsize=8.0, fontweight="bold",
            color=C_SNUM, zorder=4)


def main():
    fig = plt.figure(figsize=(13.5, 3.8), facecolor="white")
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    W_IO  = 1.30   # I/O box width
    W_STG = 1.78   # stage box width
    H     = 1.40   # box height
    Y_BOX = 1.20   # bottom y  (center at 1.90)
    GAP   = 0.24

    XS = [
        0.18,                                    # IoT Stream  (I/O)
        0.18 + W_IO + GAP,                       # S1
        0.18 + W_IO + GAP +   (W_STG + GAP),    # S2
        0.18 + W_IO + GAP + 2*(W_STG + GAP),    # S3
        0.18 + W_IO + GAP + 3*(W_STG + GAP),    # S4
        0.18 + W_IO + GAP + 4*(W_STG + GAP),    # S5
        0.18 + W_IO + GAP + 5*(W_STG + GAP),    # Anomaly Label (I/O)
    ]

    # ── Boxes ─────────────────────────────────────────────────────────────────
    boxes = [
        (XS[0], W_IO,  "IoT Sensor\nStream",      "",                    True),
        (XS[1], W_STG, "Stage 1\nWindowing",       "W = 512 samples",     False),
        (XS[2], W_STG, "Stage 2\nFFT Extraction",  "Top-K |FFT| bins",    False),
        (XS[3], W_STG, "Stage 3\nStat. Fusion",    "μ, σ, γ₁, γ₂, RMS",  False),
        (XS[4], W_STG, "Stage 4\nSMOTE",           "ratio = 0.25",        False),
        (XS[5], W_STG, "Stage 5\nEnsemble Vote",   "RF + XGBoost",        False),
        (XS[6], W_IO,  "Anomaly\nLabel",            "",                    True),
    ]
    for x, w, title, sub, io in boxes:
        draw_box(ax, x, Y_BOX, w, H, title, sub, io)

    # ── Arrows ────────────────────────────────────────────────────────────────
    widths = [W_IO] + [W_STG] * 5 + [W_IO]
    for i in range(len(XS) - 1):
        draw_arrow(ax, XS[i] + widths[i], XS[i + 1], y=Y_BOX + H / 2)

    # ── Stage numbers ─────────────────────────────────────────────────────────
    for i, x in enumerate(XS[1:6], start=1):
        stage_label(ax, x + W_STG / 2, Y_BOX + H, i)

    # ── Feature dimension annotation ─────────────────────────────────────────
    x_ann = XS[3] + W_STG / 2
    ax.annotate(
        "φ ∈ ℝ⁴⁵",
        xy=(x_ann, Y_BOX), xytext=(x_ann, Y_BOX - 0.56),
        ha="center", fontsize=8.5, color="#7D3C98",
        arrowprops=dict(arrowstyle="-", color="#7D3C98", lw=1.0),
        zorder=6,
    )

    # ── Top rule + title ──────────────────────────────────────────────────────
    ax.axhline(y=3.40, xmin=0.01, xmax=0.99,
               color="#AAB7B8", lw=0.8, zorder=1)
    ax.text(6.75, 3.58,
            "Fig. 1.  ImFREQ-Lite Five-Stage Anomaly Detection Pipeline",
            ha="center", va="center",
            fontsize=9.8, fontweight="bold", color=C_LABEL)

    # ── Legend ────────────────────────────────────────────────────────────────
    io_p    = mpatches.Patch(facecolor=C_IO,    edgecolor=C_EDGE, lw=1.2,
                              label="Input / Output")
    stage_p = mpatches.Patch(facecolor=C_STAGE, edgecolor=C_EDGE, lw=1.2,
                              label="Processing Stage")
    ax.legend(handles=[io_p, stage_p],
              loc="lower right", bbox_to_anchor=(0.99, 0.03),
              fontsize=7.5, framealpha=0.9, edgecolor="#AAB7B8", ncol=2)

    # ── Save ─────────────────────────────────────────────────────────────────
    import os
    os.makedirs("figures", exist_ok=True)
    out = "figures/fig1_pipeline.png"
    plt.savefig(out, dpi=260, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
