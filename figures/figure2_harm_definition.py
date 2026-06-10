#!/usr/bin/env python3
"""
Generate Figure 2: Harm definition diagram.

This figure illustrates the harm definition from the paper Section II-C.
It shows:
- Environment Y_S (agent with memory m executes, produces bad action)
- Environment Y_S' (agent without memory m executes, produces good action)
- Causal contrast: Delta = harm(Y_S) - harm(Y_S')

The diagram uses boxes, arrows, and text annotations to convey the
causal inference structure of the harm definition.
"""

import argparse
import os
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Require matplotlib >= 3.0 for patching compatibility
assert matplotlib.__version__ >= "3.0", \
    f"matplotlib version {matplotlib.__version__} < 3.0; upgrade required"

# Output defaults
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "results",
    "figures",
    "figure2.pdf",
)


def draw_box(ax, x, y, width, height, label, color, fontsize=10):
    """Draw a rounded rectangle box with a text label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x, y : float
        Bottom-left corner coordinates.
    width, height : float
        Box dimensions.
    label : str
        Text to display inside the box.
    color : str
        Fill color name or hex string.
    fontsize : int
        Font size for the label text.
    """
    box = mpatches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        wrap=True,
    )


def draw_arrow(ax, x1, y1, x2, y2, label="", fontsize=9, color="black", label_offset=(0, 0.08)):
    """Draw an arrow from (x1, y1) to (x2, y2) with an optional label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x1, y1 : float
        Arrow tail coordinates.
    x2, y2 : float
        Arrow head coordinates.
    label : str
        Optional text placed at the midpoint of the arrow.
    fontsize : int
        Font size for the label.
    color : str
        Arrow color.
    label_offset : tuple
        (dx, dy) offset for label from midpoint.
    """
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=2),
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mx + label_offset[0],
            my + label_offset[1],
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
            style="italic",
        )


def generate_figure2(output_path: str):
    """Generate Figure 2: Harm definition diagram.

    Parameters
    ----------
    output_path : str
        Path where the PDF will be saved.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # ------------------------------------------------------------------
    # Layout constants (all in data coordinates)
    # ------------------------------------------------------------------
    x_left = 0.5    # left edge of Y_S zone
    x_center = 5.5  # center of Delta zone
    x_right = 10.5  # left edge of Y_S' zone
    zone_width = 4.5

    # Colors
    color_memory = "#FFE0B2"     # orange-tinted, memory present
    color_nomemory = "#C8E6C9"   # green-tinted, memory absent
    color_bad = "#FFCDD2"        # red-tinted, bad outcome
    color_good = "#C8E6C9"      # green-tinted, good outcome
    color_delta = "#BBDEFB"      # blue-tinted, causal contrast

    # ------------------------------------------------------------------
    # Zone 1: Environment Y_S (with memory m)
    # ------------------------------------------------------------------
    ax.text(x_left + zone_width / 2, 0.88,
            r"$\mathcal{Y}_S$ (with memory $m$)",
            ha="center", va="center", fontsize=13, fontweight="bold")

    # Environment box
    draw_box(ax, x_left + 0.2, 0.72, zone_width - 0.4, 0.10,
             "Environment\n(memories include m)",
             color_memory, fontsize=9)

    # Agent executes (with memory)
    draw_box(ax, x_left + 0.6, 0.58, zone_width - 1.2, 0.10,
             "Agent (reads memory m)",
             color_memory, fontsize=9)

    # Arrow: environment -> agent
    draw_arrow(ax, x_left + zone_width / 2, 0.72,
               x_left + zone_width / 2, 0.68,
               label="observe", label_offset=(0.35, 0))

    # Action produced
    draw_box(ax, x_left + 0.6, 0.44, zone_width - 1.2, 0.10,
             "Action: uses m",
             color_memory, fontsize=9)

    # Arrow: agent -> action
    draw_arrow(ax, x_left + zone_width / 2, 0.58,
               x_left + zone_width / 2, 0.54,
               label="act", label_offset=(0.25, 0))

    # Outcome: BAD
    draw_box(ax, x_left + 0.6, 0.30, zone_width - 1.2, 0.10,
             "Outcome: BAD  (X)",
             color_bad, fontsize=9)

    # Arrow: action -> outcome
    draw_arrow(ax, x_left + zone_width / 2, 0.44,
               x_left + zone_width / 2, 0.40,
               label="leads to", label_offset=(0.40, 0))

    # Harm value
    ax.text(x_left + zone_width / 2, 0.20,
            r"$\mathrm{harm}(\mathcal{Y}_S) = 1$",
            ha="center", va="center",
            fontsize=11, bbox=dict(boxstyle="round", facecolor=color_bad, alpha=0.5))

    # Vertical brace on the left zone
    brace_x = x_left - 0.20
    ax.plot([brace_x, brace_x], [0.30, 0.82], color="black", lw=1.5)
    ax.plot([brace_x - 0.05, brace_x], [0.30, 0.30], color="black", lw=1.5)
    ax.plot([brace_x - 0.05, brace_x], [0.82, 0.82], color="black", lw=1.5)
    ax.text(brace_x - 0.15, 0.56, "Treatment\n(memory m present)",
            ha="center", va="center", fontsize=8, rotation=90)

    # ------------------------------------------------------------------
    # Zone 2: Causal contrast Delta
    # ------------------------------------------------------------------
    # Delta box in the center
    draw_box(ax, x_center - 1.2, 0.50, 2.4, 0.14,
             r"$\Delta = \mathrm{harm}(\mathcal{Y}_S) - \mathrm{harm}(\mathcal{Y}_{S'})$",
             color_delta, fontsize=11)

    # Arrow from Y_S harm UP to Delta (pointing from harm to delta)
    # Tail at harm value (below), head at Delta (center)
    ax.annotate("",
               xy=(x_center - 0.6, 0.50),
               xytext=(x_left + zone_width / 2, 0.24),
               arrowprops=dict(arrowstyle="->", color="blue", lw=1.5,
                              connectionstyle="arc3,rad=0.15"))
    # Label placed below the outcome box to avoid overlap
    ax.text(3.5, 0.26,
            r"$\mathrm{harm}(\mathcal{Y}_S)$",
            ha="center", fontsize=8, color="blue")

    # Arrow from Y_S' harm UP to Delta (pointing from harm to delta)
    ax.annotate("",
               xy=(x_center + 0.6, 0.50),
               xytext=(x_right + zone_width / 2, 0.24),
               arrowprops=dict(arrowstyle="->", color="blue", lw=1.5,
                              connectionstyle="arc3,rad=-0.15"))
    # Label placed below the outcome box to avoid overlap
    ax.text(9.5, 0.26,
            r"$\mathrm{harm}(\mathcal{Y}_{S'})$",
            ha="center", fontsize=8, color="blue")

    # Result arrow (Delta > 0) pointing downward from Delta
    ax.annotate("",
               xy=(x_center, 0.44),
               xytext=(x_center, 0.50),
               arrowprops=dict(arrowstyle="->", color="red", lw=2))
    # Text below the arrow, positioned to not overlap with outcome boxes
    ax.text(x_center, 0.36,
            r"$\Delta > 0$: memory $m$ causes harm",
            ha="center", va="top", fontsize=9, color="red")

    # ------------------------------------------------------------------
    # Zone 3: Environment Y_S' (without memory m)
    # ------------------------------------------------------------------
    ax.text(x_right + zone_width / 2, 0.88,
            r"$\mathcal{Y}_{S'}$ (without memory $m$)",
            ha="center", va="center", fontsize=13, fontweight="bold")

    # Environment box (memory m deleted)
    draw_box(ax, x_right + 0.2, 0.72, zone_width - 0.4, 0.10,
             r"Environment $\mathcal{Y}_{S'}$" "\n(memories: S \\ {m})",
             color_nomemory, fontsize=9)

    # Agent executes (without memory)
    draw_box(ax, x_right + 0.6, 0.58, zone_width - 1.2, 0.10,
             "Agent (no memory m)",
             color_nomemory, fontsize=9)

    # Arrow: environment -> agent
    draw_arrow(ax, x_right + zone_width / 2, 0.72,
               x_right + zone_width / 2, 0.68,
               label="observe", label_offset=(0.35, 0))

    # Action produced
    draw_box(ax, x_right + 0.6, 0.44, zone_width - 1.2, 0.10,
             "Action: no m",
             color_nomemory, fontsize=9)

    # Arrow: agent -> action
    draw_arrow(ax, x_right + zone_width / 2, 0.58,
               x_right + zone_width / 2, 0.54,
               label="act", label_offset=(0.25, 0))

    # Outcome: GOOD
    draw_box(ax, x_right + 0.6, 0.30, zone_width - 1.2, 0.10,
             "Outcome: GOOD  (V)",
             color_good, fontsize=9)

    # Arrow: action -> outcome
    draw_arrow(ax, x_right + zone_width / 2, 0.44,
               x_right + zone_width / 2, 0.40,
               label="leads to", label_offset=(0.40, 0))

    # Harm value (0, because outcome is good)
    ax.text(x_right + zone_width / 2, 0.20,
            r"$\mathrm{harm}(\mathcal{Y}_{S'}) = 0$",
            ha="center", va="center",
            fontsize=11, bbox=dict(boxstyle="round", facecolor=color_good, alpha=0.5))

    # Vertical brace on the right zone
    brace_x_r = x_right + zone_width + 0.20
    ax.plot([brace_x_r, brace_x_r], [0.30, 0.82], color="black", lw=1.5)
    ax.plot([brace_x_r, brace_x_r + 0.05], [0.30, 0.30], color="black", lw=1.5)
    ax.plot([brace_x_r, brace_x_r + 0.05], [0.82, 0.82], color="black", lw=1.5)
    ax.text(brace_x_r + 0.20, 0.56, "Control\n(memory m deleted)",
            ha="center", va="center", fontsize=8, rotation=-90)

    # ------------------------------------------------------------------
    # Bottom annotation: definition text
    # ------------------------------------------------------------------
    ax.text(5.5, 0.06,
            r"Definition: $\mathrm{harm}(\mathcal{Y}_S) = \mathbb{1}[\mathrm{action\ is\ bad\ in\ }\mathcal{Y}_S]$;  "
            r"$\Delta = \mathrm{harm}(\mathcal{Y}_S) - \mathrm{harm}(\mathcal{Y}_{S'})$ "
            r"measures causal effect of memory $m$ on harm.",
            ha="center", va="bottom",
            fontsize=9, style="italic",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"))

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure 2 saved to {output_path}")
    plt.close(fig)


def main():
    """Parse CLI arguments and dispatch to figure generator."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 2: Harm definition diagram"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output PDF file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock data (no-op for this conceptual figure)",
    )
    args = parser.parse_args()

    # Fast-fail: verify output directory is writable
    output_dir = os.path.dirname(os.path.abspath(args.output))
    assert output_dir, f"Cannot determine output directory from path: {args.output!r}"
    try:
        os.makedirs(output_dir, exist_ok=True)
        test_path = os.path.join(output_dir, ".write_test")
        with open(test_path, "w") as f:
            f.write("test")
        os.remove(test_path)
    except OSError as exc:
        raise OSError(
            f"Output directory {output_dir!r} is not writable: {exc}"
        ) from exc

    if args.use_mock:
        print("Note: --use-mock flag is accepted but not required for Figure 2 "
              "(conceptual diagram, no data dependency).")

    generate_figure2(args.output)


if __name__ == "__main__":
    main()
