#!/usr/bin/env python3
"""Generate a vertical validation-locked protocol schematic."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle


BLUE = "#dde4eb"
BLUE_EDGE = "#8d9cab"
RED = "#ff2b2b"
INK = "#1f2d3a"
MUTED = "#65707a"
YHAT_LABEL_DY = 0.040
CIRCLE_ARROW_START_DY = 0.045
RECT_ARROW_GAP = 0.006


def _block(ax, center, size, label, *, face=BLUE, edge=BLUE_EDGE, color=INK, lw=0.75, fontsize=10.2):
    x, y = center
    w, h = size
    ax.add_patch(
        Rectangle(
            (x - w / 2, y - h / 2),
            w,
            h,
            linewidth=lw,
            edgecolor=edge,
            facecolor=face,
        )
    )
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color=color, linespacing=0.95)
    return x, y, w, h


def _objective(ax, center, size, label):
    return _block(
        ax,
        center,
        size,
        label,
        face="white",
        edge=RED,
        color=INK,
        lw=1.05,
        fontsize=10.2,
    )


def _circle(ax, center, label):
    x, y = center
    ax.add_patch(Ellipse((x, y), width=0.148, height=0.076, linewidth=0.8, edgecolor="#28465a", facecolor="white"))
    ax.text(x, y - 0.001, label, ha="center", va="center", fontsize=10.2, color=INK)
    return x, y


def _arrow(ax, start, end, *, dashed=False, color="#27323a", mutation=8.0, lw=0.8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=lw,
            color=color,
            linestyle=(0, (3, 2)) if dashed else "solid",
            shrinkA=2,
            shrinkB=2,
        )
    )


def _caption(ax, y, label, title):
    ax.text(
        0.08,
        y,
        rf"$\bf{{{label}}}$ {title}",
        ha="left",
        va="center",
        fontsize=10.1,
        color="black",
    )


def _panel_guides(ax, y_mid):
    ax.plot([0.08, 0.92], [y_mid - 0.155, y_mid - 0.155], color="0.90", linewidth=0.55)


def _draw_training(ax, y_mid):
    rail_y = y_mid + 0.035
    data_y = y_mid - 0.095
    _caption(ax, y_mid + 0.130, "(a)", "Candidate Training")

    x = _circle(ax, (0.17, data_y), r"$x_{\mathrm{train}}$")
    y = _circle(ax, (0.78, data_y), r"$y_{\mathrm{train}}$")
    prep = _block(ax, (0.17, rail_y), (0.225, 0.082), "preprocess")
    model = _block(ax, (0.43, rail_y), (0.225, 0.082), r"model $f_\theta$")
    loss = _objective(ax, (0.78, rail_y), (0.245, 0.094), "train\nloss")

    _arrow(ax, (x[0], x[1] + CIRCLE_ARROW_START_DY), (prep[0], prep[1] - prep[3] / 2 - RECT_ARROW_GAP))
    _arrow(ax, (prep[0] + prep[2] / 2, rail_y), (model[0] - model[2] / 2, rail_y))
    _arrow(ax, (model[0] + model[2] / 2, rail_y), (loss[0] - loss[2] / 2, rail_y))
    ax.text(0.61, rail_y + 0.064, r"$z$", ha="center", va="center", fontsize=10.2, color=RED)
    _arrow(ax, (y[0], y[1] + CIRCLE_ARROW_START_DY), (loss[0], loss[1] - loss[3] / 2 - RECT_ARROW_GAP), dashed=True)
    _panel_guides(ax, y_mid)


def _draw_selection(ax, y_mid):
    rail_y = y_mid + 0.035
    data_y = y_mid - 0.095
    _caption(ax, y_mid + 0.130, "(b)", "Validation-Locked Selection")

    x = _circle(ax, (0.24, data_y), r"$x_{\mathrm{val}}$")
    y = _circle(ax, (0.76, data_y), r"$y_{\mathrm{val}}$")
    pipe = _block(ax, (0.24, rail_y), (0.300, 0.092), r"candidate $\mathcal{H}$" + "\npipeline")
    select = _objective(ax, (0.76, rail_y), (0.270, 0.090), r"select $h^*$")

    _arrow(ax, (x[0], x[1] + CIRCLE_ARROW_START_DY), (pipe[0], pipe[1] - pipe[3] / 2 - RECT_ARROW_GAP))
    _arrow(ax, (pipe[0] + pipe[2] / 2, rail_y), (select[0] - select[2] / 2, rail_y))
    ax.text(0.50, rail_y + YHAT_LABEL_DY, r"$\hat{y}_{\mathrm{val}}$", ha="center", va="center", fontsize=10.2, color=MUTED)
    _arrow(ax, (y[0], y[1] + CIRCLE_ARROW_START_DY), (select[0], select[1] - select[3] / 2 - RECT_ARROW_GAP), dashed=True)
    _panel_guides(ax, y_mid)


def _draw_reporting(ax, y_mid):
    rail_y = y_mid + 0.035
    data_y = y_mid - 0.095
    _caption(ax, y_mid + 0.130, "(c)", "Held-Out Reporting")

    x = _circle(ax, (0.24, data_y), r"$x_{\mathrm{test}}$")
    y = _circle(ax, (0.76, data_y), r"$y_{\mathrm{test}}$")
    pipe = _block(ax, (0.24, rail_y), (0.300, 0.092), r"frozen $h^*$" + "\npipeline")
    report = _objective(ax, (0.76, rail_y), (0.270, 0.094), "report\nmetrics")

    _arrow(ax, (x[0], x[1] + CIRCLE_ARROW_START_DY), (pipe[0], pipe[1] - pipe[3] / 2 - RECT_ARROW_GAP))
    _arrow(ax, (pipe[0] + pipe[2] / 2, rail_y), (report[0] - report[2] / 2, rail_y))
    ax.text(0.50, rail_y + YHAT_LABEL_DY, r"$\hat{y}_{\mathrm{test}}$", ha="center", va="center", fontsize=10.2, color=MUTED)
    _arrow(ax, (y[0], y[1] + CIRCLE_ARROW_START_DY), (report[0], report[1] - report[3] / 2 - RECT_ARROW_GAP), dashed=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
            "font.size": 10.2,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    ):
        fig, ax = plt.subplots(figsize=(3.50, 5.05))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        _draw_training(ax, 0.84)
        _draw_selection(ax, 0.50)
        _draw_reporting(ax, 0.16)

        _arrow(ax, (0.50, 0.338), (0.50, 0.303), color=RED, mutation=8.5, lw=0.9)
        ax.text(0.56, 0.318, "freeze", ha="left", va="center", fontsize=9.2, color=RED)

        fig.subplots_adjust(left=0.015, right=0.985, bottom=0.020, top=0.985)
        for ext in ("pdf", "png"):
            fig.savefig(args.out_dir / f"validation_locked_protocol.{ext}", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
