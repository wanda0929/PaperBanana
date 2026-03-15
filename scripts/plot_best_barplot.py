#!/usr/bin/env python3
"""
Generate 3 grouped bar plots of best fidelity per task,
one figure per noise level (0.0 / 0.02 / 0.1).

For each task the best-performing LLM model is already resolved in
Best_performance.csv (one row per task × noise × prompt_mode).

Bar height  = best_of_runs_fidelity   (of the best model)
Black line  = mean_fidelity           (of the best model)
X-axis      = task name  (13 tasks)
Bar groups  = 3 prompt conditions per task

Data source : ALL_IN_AVERAGE/Best_performance.csv
Reference   : /Users/wanda/QSTC PROJECT/PaperBanana  (NeurIPS-2025 style guide)
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = (
    pathlib.Path(__file__).resolve().parents[1].parent
    / "quctrl-batch-experiments"
    / "data"
)
BEST_CSV = DATA_DIR / "ALL_IN_AVERAGE" / "Best_performance.csv"
OUTPUT_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "results" / "best_fidelity_noise"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NOISE_LEVELS = [0.0, 0.02, 0.1]
NOISE_LABELS = {0.0: "0.0", 0.02: "0.02", 0.1: "0.1"}

PROMPT_MODE_ORDER = ["function", "normal_physical", "normal_simple"]
PROMPT_MODE_LABELS = {
    "function": "Function Review",
    "normal_physical": "Physical Review",
    "normal_simple": "Normal Simple",
}

# Short x-axis labels for the 13 tasks (single-line for compact layout)
TASK_DISPLAY = {
    "control_crab_2qubit_interaction": "CRAB-2Q",
    "control_grape_cphase": "GRAPE-CP",
    "control_grape_toffoli": "GRAPE-Tof",
    "crab_qft": "CRAB-QFT",
    "decoherence_suppression": "Decoh.",
    "dicke": "Dicke",
    "drag_pulse": "DRAG",
    "lambda_transfer": "Lambda",
    "landau_zener": "L-Zener",
    "lindbladian": "Lindbl.",
    "qutip_single_qubit": "QuTiP-SQ",
    "single_qubit_gate": "SQG",
    "two_spin_transfer": "2-Spin",
}

# Colour per prompt condition (NeurIPS-2025 soft pastels)
COLORS = {
    "function": "#5B8DBE",
    "normal_physical": "#F4A261",
    "normal_simple": "#7FB685",
}
HATCHES = {
    "function": "",
    "normal_physical": "///",
    "normal_simple": "...",
}

# Canonical task order (alphabetical for consistency)
TASK_ORDER = sorted(TASK_DISPLAY.keys())


def make_bar_plot(df_noise: pd.DataFrame, noise_level: float) -> plt.Figure:
    """Draw one figure for the given noise_level subset."""
    n_tasks = len(TASK_ORDER)
    n_groups = len(PROMPT_MODE_ORDER)

    group_width = 0.70
    bar_w = group_width / n_groups
    x_centers = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(7, 2.5), constrained_layout=False)
    fig.subplots_adjust(top=0.85, bottom=0.30, left=0.08, right=0.97)

    for j, pm in enumerate(PROMPT_MODE_ORDER):
        offsets = x_centers - group_width / 2 + bar_w * (j + 0.5)
        bests, means = [], []

        for task in TASK_ORDER:
            row = df_noise[(df_noise["task"] == task) & (df_noise["prompt_mode"] == pm)]
            if row.empty:
                bests.append(0.0)
                means.append(0.0)
            else:
                bests.append(float(row["best_of_runs_fidelity"].values[0]))
                means.append(float(row["mean_fidelity"].values[0]))

        ax.bar(
            offsets,
            bests,
            width=bar_w * 0.9,
            color=COLORS[pm],
            edgecolor="black",
            linewidth=0.5,
            hatch=HATCHES[pm],
            label=PROMPT_MODE_LABELS[pm],
            zorder=3,
        )

        half = bar_w * 0.9 / 2
        for x_pos, mean_val in zip(offsets, means):
            ax.hlines(
                mean_val,
                x_pos - half,
                x_pos + half,
                colors="black",
                linewidths=1.2,
                zorder=4,
            )

    # ---- axes formatting ----
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [TASK_DISPLAY.get(t, t) for t in TASK_ORDER],
        fontsize=5.5,
        ha="right",
        rotation=40,
    )
    ax.set_xlabel("Task", fontsize=9)
    ax.set_ylabel("Best Fidelity", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.5, n_tasks - 0.5)
    ax.set_title(
        f"Best-Model Fidelity per Task — Noise {NOISE_LABELS[noise_level]}",
        fontweight="bold",
        fontsize=9,
        pad=4,
    )
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(axis="both", which="major", length=2, pad=2, labelsize=6)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # ---- legend (compact, below x-axis) ----
    bar_handles = [
        mpatches.Patch(
            facecolor=COLORS[pm],
            edgecolor="black",
            linewidth=0.5,
            hatch=HATCHES[pm],
            label=PROMPT_MODE_LABELS[pm],
        )
        for pm in PROMPT_MODE_ORDER
    ]
    mean_handle = Line2D([0], [0], color="black", linewidth=1.2, label="Mean fidelity")
    fig.legend(
        handles=bar_handles + [mean_handle],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.52, 0.01),
        frameon=False,
        fontsize=6,
        handlelength=1.2,
        handleheight=0.6,
        columnspacing=0.8,
    )

    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 6,
            "ytick.labelsize": 7,
        }
    )

    df = pd.read_csv(BEST_CSV)
    df["noise_level"] = df["noise_level"].astype(float)

    for noise in NOISE_LEVELS:
        df_sub = df[np.isclose(df["noise_level"], noise)].copy()
        if df_sub.empty:
            print(f"[WARNING] No data for noise_level={noise}, skipping.")
            continue

        fig = make_bar_plot(df_sub, noise)
        noise_str = str(noise).replace(".", "p")
        stem = f"best_barplot_noise{noise_str}"
        for ext in ("pdf", "png"):
            out = OUTPUT_DIR / f"{stem}.{ext}"
            fig.savefig(out, bbox_inches="tight", dpi=200 if ext == "png" else None)
            print(f"Saved: {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
