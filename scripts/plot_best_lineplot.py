#!/usr/bin/env python3
"""
Generate 6 line plots (2 metrics × 3 noise levels) of resource usage per task.

For each noise level (0.0 / 0.02 / 0.1):
  - Plot A: best_run_tokens       → best_lineplot_tokens_noise<N>
  - Plot B: best_run_k_reach_tol  → best_lineplot_iterations_noise<N>

Each plot shows 3 lines, one per prompt condition.
X-axis = task name (13 tasks).
Markers at every data point; legend placed below the axes.

Data source : ALL_IN_AVERAGE/Best_performance.csv
Reference   : /Users/wanda/QSTC PROJECT/PaperBanana  (NeurIPS-2025 style guide)
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = (
    pathlib.Path(__file__).resolve().parents[1].parent
    / "quctrl-batch-experiments"
    / "data"
)
BEST_CSV = DATA_DIR / "ALL_IN_AVERAGE" / "Best_performance.csv"
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"

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

TASK_DISPLAY = {
    "cd_driving": "CD-Drive",
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
    "symplectic_oscillator": "Sympl-Osc",
    "transmon_xgate": "Transmon-X",
    "two_spin_transfer": "2-Spin",
}

TASK_ORDER = sorted(TASK_DISPLAY.keys())
X_TICKS = np.arange(len(TASK_ORDER))

# Colour + marker per prompt condition
COLORS = {
    "function": "#2C6FAC",
    "normal_physical": "#E07B39",
    "normal_simple": "#3A8C5C",
}
MARKERS = {
    "function": "o",
    "normal_physical": "s",
    "normal_simple": "^",
}


# ---------------------------------------------------------------------------
# Generic single-metric line plot
# ---------------------------------------------------------------------------
def make_single_line_plot(
    df_noise: pd.DataFrame,
    noise_level: float,
    metric_col: str,
    y_label: str,
    title_suffix: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 2.5), constrained_layout=False)
    fig.subplots_adjust(top=0.85, bottom=0.30, left=0.10, right=0.97)

    for pm in PROMPT_MODE_ORDER:
        vals = []
        for task in TASK_ORDER:
            row = df_noise[(df_noise["task"] == task) & (df_noise["prompt_mode"] == pm)]
            vals.append(float(row[metric_col].values[0]) if not row.empty else np.nan)

        ax.plot(
            X_TICKS,
            vals,
            color=COLORS[pm],
            marker=MARKERS[pm],
            markersize=6,
            linewidth=1.8,
            linestyle="-",
            zorder=3,
            label=PROMPT_MODE_LABELS[pm],
        )

    # ---- axes formatting ----
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(
        [TASK_DISPLAY.get(t, t) for t in TASK_ORDER],
        fontsize=5.5,
        ha="right",
        rotation=40,
    )
    ax.set_xlabel("Task", fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(
        f"{title_suffix} — Noise {NOISE_LABELS[noise_level]}",
        fontweight="bold",
        fontsize=9,
        pad=4,
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    ax.tick_params(axis="both", which="major", length=2, pad=2, labelsize=6)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # ---- legend (compact, below x-axis) ----
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[pm],
            marker=MARKERS[pm],
            markersize=4,
            linewidth=1.4,
            label=PROMPT_MODE_LABELS[pm],
        )
        for pm in PROMPT_MODE_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.53, 0.01),
        frameon=False,
        fontsize=6,
        handlelength=1.5,
        columnspacing=1.0,
    )

    return fig


def make_token_plot(df_noise: pd.DataFrame, noise_level: float) -> plt.Figure:
    return make_single_line_plot(
        df_noise,
        noise_level,
        metric_col="best_run_tokens",
        y_label="Best-Run Tokens",
        title_suffix="Token Usage vs Task",
    )


def make_iteration_plot(df_noise: pd.DataFrame, noise_level: float) -> plt.Figure:
    return make_single_line_plot(
        df_noise,
        noise_level,
        metric_col="best_run_k_reach_tol",
        y_label="Best-Run Iterations to Reach Tolerance",
        title_suffix="Iterations to Tolerance vs Task",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
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

        noise_str = str(noise).replace(".", "p")
        for make_func, stem_metric, subdir in [
            (make_token_plot, "tokens", "best_tokens_noise"),
            (make_iteration_plot, "iterations", "best_iterations_noise"),
        ]:
            OUTPUT_DIR = RESULTS_DIR / subdir
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            fig = make_func(df_sub, noise)
            stem = f"best_lineplot_{stem_metric}_noise{noise_str}"
            for ext in ("pdf", "png"):
                out = OUTPUT_DIR / f"{stem}.{ext}"
                fig.savefig(out, bbox_inches="tight", dpi=200 if ext == "png" else None)
                print(f"Saved: {out}")
            plt.close(fig)


if __name__ == "__main__":
    main()
