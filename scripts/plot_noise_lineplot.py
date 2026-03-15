#!/usr/bin/env python3
"""
Generate 6 line plots (2 metrics × 3 prompt conditions) of resource usage vs
noise level.

For each prompt condition (function / normal_physical / normal_simple):
  - Plot A: avg_best_run_tokens       vs noise level  → noise_lineplot_tokens_<mode>
  - Plot B: avg_best_run_k_reach_tol  vs noise level  → noise_lineplot_iterations_<mode>

Each plot shows:
  - Thin semi-transparent per-model lines (spread / per-model view)
  - Bold mean line (average over all 6 models)
  - Legend placed below the axes (no overlap with title)

Follows the PaperBanana NeurIPS-2025 style guide.
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
AVG_CSV = DATA_DIR / "ALL_IN_AVERAGE" / "average_over_all.csv"
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NOISE_LEVELS = [0.0, 0.02, 0.1]
NOISE_LABELS = ["0.0", "0.02", "0.1"]

PROMPT_MODE_ORDER = ["function", "normal_physical", "normal_simple"]
PROMPT_MODE_TITLES = {
    "function": "Function Review",
    "normal_physical": "Physical Review",
    "normal_simple": "Normal Simple",
}

MODEL_ORDER = [
    "claude-sonnet",
    "gemini-flash",
    "gpt-5.1",
    "gpt-oss",
    "Qwen3",
    "Qwen3.5",
]
MODEL_LABELS = {
    "claude-sonnet": "Claude Sonnet",
    "gemini-flash": "Gemini Flash",
    "gpt-5.1": "GPT-5.1",
    "gpt-oss": "GPT-OSS",
    "Qwen3": "Qwen3",
    "Qwen3.5": "Qwen3.5",
}

COLORS = {
    "claude-sonnet": "#5B8DBE",
    "gemini-flash": "#F4A261",
    "gpt-5.1": "#7FB685",
    "gpt-oss": "#E76F51",
    "Qwen3": "#9B8EC5",
    "Qwen3.5": "#D4A5C9",
}
MARKERS = {
    "claude-sonnet": "o",
    "gemini-flash": "s",
    "gpt-5.1": "^",
    "gpt-oss": "D",
    "Qwen3": "P",
    "Qwen3.5": "X",
}

MEAN_COLOR = "#1A1A2E"  # near-black for the cross-model mean line


# ---------------------------------------------------------------------------
# Generic single-metric line plot
# ---------------------------------------------------------------------------
def make_single_line_plot(
    df_prompt: pd.DataFrame,
    prompt_mode: str,
    metric_col: str,
    y_label: str,
    line_color: str,
    title_suffix: str,
) -> plt.Figure:
    """
    One figure, one y-axis.
    Thin per-model lines + bold cross-model mean.
    Legend is placed below the plot to avoid overlapping the title.
    """
    models_present = [m for m in MODEL_ORDER if m in df_prompt["model"].unique()]
    x_ticks = np.arange(len(NOISE_LEVELS))

    fig, ax = plt.subplots(figsize=(7, 2.5), constrained_layout=False)
    fig.subplots_adjust(top=0.85, bottom=0.28, left=0.12, right=0.97)

    # ---- per-model thin traces ----
    for model in models_present:
        vals = []
        for noise in NOISE_LEVELS:
            row = df_prompt[
                (df_prompt["model"] == model)
                & (np.isclose(df_prompt["noise_level"], noise))
            ]
            vals.append(float(row[metric_col].values[0]) if not row.empty else np.nan)

        ax.plot(
            x_ticks,
            vals,
            color=COLORS[model],
            marker=MARKERS[model],
            markersize=5,
            linewidth=1.1,
            alpha=0.50,
            linestyle="-",
            zorder=2,
        )

    # ---- cross-model mean (bold) ----
    mean_vals = [
        df_prompt[np.isclose(df_prompt["noise_level"], noise)][metric_col].mean()
        for noise in NOISE_LEVELS
    ]
    ax.plot(
        x_ticks,
        mean_vals,
        color=line_color,
        marker="o",
        markersize=8,
        linewidth=2.6,
        linestyle="-",
        zorder=5,
    )

    # ---- axes formatting ----
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(NOISE_LABELS, fontsize=8)
    ax.set_xlabel("Noise Level", fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(
        f"{title_suffix} — {PROMPT_MODE_TITLES[prompt_mode]}",
        fontweight="bold",
        fontsize=9,
        pad=4,
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    ax.tick_params(axis="both", which="major", length=2, pad=2, labelsize=7)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # ---- legend (below axes, two rows) ----
    per_model_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[m],
            marker=MARKERS[m],
            markersize=5,
            linewidth=1.1,
            alpha=0.65,
            label=MODEL_LABELS[m],
        )
        for m in models_present
    ]
    mean_handle = Line2D(
        [0],
        [0],
        color=line_color,
        marker="o",
        markersize=7,
        linewidth=2.4,
        label="Mean (all models)",
    )

    fig.legend(
        handles=per_model_handles + [mean_handle],
        loc="lower center",
        ncol=7,
        bbox_to_anchor=(0.54, 0.01),
        frameon=False,
        fontsize=6,
        handlelength=1.2,
        columnspacing=0.8,
    )

    return fig


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def make_token_plot(df_prompt: pd.DataFrame, prompt_mode: str) -> plt.Figure:
    return make_single_line_plot(
        df_prompt,
        prompt_mode,
        metric_col="avg_best_run_tokens",
        y_label="Avg Best-Run Tokens",
        line_color="#2C6FAC",
        title_suffix="Token Usage vs Noise Level",
    )


def make_iteration_plot(df_prompt: pd.DataFrame, prompt_mode: str) -> plt.Figure:
    return make_single_line_plot(
        df_prompt,
        prompt_mode,
        metric_col="avg_best_run_k_reach_tol",
        y_label="Avg Best-Run Iterations to Reach Tolerance",
        line_color="#C0392B",
        title_suffix="Iterations to Tolerance vs Noise Level",
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
            "xtick.labelsize": 8,
            "ytick.labelsize": 7,
        }
    )

    df = pd.read_csv(AVG_CSV)
    df["noise_level"] = df["noise_level"].astype(float)

    for prompt_mode in PROMPT_MODE_ORDER:
        df_sub = df[df["prompt_mode"] == prompt_mode].copy()
        if df_sub.empty:
            print(f"[WARNING] No data for prompt_mode='{prompt_mode}', skipping.")
            continue

        for make_func, stem_metric, subdir in [
            (make_token_plot, "tokens", "noise_level_token"),
            (make_iteration_plot, "iterations", "noise_level_iterations"),
        ]:
            OUTPUT_DIR = RESULTS_DIR / subdir
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            fig = make_func(df_sub, prompt_mode)
            stem = f"noise_lineplot_{stem_metric}_{prompt_mode}"
            out_pdf = OUTPUT_DIR / f"{stem}.pdf"
            out_png = OUTPUT_DIR / f"{stem}.png"
            fig.savefig(out_pdf, bbox_inches="tight")
            fig.savefig(out_png, bbox_inches="tight", dpi=200)
            print(f"Saved: {out_pdf}")
            print(f"Saved: {out_png}")
            plt.close(fig)


if __name__ == "__main__":
    main()
