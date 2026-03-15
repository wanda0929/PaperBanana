#!/usr/bin/env python3
"""
Generate 6 line plots (2 metrics × 3 noise levels) of resource usage vs
prompt condition.

For each noise level (0.0 / 0.02 / 0.1):
  - Plot A: avg_best_run_tokens       → prompt_lineplot_tokens_noise<N>
  - Plot B: avg_best_run_k_reach_tol  → prompt_lineplot_iterations_noise<N>

Each plot shows:
  - Thin semi-transparent per-model lines (spread view)
  - Bold mean line (average over all 6 models)
  - Legend placed below the axes (no overlap with title)

Data source : ALL_IN_AVERAGE/average_over_all.csv
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
AVG_CSV = DATA_DIR / "ALL_IN_AVERAGE" / "average_over_all.csv"
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

MODEL_ORDER = [
    "claude-sonnet",
    "gemini-flash",
    "gpt-5.1",
    "gpt-oss",
    "Qwen3",
    "Qwen3.5",
]
MODEL_DISPLAY = {
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

X_TICKS = np.arange(len(PROMPT_MODE_ORDER))


# ---------------------------------------------------------------------------
# Generic single-metric line plot
# ---------------------------------------------------------------------------
def make_single_line_plot(
    df_noise: pd.DataFrame,
    noise_level: float,
    metric_col: str,
    y_label: str,
    line_color: str,
    title_suffix: str,
) -> plt.Figure:
    """
    One figure, one y-axis.
    Thin per-model lines + bold cross-model mean.
    Legend placed below the axes.
    """
    models_present = [m for m in MODEL_ORDER if m in df_noise["model"].unique()]

    fig, ax = plt.subplots(figsize=(7, 2.5), constrained_layout=False)
    fig.subplots_adjust(top=0.85, bottom=0.28, left=0.12, right=0.97)

    # ---- per-model thin traces ----
    for model in models_present:
        vals = []
        for pm in PROMPT_MODE_ORDER:
            row = df_noise[
                (df_noise["model"] == model) & (df_noise["prompt_mode"] == pm)
            ]
            vals.append(float(row[metric_col].values[0]) if not row.empty else np.nan)

        ax.plot(
            X_TICKS,
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
        df_noise[df_noise["prompt_mode"] == pm][metric_col].mean()
        for pm in PROMPT_MODE_ORDER
    ]
    ax.plot(
        X_TICKS,
        mean_vals,
        color=line_color,
        marker="o",
        markersize=8,
        linewidth=2.6,
        linestyle="-",
        zorder=5,
    )

    # ---- axes formatting ----
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels([PROMPT_MODE_LABELS[m] for m in PROMPT_MODE_ORDER], fontsize=8)
    ax.set_xlabel("Prompt Condition", fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(
        f"{title_suffix} — Noise Level {NOISE_LABELS[noise_level]}",
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

    # ---- legend below axes ----
    per_model_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[m],
            marker=MARKERS[m],
            markersize=5,
            linewidth=1.1,
            alpha=0.65,
            label=MODEL_DISPLAY[m],
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
def make_token_plot(df_noise: pd.DataFrame, noise_level: float) -> plt.Figure:
    return make_single_line_plot(
        df_noise,
        noise_level,
        metric_col="avg_best_run_tokens",
        y_label="Avg Best-Run Tokens",
        line_color="#2C6FAC",
        title_suffix="Token Usage vs Prompt Condition",
    )


def make_iteration_plot(df_noise: pd.DataFrame, noise_level: float) -> plt.Figure:
    return make_single_line_plot(
        df_noise,
        noise_level,
        metric_col="avg_best_run_k_reach_tol",
        y_label="Avg Best-Run Iterations to Reach Tolerance",
        line_color="#C0392B",
        title_suffix="Iterations to Tolerance vs Prompt Condition",
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

    for noise in NOISE_LEVELS:
        df_sub = df[np.isclose(df["noise_level"], noise)].copy()
        if df_sub.empty:
            print(f"[WARNING] No data for noise_level={noise}, skipping.")
            continue

        noise_str = str(noise).replace(".", "p")
        for make_func, stem_metric, subdir in [
            (make_token_plot, "tokens", "prompt_token"),
            (make_iteration_plot, "iterations", "prompt_iterations"),
        ]:
            OUTPUT_DIR = RESULTS_DIR / subdir
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            fig = make_func(df_sub, noise)
            stem = f"prompt_lineplot_{stem_metric}_noise{noise_str}"
            out_pdf = OUTPUT_DIR / f"{stem}.pdf"
            out_png = OUTPUT_DIR / f"{stem}.png"
            fig.savefig(out_pdf, bbox_inches="tight")
            fig.savefig(out_png, bbox_inches="tight", dpi=200)
            print(f"Saved: {out_pdf}")
            print(f"Saved: {out_png}")
            plt.close(fig)


if __name__ == "__main__":
    main()
