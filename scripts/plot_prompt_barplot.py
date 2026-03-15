#!/usr/bin/env python3
"""
Generate 3 grouped bar plots of average fidelity vs prompt condition,
one figure per noise level (0.0 / 0.02 / 0.1).

Bar height  = avg_best_of_runs_fidelity
Black line  = avg_mean_fidelity (horizontal stroke across each bar)
X-axis      = prompt condition (Function Review / Physical Review / Normal Simple)
Bar groups  = 6 LLM models per prompt condition

Data source : ALL_IN_AVERAGE/average_over_all.csv
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
AVG_CSV = DATA_DIR / "ALL_IN_AVERAGE" / "average_over_all.csv"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "prompt_fidelity"

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

# Soft-pastel palette (NeurIPS-2025 style) + hatch for accessibility
COLORS = {
    "claude-sonnet": "#5B8DBE",
    "gemini-flash": "#F4A261",
    "gpt-5.1": "#7FB685",
    "gpt-oss": "#E76F51",
    "Qwen3": "#9B8EC5",
    "Qwen3.5": "#D4A5C9",
}
HATCHES = {
    "claude-sonnet": "",
    "gemini-flash": "///",
    "gpt-5.1": "...",
    "gpt-oss": "xxx",
    "Qwen3": "---",
    "Qwen3.5": "+++",
}


def make_bar_plot(df_noise: pd.DataFrame, noise_level: float) -> plt.Figure:
    """Draw one figure for the given noise_level subset."""
    models_present = [m for m in MODEL_ORDER if m in df_noise["model"].unique()]
    n_models = len(models_present)
    n_groups = len(PROMPT_MODE_ORDER)

    group_width = 0.72
    bar_w = group_width / n_models
    x_centers = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(7, 2.5), constrained_layout=False)
    fig.subplots_adjust(top=0.85, bottom=0.28, left=0.10, right=0.97)

    for j, model in enumerate(models_present):
        offsets = x_centers - group_width / 2 + bar_w * (j + 0.5)
        bests, means = [], []
        for prompt_mode in PROMPT_MODE_ORDER:
            row = df_noise[
                (df_noise["model"] == model) & (df_noise["prompt_mode"] == prompt_mode)
            ]
            if row.empty:
                bests.append(0.0)
                means.append(0.0)
            else:
                bests.append(float(row["avg_best_of_runs_fidelity"].values[0]))
                means.append(float(row["avg_mean_fidelity"].values[0]))

        ax.bar(
            offsets,
            bests,
            width=bar_w * 0.9,
            color=COLORS[model],
            edgecolor="black",
            linewidth=0.5,
            hatch=HATCHES[model],
            label=MODEL_DISPLAY[model],
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
    ax.set_xticklabels([PROMPT_MODE_LABELS[m] for m in PROMPT_MODE_ORDER], fontsize=8)
    ax.set_xlabel("Prompt Condition", fontsize=9)
    ax.set_ylabel("Fidelity", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_title(
        f"Prompt-Condition Performance — Noise Level {NOISE_LABELS[noise_level]}",
        fontweight="bold",
        fontsize=9,
        pad=4,
    )
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(axis="both", which="major", length=2, pad=2, labelsize=7)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # ---- legend (compact, below x-axis) ----
    bar_handles = [
        mpatches.Patch(
            facecolor=COLORS[m],
            edgecolor="black",
            linewidth=0.5,
            hatch=HATCHES[m],
            label=MODEL_DISPLAY[m],
        )
        for m in models_present
    ]
    mean_handle = Line2D([0], [0], color="black", linewidth=1.2, label="Mean fidelity")
    fig.legend(
        handles=bar_handles + [mean_handle],
        loc="lower center",
        ncol=7,
        bbox_to_anchor=(0.53, 0.01),
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
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    df = pd.read_csv(AVG_CSV)
    df["noise_level"] = df["noise_level"].astype(float)

    for noise in NOISE_LEVELS:
        df_sub = df[np.isclose(df["noise_level"], noise)].copy()
        if df_sub.empty:
            print(f"[WARNING] No data for noise_level={noise}, skipping.")
            continue

        fig = make_bar_plot(df_sub, noise)

        noise_str = str(noise).replace(".", "p")
        stem = f"prompt_barplot_noise{noise_str}"
        out_pdf = OUTPUT_DIR / f"{stem}.pdf"
        out_png = OUTPUT_DIR / f"{stem}.png"
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, bbox_inches="tight", dpi=200)
        print(f"Saved: {out_pdf}")
        print(f"Saved: {out_png}")
        plt.close(fig)


if __name__ == "__main__":
    main()
