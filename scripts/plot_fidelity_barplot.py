#!/usr/bin/env python3
"""
Generate grouped bar plots of best/mean fidelity per task,
grouped by prompt_mode on the x-axis (noise_level=0 only).
Follows PaperBanana NeurIPS 2025 style guide.
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_DIR = (
    pathlib.Path(__file__).resolve().parents[1].parent
    / "quctrl-batch-experiments"
    / "data"
)
OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"

TASKS = {
    "CRAB 2-Qubit Interaction": DATA_DIR
    / "control_crab_2qubit_interaction"
    / "final_agg_by_noise_model.csv",
    "GRAPE CPhase Gate": DATA_DIR
    / "control_grape_cphase"
    / "final_agg_by_noise_model.csv",
    "GRAPE Toffoli Gate": DATA_DIR
    / "control_grape_toffoli"
    / "final_agg_by_noise_model_control_grape_toffoli.csv",
    "CRAB QFT": DATA_DIR / "crab_qft" / "final_agg_by_noise_model_crab_qft.csv",
    "Decoherence Suppression": DATA_DIR
    / "decoherence_suppression"
    / "final_agg_by_noise_model.csv",
    "Dicke State Preparation": DATA_DIR
    / "Dicke"
    / "final_agg_by_noise_model_dicke.csv",
    "DRAG Pulse": DATA_DIR / "drag_pulse" / "final_agg_by_noise_model_drag_pulse.csv",
    "Lambda Transfer": DATA_DIR
    / "lambda_transfer"
    / "final_agg_by_noise_model_lambda_transfer.csv",
    "Landau-Zener": DATA_DIR
    / "landau_zener_csv_datas"
    / "final_agg_by_noise_model.csv",
    "Lindbladian": DATA_DIR / "lindbladian" / "final_agg_by_noise_model.csv",
    "QuTiP Single Qubit": DATA_DIR
    / "qutip_single_qubit"
    / "final_agg_by_noise_model_qutip_single_qubit.csv",
    "Single Qubit Gate": DATA_DIR
    / "Single_qubit_gate"
    / "final_agg_by_noise_model_single_qubit_gate.csv",
    "Two-Spin Transfer": DATA_DIR
    / "two_spin_transfer"
    / "final_agg_by_noise_model_two_spin_transfer.csv",
}

PROMPT_MODE_ORDER = ["function", "normal_physical", "normal_simple"]
PROMPT_MODE_LABELS = {
    "function": "Function",
    "normal_physical": "Normal Physical",
    "normal_simple": "Normal Simple",
}

MODEL_UNIFY = {
    "claude-sonnet": "Claude Sonnet",
    "gemini-flash": "Gemini Flash",
    "gpt-5.1": "GPT-5.1",
    "gpt-oss-120b": "GPT-OSS",
    "gpt-oss": "GPT-OSS",
    "qwen3": "Qwen3",
    "qwen3-30b": "Qwen3",
    "qwen3.5": "Qwen3.5",
    "qwen3.5-plus": "Qwen3.5",
}

MODEL_ORDER = [
    "Claude Sonnet",
    "Gemini Flash",
    "GPT-5.1",
    "GPT-OSS",
    "Qwen3",
    "Qwen3.5",
]

COLORS = {
    "Claude Sonnet": "#5B8DBE",
    "Gemini Flash": "#F4A261",
    "GPT-5.1": "#7FB685",
    "GPT-OSS": "#E76F51",
    "Qwen3": "#9B8EC5",
    "Qwen3.5": "#D4A5C9",
}


def load_task_data(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["noise_level"] = df["noise_level"].astype(float)
    df = df[df["noise_level"] == 0.0].copy()
    df["model_label"] = df["model"].map(MODEL_UNIFY)
    return df


def plot_task(ax: plt.Axes, df: pd.DataFrame, task_name: str):
    modes = [m for m in PROMPT_MODE_ORDER if m in df["prompt_mode"].unique()]
    models = [m for m in MODEL_ORDER if m in df["model_label"].unique()]
    n_modes = len(modes)
    n_models = len(models)

    group_width = 0.7
    bar_w = group_width / n_models
    x_centers = np.arange(n_modes)

    for j, model in enumerate(models):
        offsets = x_centers - group_width / 2 + bar_w * (j + 0.5)
        bests, means = [], []
        for mode in modes:
            row = df[(df["prompt_mode"] == mode) & (df["model_label"] == model)]
            if row.empty:
                bests.append(0)
                means.append(0)
            else:
                bests.append(row["best_of_runs_fidelity"].values[0])
                means.append(row["mean_fidelity"].values[0])

        ax.bar(
            offsets,
            bests,
            width=bar_w * 0.9,
            color=COLORS[model],
            edgecolor="black",
            linewidth=0.4,
            label=model,
            zorder=3,
        )
        for x_pos, mean_val in zip(offsets, means):
            ax.hlines(
                mean_val,
                x_pos - bar_w * 0.9 / 2,
                x_pos + bar_w * 0.9 / 2,
                colors="black",
                linewidths=0.8,
                zorder=4,
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels([PROMPT_MODE_LABELS[m] for m in modes])
    ax.set_ylabel("Fidelity", labelpad=2)
    ax.set_ylim(0, 1.05)
    ax.set_title(task_name, fontweight="bold", fontsize=8, pad=2)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.tick_params(axis="both", length=2, pad=2)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
        }
    )

    fig, axes = plt.subplots(len(TASKS), 1, figsize=(7, 11), constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.03, h_pad=0.02, w_pad=0.02)

    all_models_in_fig = set()
    for idx, (task_name, csv_path) in enumerate(TASKS.items()):
        df = load_task_data(csv_path)
        all_models_in_fig.update(df["model_label"].unique())
        plot_task(axes[idx], df, task_name)

    legend_models = [m for m in MODEL_ORDER if m in all_models_in_fig]
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=COLORS[m], edgecolor="black", linewidth=0.6
        )
        for m in legend_models
    ]
    mean_line = Line2D([0], [0], color="black", linewidth=0.8, label="Mean fidelity")
    handles.append(mean_line)
    legend_models.append("Mean fidelity")

    fig.legend(
        handles,
        legend_models,
        loc="upper center",
        ncol=len(legend_models),
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=7,
    )

    out_pdf = OUTPUT_DIR / "fidelity_barplot.pdf"
    out_png = OUTPUT_DIR / "fidelity_barplot.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
