#!/usr/bin/env python3
"""
For every combination of (task × noise_level × prompt_mode), find the
LLM model with the highest best_of_runs_fidelity and record its metrics.

Output columns:
    task, noise_level, prompt_mode, best_model,
    best_of_runs_fidelity, mean_fidelity,
    best_run_tokens, best_run_k_reach_tol

Total expected rows: 16 tasks × 3 noise levels × 3 prompt modes = 144

Output: ALL_IN_AVERAGE/Best_performance.csv
"""

import pathlib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = (
    pathlib.Path(__file__).resolve().parents[1].parent
    / "quctrl-batch-experiments"
    / "data"
)
OUT_DIR = DATA_DIR / "ALL_IN_AVERAGE"
OUT_FILE = OUT_DIR / "Best_performance.csv"

# ---------------------------------------------------------------------------
# All 16 task CSVs
# ---------------------------------------------------------------------------
TASK_CSVS = [
    DATA_DIR / "cd_driving" / "final_agg_by_noise_model.csv",
    DATA_DIR / "control_crab_2qubit_interaction" / "final_agg_by_noise_model.csv",
    DATA_DIR / "control_grape_cphase" / "final_agg_by_noise_model.csv",
    DATA_DIR
    / "control_grape_toffoli"
    / "final_agg_by_noise_model_control_grape_toffoli.csv",
    DATA_DIR / "crab_qft" / "final_agg_by_noise_model_crab_qft.csv",
    DATA_DIR / "decoherence_suppression" / "final_agg_by_noise_model.csv",
    DATA_DIR / "Dicke" / "final_agg_by_noise_model_dicke.csv",
    DATA_DIR / "drag_pulse" / "final_agg_by_noise_model_drag_pulse.csv",
    DATA_DIR / "lambda_transfer" / "final_agg_by_noise_model_lambda_transfer.csv",
    DATA_DIR / "landau_zener_csv_datas" / "final_agg_by_noise_model.csv",
    DATA_DIR / "lindbladian" / "final_agg_by_noise_model.csv",
    DATA_DIR / "qutip_single_qubit" / "final_agg_by_noise_model_qutip_single_qubit.csv",
    DATA_DIR / "Single_qubit_gate" / "final_agg_by_noise_model_single_qubit_gate.csv",
    DATA_DIR
    / "symplectic_oscillator"
    / "final_agg_by_noise_model_symplectic_oscillator.csv",
    DATA_DIR / "transmon_xgate" / "final_agg_by_noise_model_transmon_xgate.csv",
    DATA_DIR / "two_spin_transfer" / "final_agg_by_noise_model_two_spin_transfer.csv",
]

# ---------------------------------------------------------------------------
# Model-name normalisation (same as aggregate_noise.py)
# ---------------------------------------------------------------------------
MODEL_UNIFY = {
    "claude-sonnet": "claude-sonnet",
    "gemini-flash": "gemini-flash",
    "gpt-5.1": "gpt-5.1",
    "gpt-oss": "gpt-oss",
    "gpt-oss-120b": "gpt-oss",
    "qwen3": "Qwen3",
    "qwen3-30b": "Qwen3",
    "qwen3.5": "Qwen3.5",
    "qwen3.5-plus": "Qwen3.5",
}

METRICS = [
    "best_of_runs_fidelity",
    "mean_fidelity",
    "best_run_tokens",
    "best_run_k_reach_tol",
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for csv_path in TASK_CSVS:
        if not csv_path.exists():
            print(f"[WARNING] File not found, skipping: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["noise_level"] = df["noise_level"].astype(float)
        df["model"] = df["model"].map(MODEL_UNIFY).fillna(df["model"])
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)

    # For each (task, noise_level, prompt_mode) pick the row with the
    # highest best_of_runs_fidelity
    cols_needed = ["task", "noise_level", "prompt_mode", "model"] + METRICS
    all_data = all_data[[c for c in cols_needed if c in all_data.columns]]

    def pick_best(group):
        best_row = group.loc[group["best_of_runs_fidelity"].idxmax()]
        return pd.Series(
            {
                "best_model": best_row["model"],
                "best_of_runs_fidelity": best_row["best_of_runs_fidelity"],
                "mean_fidelity": best_row["mean_fidelity"],
                "best_run_tokens": best_row["best_run_tokens"],
                "best_run_k_reach_tol": best_row["best_run_k_reach_tol"],
            }
        )

    result = (
        all_data.groupby(["task", "noise_level", "prompt_mode"], as_index=False)
        .apply(pick_best)
        .reset_index(drop=True)
    )

    # Sort: task → noise_level → prompt_mode
    noise_order = {0.0: 0, 0.02: 1, 0.1: 2}
    prompt_order = {"function": 0, "normal_physical": 1, "normal_simple": 2}
    result["_ns"] = result["noise_level"].map(noise_order).fillna(99)
    result["_ps"] = result["prompt_mode"].map(prompt_order).fillna(99)
    result = result.sort_values(["task", "_ns", "_ps"]).drop(columns=["_ns", "_ps"])

    result.to_csv(OUT_FILE, index=False)
    print(f"Saved  : {OUT_FILE}")
    print(f"Shape  : {result.shape}  (expected 117 rows)")
    print(result.head(9).to_string())


if __name__ == "__main__":
    main()
