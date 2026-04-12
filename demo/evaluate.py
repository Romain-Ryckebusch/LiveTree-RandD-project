import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual, predicted):
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def evaluate(baseline_pred, imputed_pred, naive_pred=None):
    results = {
        "baseline_vs_imputed": {
            "mae": mae(baseline_pred, imputed_pred),
            "rmse": rmse(baseline_pred, imputed_pred),
            "mape": mape(baseline_pred, imputed_pred),
        }
    }

    if naive_pred is not None:
        results["baseline_vs_naive"] = {
            "mae": mae(baseline_pred, naive_pred),
            "rmse": rmse(baseline_pred, naive_pred),
            "mape": mape(baseline_pred, naive_pred),
        }

    return results


def print_report(results):
    for comparison, metrics in results.items():
        print(f"\n{comparison}")
        for name, value in metrics.items():
            if name == "mape":
                print(f"  {name.upper()}: {value:.2f}%")
            else:
                print(f"  {name.upper()}: {value:.2f}")


@dataclass
class ScenarioResult:
    scenario_label: str
    target_date: str

    gap_mode: str
    block_length: int = 0
    n_blocks: int = 0
    n_points: int = 0
    seed: int = 42

    baseline_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    imputed_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    target_timestamps: list = field(default_factory=list)

    history_dates: Optional[pd.Series] = None
    history_clean: Optional[np.ndarray] = None
    history_gapped: Optional[np.ndarray] = None
    history_imputed: Optional[np.ndarray] = None
    quality_flags: Optional[np.ndarray] = None

    naive_pred: Optional[np.ndarray] = None
    history_naive: Optional[np.ndarray] = None

    # (start, end) tuples for blocks, ints for random
    gap_info: list = field(default_factory=list)

    metrics: Dict = field(default_factory=dict)


def save_summary_table(results: List[ScenarioResult], output_dir: str):
    rows = []
    for r in results:
        row = {
            "scenario_label": r.scenario_label,
            "target_date": r.target_date,
            "gap_mode": r.gap_mode,
            "block_length": r.block_length,
            "n_blocks": r.n_blocks,
            "n_points": r.n_points,
            "seed": r.seed,
        }
        for comparison, metrics in r.metrics.items():
            prefix = {
                "baseline_vs_imputed": "bvi",
                "baseline_vs_naive": "bvn",
            }.get(comparison, comparison)
            for metric_name, value in metrics.items():
                row[f"{prefix}_{metric_name}"] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"Summary JSON saved to {json_path}")


def compute_aggregate(results: List[ScenarioResult]) -> Dict:
    if not results:
        return {}

    all_metrics = {}
    for r in results:
        for comparison, metrics in r.metrics.items():
            for name, value in metrics.items():
                key = f"{comparison}.{name}"
                all_metrics.setdefault(key, []).append(value)

    aggregate = {}
    for key, values in all_metrics.items():
        arr = np.array([v for v in values if not np.isnan(v)])
        if len(arr) == 0:
            continue
        aggregate[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": len(arr),
        }

    return aggregate
