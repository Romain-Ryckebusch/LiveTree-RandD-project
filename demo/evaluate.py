"""
Evaluation module: compare predictions from imputed data vs. clean baseline.
"""
import numpy as np


def mae(actual, predicted):
    """Mean Absolute Error."""
    return np.mean(np.abs(actual - predicted))


def rmse(actual, predicted):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mape(actual, predicted):
    """Mean Absolute Percentage Error (ignores zero actuals)."""
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def evaluate(baseline_pred, imputed_pred, actual=None):
    """
    Compare baseline (no-gap) predictions against imputed predictions.

    Parameters
    ----------
    baseline_pred : np.ndarray
        Predictions from clean (no-gap) data.
    imputed_pred : np.ndarray
        Predictions from imputed (gap-filled) data.
    actual : np.ndarray, optional
        Actual measured values, if available.

    Returns
    -------
    dict with metric results.
    """
    results = {
        "baseline_vs_imputed": {
            "mae": mae(baseline_pred, imputed_pred),
            "rmse": rmse(baseline_pred, imputed_pred),
            "mape": mape(baseline_pred, imputed_pred),
        }
    }

    if actual is not None:
        results["baseline_vs_actual"] = {
            "mae": mae(actual, baseline_pred),
            "rmse": rmse(actual, baseline_pred),
            "mape": mape(actual, baseline_pred),
        }
        results["imputed_vs_actual"] = {
            "mae": mae(actual, imputed_pred),
            "rmse": rmse(actual, imputed_pred),
            "mape": mape(actual, imputed_pred),
        }

    return results


def print_report(results):
    """Print a formatted evaluation report."""
    for comparison, metrics in results.items():
        print(f"\n--- {comparison} ---")
        for name, value in metrics.items():
            if name == "mape":
                print(f"  {name.upper()}: {value:.2f}%")
            else:
                print(f"  {name.upper()}: {value:.2f}")
