import pandas as pd

from mimic_fairness.paths import load_config, project_root
from mimic_fairness.plots import plot_fairness_comparison


HELDOUT_PLOT_METRICS = [
    "accuracy",
    "auc",
    "fnr",
    "fpr",
    "recall",
    "specificity",
    "precision",
    "predicted_positive_rate",
]


def _first_existing_path(*paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these files exist: {[str(path) for path in paths]}")


def _save_long_metric_summary(
    baseline_results: pd.DataFrame,
    reweighted_results: pd.DataFrame,
    output_path,
) -> None:
    baseline = baseline_results.copy()
    baseline.insert(0, "model", "baseline")

    reweighted = reweighted_results.copy()
    reweighted.insert(0, "model", "reweighted")

    pd.concat([baseline, reweighted], ignore_index=True).to_csv(output_path, index=False)


def _save_metric_comparison(
    baseline_results: pd.DataFrame,
    reweighted_results: pd.DataFrame,
    output_path,
) -> pd.DataFrame:
    comparison = baseline_results.merge(
        reweighted_results,
        on="fairness_group",
        suffixes=("_baseline", "_reweighted"),
        validate="one_to_one",
    )

    for metric in HELDOUT_PLOT_METRICS:
        baseline_col = f"{metric}_baseline"
        reweighted_col = f"{metric}_reweighted"
        if baseline_col in comparison.columns and reweighted_col in comparison.columns:
            comparison[f"{metric}_delta"] = comparison[reweighted_col] - comparison[baseline_col]

    comparison = comparison.sort_values("fnr_delta")
    comparison.to_csv(output_path, index=False)
    return comparison


def _save_heldout_plots(
    baseline_results: pd.DataFrame,
    reweighted_results: pd.DataFrame,
    plots_dir,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    for metric in HELDOUT_PLOT_METRICS:
        if metric not in baseline_results.columns or metric not in reweighted_results.columns:
            continue

        output_path = plots_dir / f"heldout_test_{metric}_baseline_vs_reweighted.png"
        plot_fairness_comparison(
            baseline_results,
            reweighted_results,
            str(output_path),
            metric=metric,
            title=f"Held-Out Test {metric.replace('_', ' ').title()}: Baseline vs Reweighted",
        )


def main() -> None:
    root = project_root()
    cfg = load_config()

    outputs_dir = root / cfg["paths"]["outputs_dir"]
    tables_dir = outputs_dir / "tables"
    plots_dir = outputs_dir / "plots"

    baseline_path = _first_existing_path(
        tables_dir / "heldout_test_baseline_fairness_results.parquet",
        tables_dir / "test_baseline_fairness_results.parquet",
    )
    reweighted_path = _first_existing_path(
        tables_dir / "heldout_test_reweighted_fairness_results.parquet",
        tables_dir / "test_reweighted_fairness_results.parquet",
    )

    baseline_results = pd.read_parquet(baseline_path)
    reweighted_results = pd.read_parquet(reweighted_path)

    comparison = _save_metric_comparison(
        baseline_results,
        reweighted_results,
        tables_dir / "heldout_test_fairness_comparison.csv",
    )
    _save_long_metric_summary(
        baseline_results,
        reweighted_results,
        tables_dir / "heldout_test_group_metrics_long.csv",
    )
    _save_heldout_plots(baseline_results, reweighted_results, plots_dir)

    print("Saved held-out test comparison and graph outputs.")
    print(f"Comparison: {tables_dir / 'heldout_test_fairness_comparison.csv'}")
    print(f"Long metrics: {tables_dir / 'heldout_test_group_metrics_long.csv'}")
    print(f"Plots: {plots_dir}")
    print("\nHeld-out test metric comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
