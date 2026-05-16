from mimic_fairness.paths import load_config, project_root
from mimic_fairness.evaluate import evaluate_fairness
from mimic_fairness.plots import plot_fairness_comparison


def main():
    root = project_root()
    cfg = load_config()

    label_name = cfg["active_label"]
    cohort_path = root / cfg["paths"]["interim_dir"] / f"{label_name}_cohort.parquet"

    models_dir = root / cfg["paths"]["models_dir"]
    baseline_model_path = models_dir / "best_model"
    reweighted_model_path = models_dir / "reweighted" / "best_model"

    if not baseline_model_path.exists():
        raise FileNotFoundError(f"Baseline model not found at {baseline_model_path}")
    if not reweighted_model_path.exists():
        raise FileNotFoundError(f"Reweighted model not found at {reweighted_model_path}")

    print(
        "EXPLORATORY FULL-COHORT COMPARISON: this uses all cohort rows and is "
        "not the held-out test comparison."
    )

    print("Evaluating baseline model on full cohort...")
    baseline_results = evaluate_fairness(
        cohort_path=str(cohort_path),
        checkpoint_path=str(baseline_model_path),
        max_length=cfg["cohort"]["max_note_length"],
        label_column=label_name,
        batch_size=cfg["model"]["batch_size"],
    )

    print("\nEvaluating reweighted model on full cohort...")
    reweighted_results = evaluate_fairness(
        cohort_path=str(cohort_path),
        checkpoint_path=str(reweighted_model_path),
        max_length=cfg["cohort"]["max_note_length"],
        label_column=label_name,
        batch_size=cfg["model"]["batch_size"],
    )

    results_dir = root / cfg["paths"]["outputs_dir"] / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline_results.to_parquet(results_dir / "baseline_fairness_results.parquet", index=False)
    reweighted_results.to_parquet(results_dir / "reweighted_fairness_results.parquet", index=False)

    comparison = baseline_results.merge(
        reweighted_results,
        on="fairness_group",
        suffixes=("_baseline", "_reweighted"),
    )

    comparison["fnr_delta"] = comparison["fnr_reweighted"] - comparison["fnr_baseline"]
    comparison["accuracy_delta"] = comparison["accuracy_reweighted"] - comparison["accuracy_baseline"]

    comparison = comparison.sort_values("fnr_delta")

    comparison.to_csv(results_dir / "fairness_comparison.csv", index=False)

    print("\n" + "="*80)
    print("EXPLORATORY FULL-COHORT FAIRNESS COMPARISON RESULTS")
    print("="*80)
    print(comparison.to_string())

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Mean FNR Baseline: {baseline_results['fnr'].mean():.4f}")
    print(f"Mean FNR Reweighted: {reweighted_results['fnr'].mean():.4f}")
    print(f"Mean FNR Improvement: {(baseline_results['fnr'].mean() - reweighted_results['fnr'].mean()):.4f}")

    print(f"\nMean Accuracy Baseline: {baseline_results['accuracy'].mean():.4f}")
    print(f"Mean Accuracy Reweighted: {reweighted_results['accuracy'].mean():.4f}")
    print(f"Mean Accuracy Change: {(reweighted_results['accuracy'].mean() - baseline_results['accuracy'].mean()):.4f}")

    plots_dir = root / cfg["paths"]["outputs_dir"] / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots...")
    plot_fairness_comparison(
        baseline_results,
        reweighted_results,
        str(plots_dir / "fnr_comparison.png"),
        metric="fnr",
    )

    plot_fairness_comparison(
        baseline_results,
        reweighted_results,
        str(plots_dir / "accuracy_comparison.png"),
        metric="accuracy",
    )

    print(f"Results saved to {results_dir}")
    print(f"Plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
