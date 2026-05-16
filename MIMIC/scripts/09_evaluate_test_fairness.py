import pandas as pd

from mimic_fairness.paths import load_config, project_root
from mimic_fairness.evaluate import evaluate_fairness


KEY_COLUMNS = ["SUBJECT_ID", "HADM_ID"]


def _validate_paired_prediction_files(baseline_path, reweighted_path) -> None:
    baseline = pd.read_parquet(baseline_path, columns=KEY_COLUMNS + ["split"])
    reweighted = pd.read_parquet(reweighted_path, columns=KEY_COLUMNS + ["split"])

    if baseline["HADM_ID"].duplicated().any():
        raise ValueError("Baseline test predictions contain duplicate HADM_ID values.")
    if reweighted["HADM_ID"].duplicated().any():
        raise ValueError("Reweighted test predictions contain duplicate HADM_ID values.")

    if set(baseline["split"].unique()) != {"test"}:
        raise ValueError("Baseline test predictions include non-test rows.")
    if set(reweighted["split"].unique()) != {"test"}:
        raise ValueError("Reweighted test predictions include non-test rows.")

    baseline_keys = baseline[KEY_COLUMNS]
    reweighted_keys = reweighted[KEY_COLUMNS]
    if baseline_keys.equals(reweighted_keys):
        print("Validated paired test predictions: identical SUBJECT_ID/HADM_ID order.")
        return

    paired = baseline_keys.merge(
        reweighted_keys,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(baseline_keys) or len(paired) != len(reweighted_keys):
        raise ValueError(
            "Baseline and reweighted test predictions do not contain identical "
            "SUBJECT_ID/HADM_ID pairs."
        )

    print(
        "Validated paired test predictions: same SUBJECT_ID/HADM_ID set; downstream paired "
        "tests merge by SUBJECT_ID/HADM_ID."
    )


def main() -> None:
    root = project_root()
    cfg = load_config()

    label_name = cfg["active_label"]
    cohort_path = root / cfg["paths"]["interim_dir"] / f"{label_name}_cohort.parquet"
    split_path = root / cfg["paths"]["interim_dir"] / f"{label_name}_splits.parquet"

    models_dir = root / cfg["paths"]["models_dir"]
    baseline_model_path = models_dir / "best_model"
    reweighted_model_path = models_dir / "reweighted" / "best_model"

    if not baseline_model_path.exists():
        raise FileNotFoundError(f"Baseline model not found at {baseline_model_path}")
    if not reweighted_model_path.exists():
        raise FileNotFoundError(f"Reweighted model not found at {reweighted_model_path}")

    results_dir = root / cfg["paths"]["outputs_dir"] / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    baseline_predictions_path = results_dir / "test_baseline_predictions.parquet"
    reweighted_predictions_path = results_dir / "test_reweighted_predictions.parquet"

    print("Evaluating baseline model on held-out test split...")
    baseline_results = evaluate_fairness(
        cohort_path=str(cohort_path),
        checkpoint_path=str(baseline_model_path),
        max_length=cfg["cohort"]["max_note_length"],
        label_column=label_name,
        batch_size=cfg["model"]["batch_size"],
        split_path=str(split_path),
        split="test",
        predictions_output_path=str(baseline_predictions_path),
    )

    print("\nEvaluating reweighted model on held-out test split...")
    reweighted_results = evaluate_fairness(
        cohort_path=str(cohort_path),
        checkpoint_path=str(reweighted_model_path),
        max_length=cfg["cohort"]["max_note_length"],
        label_column=label_name,
        batch_size=cfg["model"]["batch_size"],
        split_path=str(split_path),
        split="test",
        predictions_output_path=str(reweighted_predictions_path),
    )

    _validate_paired_prediction_files(baseline_predictions_path, reweighted_predictions_path)

    baseline_results.to_parquet(
        results_dir / "test_baseline_fairness_results.parquet",
        index=False,
    )
    reweighted_results.to_parquet(
        results_dir / "test_reweighted_fairness_results.parquet",
        index=False,
    )

    comparison = baseline_results.merge(
        reweighted_results,
        on="fairness_group",
        suffixes=("_baseline", "_reweighted"),
    )
    comparison["fnr_delta"] = comparison["fnr_reweighted"] - comparison["fnr_baseline"]
    comparison["accuracy_delta"] = (
        comparison["accuracy_reweighted"] - comparison["accuracy_baseline"]
    )
    comparison = comparison.sort_values("fnr_delta")
    comparison.to_csv(results_dir / "test_fairness_comparison.csv", index=False)

    print("\nHeld-out test fairness comparison:")
    print(comparison.to_string(index=False))
    print(f"\nSaved held-out test results to {results_dir}")


if __name__ == "__main__":
    main()
