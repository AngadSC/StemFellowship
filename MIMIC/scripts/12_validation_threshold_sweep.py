import pandas as pd
from mimic_fairness.thresholding import evaluate_model_thresholds

from mimic_fairness.evaluate import evaluate_fairness
from mimic_fairness.paths import load_config, project_root


KEY_COLUMNS = ["SUBJECT_ID", "HADM_ID"]


def _create_predictions(
    *,
    cohort_path,
    split_path,
    checkpoint_path,
    max_length: int,
    label_column: str,
    batch_size: int,
    split: str,
    output_path,
) -> pd.DataFrame:
    evaluate_fairness(
        cohort_path=str(cohort_path),
        checkpoint_path=str(checkpoint_path),
        max_length=max_length,
        label_column=label_column,
        batch_size=batch_size,
        split_path=str(split_path),
        split=split,
        predictions_output_path=str(output_path),
    )
    predictions = pd.read_parquet(output_path)

    if set(predictions["split"].unique()) != {split}:
        raise ValueError(f"{output_path} does not contain only split={split!r}.")

    required_columns = set(KEY_COLUMNS + ["fairness_group", "y_true", "y_prob", "split"])
    missing_columns = required_columns.difference(predictions.columns)
    if missing_columns:
        raise ValueError(f"{output_path} is missing columns: {sorted(missing_columns)}")

    return predictions


def _evaluate_model_thresholds(model_name: str, val_predictions: pd.DataFrame, test_predictions: pd.DataFrame):
    raise NotImplementedError(
        "Do not call _evaluate_model_thresholds directly. Use evaluate_model_thresholds from mimic_fairness.thresholding."
    )


def main() -> None:
    root = project_root()
    cfg = load_config()

    label_name = cfg["active_label"]
    cohort_path = root / cfg["paths"]["interim_dir"] / f"{label_name}_cohort.parquet"
    split_path = root / cfg["paths"]["interim_dir"] / f"{label_name}_splits.parquet"

    models_dir = root / cfg["paths"]["models_dir"] / "by_disease" / label_name
    model_paths = {
        "baseline": models_dir / "baseline" / "best_model",
        "reweighted": models_dir / "reweighted" / "best_model",
    }

    for model_name, checkpoint_path in model_paths.items():
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"{model_name} model not found at {checkpoint_path}")

    results_dir = root / cfg["paths"]["outputs_dir"] / "by_disease" / label_name / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_sweeps = []
    selected_thresholds = []
    selected_metrics = []
    test_group_metrics = []

    for model_name, checkpoint_path in model_paths.items():
        val_predictions_path = results_dir / f"val_{model_name}_predictions.parquet"
        test_predictions_path = results_dir / f"heldout_test_{model_name}_predictions.parquet"

        print(f"Creating {model_name} validation predictions from the current checkpoint...")
        val_predictions = _create_predictions(
            cohort_path=cohort_path,
            split_path=split_path,
            checkpoint_path=checkpoint_path,
            max_length=cfg["cohort"]["max_note_length"],
            label_column=label_name,
            batch_size=cfg["model"]["batch_size"],
            split="val",
            output_path=val_predictions_path,
        )

        print(f"Creating {model_name} held-out test predictions from the current checkpoint...")
        test_predictions = _create_predictions(
            cohort_path=cohort_path,
            split_path=split_path,
            checkpoint_path=checkpoint_path,
            max_length=cfg["cohort"]["max_note_length"],
            label_column=label_name,
            batch_size=cfg["model"]["batch_size"],
            split="test",
            output_path=test_predictions_path,
        )

        selection_cfg = cfg.get("threshold_selection", {})
        sweep, selected, metrics, group_metrics, _ = evaluate_model_thresholds(
            model_name,
            val_predictions,
            test_predictions,
            selection_rule=selection_cfg.get(
                "rule",
                "min_weighted_objective_at_accuracy_floor",
            ),
            accuracy_floor_drop=selection_cfg.get("accuracy_floor_drop", 0.005),
            weighted_disparity_lambda=selection_cfg.get("weighted_disparity_lambda", 0.1),
        )
        all_sweeps.append(sweep)
        selected_thresholds.append(selected)
        selected_metrics.append(metrics)
        test_group_metrics.append(group_metrics)

    pd.concat(all_sweeps, ignore_index=True).to_csv(
        results_dir / f"{label_name}_validation_threshold_sweep.csv",
        index=False,
    )
    pd.concat(selected_thresholds, ignore_index=True).to_csv(
        results_dir / f"{label_name}_validation_selected_thresholds.csv",
        index=False,
    )
    pd.concat(selected_metrics, ignore_index=True).to_csv(
        results_dir / f"{label_name}_validation_threshold_selected_metrics.csv",
        index=False,
    )
    pd.concat(test_group_metrics, ignore_index=True).to_csv(
        results_dir / f"{label_name}_heldout_test_group_metrics_val_threshold.csv",
        index=False,
    )

    print("\nSelected validation thresholds:")
    print(pd.concat(selected_thresholds, ignore_index=True).to_string(index=False))
    print(f"\nSaved threshold sweep outputs to {results_dir}")


if __name__ == "__main__":
    main()
