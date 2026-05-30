import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from mimic_fairness.evaluate import evaluate_fairness
from mimic_fairness.paths import load_config, project_root


THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.005), 3)
PRIMARY_SELECTION_RULE = "min_fnr_at_accuracy_floor"
ACCURACY_FLOOR_DROP = 0.005
KEY_COLUMNS = ["SUBJECT_ID", "HADM_ID"]


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    positives = int(tp + fn)
    negatives = int(tn + fp)
    total = int(positives + negatives)

    accuracy = (tp + tn) / total if total else np.nan
    fnr = fn / positives if positives else np.nan
    fpr = fp / negatives if negatives else np.nan
    recall = tp / positives if positives else np.nan
    specificity = tn / negatives if negatives else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    predicted_positive_rate = (tp + fp) / total if total else np.nan
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    return {
        "threshold": threshold,
        "n_samples": total,
        "positives": positives,
        "negatives": negatives,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": accuracy,
        "auc": auc,
        "fnr": fnr,
        "fpr": fpr,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "precision": precision,
        "predicted_positive_rate": predicted_positive_rate,
    }


def _group_metrics(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for group, group_df in predictions.groupby("fairness_group", sort=True):
        metrics = _binary_metrics(
            group_df["y_true"].to_numpy(),
            group_df["y_prob"].to_numpy(),
            threshold,
        )
        rows.append({"fairness_group": group, **metrics})
    return pd.DataFrame(rows)


def _sweep_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    y_true = predictions["y_true"].to_numpy()
    y_prob = predictions["y_prob"].to_numpy()
    return pd.DataFrame([_binary_metrics(y_true, y_prob, threshold) for threshold in THRESHOLDS])


def _select_threshold(sweep: pd.DataFrame) -> pd.Series:
    default_metrics = _binary_metrics(
        sweep.attrs["y_true"],
        sweep.attrs["y_prob"],
        threshold=0.5,
    )
    accuracy_floor = default_metrics["accuracy"] - ACCURACY_FLOOR_DROP

    candidates = sweep[sweep["accuracy"] >= accuracy_floor].copy()
    if candidates.empty:
        candidates = sweep.copy()

    candidates = candidates.sort_values(
        ["fnr", "accuracy", "fpr", "threshold"],
        ascending=[True, False, True, False],
    )
    selected = candidates.iloc[0].copy()
    selected["selection_rule"] = PRIMARY_SELECTION_RULE
    selected["accuracy_floor"] = accuracy_floor
    selected["default_threshold_accuracy"] = default_metrics["accuracy"]
    selected["default_threshold_fnr"] = default_metrics["fnr"]
    selected["default_threshold_fpr"] = default_metrics["fpr"]
    return selected


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
    val_sweep = _sweep_thresholds(val_predictions)
    val_sweep.attrs["y_true"] = val_predictions["y_true"].to_numpy()
    val_sweep.attrs["y_prob"] = val_predictions["y_prob"].to_numpy()

    selected = _select_threshold(val_sweep)
    threshold = float(selected["threshold"])

    val_selected = _binary_metrics(
        val_predictions["y_true"].to_numpy(),
        val_predictions["y_prob"].to_numpy(),
        threshold,
    )
    test_selected = _binary_metrics(
        test_predictions["y_true"].to_numpy(),
        test_predictions["y_prob"].to_numpy(),
        threshold,
    )

    val_selected.update(
        {
            "model": model_name,
            "split": "val",
            "selection_rule": PRIMARY_SELECTION_RULE,
            "selected_on": "val",
        }
    )
    test_selected.update(
        {
            "model": model_name,
            "split": "test",
            "selection_rule": PRIMARY_SELECTION_RULE,
            "selected_on": "val",
        }
    )

    test_group_metrics = _group_metrics(test_predictions, threshold)
    test_group_metrics.insert(0, "model", model_name)
    test_group_metrics.insert(1, "split", "test")
    test_group_metrics["selection_rule"] = PRIMARY_SELECTION_RULE
    test_group_metrics["selected_on"] = "val"

    val_sweep.insert(0, "model", model_name)
    selected = selected.to_frame().T
    selected.insert(0, "model", model_name)

    return val_sweep, selected, pd.DataFrame([val_selected, test_selected]), test_group_metrics


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

        sweep, selected, metrics, group_metrics = _evaluate_model_thresholds(
            model_name,
            val_predictions,
            test_predictions,
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
