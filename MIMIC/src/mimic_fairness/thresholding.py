import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.005), 3)
PRIMARY_SELECTION_RULE = "min_weighted_objective_at_accuracy_floor"
ACCURACY_FLOOR_DROP = 0.005
DEFAULT_WEIGHTED_DISPARITY_LAMBDA = 0.1
VALID_SELECTION_RULES = {
    "min_overall_fnr_at_accuracy_floor",
    "min_max_group_fnr_at_accuracy_floor",
    "min_fnr_disparity_at_accuracy_floor",
    "min_weighted_objective_at_accuracy_floor",
}


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
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


def group_metrics(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for group, group_df in predictions.groupby("fairness_group", sort=True):
        metrics = binary_metrics(
            group_df["y_true"].to_numpy(),
            group_df["y_prob"].to_numpy(),
            threshold,
        )
        rows.append({"fairness_group": group, **metrics})
    return pd.DataFrame(rows)


def sweep_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    y_true = predictions["y_true"].to_numpy()
    y_prob = predictions["y_prob"].to_numpy()
    base_sweep = pd.DataFrame([binary_metrics(y_true, y_prob, threshold) for threshold in THRESHOLDS])
    return _add_group_fairness_metrics(predictions, base_sweep)


def _add_group_fairness_metrics(predictions: pd.DataFrame, sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in sweep["threshold"]:
        group_df = group_metrics(predictions, threshold)
        valid_group_fnr = group_df["fnr"].dropna()
        if len(valid_group_fnr):
            max_group_fnr = valid_group_fnr.max()
            min_group_fnr = valid_group_fnr.min()
            fnr_disparity = max_group_fnr - min_group_fnr
        else:
            max_group_fnr = np.nan
            min_group_fnr = np.nan
            fnr_disparity = np.nan
        rows.append(
            {
                "threshold": threshold,
                "max_group_fnr": max_group_fnr,
                "min_group_fnr": min_group_fnr,
                "fnr_disparity": fnr_disparity,
            }
        )
    fairness_metrics = pd.DataFrame(rows)
    return sweep.merge(fairness_metrics, on="threshold")


def select_threshold(
    validation_predictions: pd.DataFrame,
    sweep: pd.DataFrame,
    selection_rule: str = PRIMARY_SELECTION_RULE,
    accuracy_floor_drop: float = ACCURACY_FLOOR_DROP,
    weighted_disparity_lambda: float = DEFAULT_WEIGHTED_DISPARITY_LAMBDA,
) -> pd.Series:
    y_true = validation_predictions["y_true"].to_numpy()
    y_prob = validation_predictions["y_prob"].to_numpy()
    default_metrics = binary_metrics(y_true, y_prob, threshold=0.5)
    accuracy_floor = default_metrics["accuracy"] - accuracy_floor_drop

    candidates = sweep[sweep["accuracy"] >= accuracy_floor].copy()
    if candidates.empty:
        candidates = sweep.copy()

    if selection_rule not in VALID_SELECTION_RULES:
        raise ValueError(
            "Unsupported threshold selection rule: "
            f"{selection_rule}. Valid options are {sorted(VALID_SELECTION_RULES)}."
        )

    if selection_rule == "min_overall_fnr_at_accuracy_floor":
        sort_by = ["fnr", "accuracy", "fpr", "threshold"]
        ascending = [True, False, True, False]
    elif selection_rule == "min_max_group_fnr_at_accuracy_floor":
        sort_by = ["max_group_fnr", "accuracy", "fpr", "threshold"]
        ascending = [True, False, True, False]
    elif selection_rule == "min_fnr_disparity_at_accuracy_floor":
        sort_by = ["fnr_disparity", "accuracy", "fpr", "threshold"]
        ascending = [True, False, True, False]
    else:
        candidates["weighted_fnr_objective"] = (
            candidates["fnr"] + weighted_disparity_lambda * candidates["fnr_disparity"]
        )
        sort_by = ["weighted_fnr_objective", "accuracy", "fpr", "threshold"]
        ascending = [True, False, True, False]

    candidates = candidates.sort_values(sort_by, ascending=ascending)
    selected = candidates.iloc[0].copy()
    selected["selection_rule"] = PRIMARY_SELECTION_RULE
    selected["accuracy_floor"] = accuracy_floor
    selected["default_threshold_accuracy"] = default_metrics["accuracy"]
    selected["default_threshold_fnr"] = default_metrics["fnr"]
    selected["default_threshold_fpr"] = default_metrics["fpr"]
    selected["selection_rule"] = selection_rule
    baseline_group_metrics = group_metrics(validation_predictions, threshold=0.5)
    valid_default_fnr = baseline_group_metrics["fnr"].dropna()
    selected["default_threshold_max_group_fnr"] = (
        valid_default_fnr.max() if len(valid_default_fnr) else np.nan
    )
    selected["default_threshold_fnr_disparity"] = (
        valid_default_fnr.max() - valid_default_fnr.min() if len(valid_default_fnr) else np.nan
    )
    if "weighted_fnr_objective" in candidates.columns:
        selected["default_threshold_weighted_fnr_objective"] = (
            default_metrics["fnr"]
            + weighted_disparity_lambda * selected["default_threshold_fnr_disparity"]
        )
    return selected


def threshold_predictions(
    predictions: pd.DataFrame,
    threshold: float,
    selection_rule: str = PRIMARY_SELECTION_RULE,
) -> pd.DataFrame:
    thresholded = predictions.copy()
    thresholded["y_pred"] = (thresholded["y_prob"] >= threshold).astype(int)
    thresholded["threshold"] = threshold
    thresholded["threshold_selected_on"] = "val"
    thresholded["threshold_selection_rule"] = selection_rule
    return thresholded


def evaluate_model_thresholds(
    model_name: str,
    val_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    selection_rule: str = PRIMARY_SELECTION_RULE,
    accuracy_floor_drop: float = ACCURACY_FLOOR_DROP,
    weighted_disparity_lambda: float = DEFAULT_WEIGHTED_DISPARITY_LAMBDA,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val_sweep = sweep_thresholds(val_predictions)
    val_sweep["selection_rule"] = selection_rule
    selected = select_threshold(
        val_predictions,
        val_sweep,
        selection_rule=selection_rule,
        accuracy_floor_drop=accuracy_floor_drop,
        weighted_disparity_lambda=weighted_disparity_lambda,
    )
    threshold = float(selected["threshold"])

    val_selected = binary_metrics(
        val_predictions["y_true"].to_numpy(),
        val_predictions["y_prob"].to_numpy(),
        threshold,
    )
    test_selected = binary_metrics(
        test_predictions["y_true"].to_numpy(),
        test_predictions["y_prob"].to_numpy(),
        threshold,
    )

    val_selected.update(
        {
            "model": model_name,
            "split": "val",
            "selection_rule": selection_rule,
            "selected_on": "val",
        }
    )
    test_selected.update(
        {
            "model": model_name,
            "split": "test",
            "selection_rule": selection_rule,
            "selected_on": "val",
        }
    )

    test_group_metrics = group_metrics(test_predictions, threshold)
    test_group_metrics.insert(0, "model", model_name)
    test_group_metrics.insert(1, "split", "test")
    test_group_metrics["selection_rule"] = selection_rule
    test_group_metrics["selected_on"] = "val"

    val_sweep.insert(0, "model", model_name)
    selected = selected.to_frame().T
    selected.insert(0, "model", model_name)

    thresholded_test_predictions = threshold_predictions(
        test_predictions,
        threshold,
        selection_rule=selection_rule,
    )
    return (
        val_sweep,
        selected,
        pd.DataFrame([val_selected, test_selected]),
        test_group_metrics,
        thresholded_test_predictions,
    )
