import argparse
import importlib.util
from copy import deepcopy

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mimic_fairness.build_cohort import build_cohort
from mimic_fairness.evaluate import evaluate_fairness
from mimic_fairness.mitigation import compute_subgroup_weights
from mimic_fairness.paths import load_config, project_root
from mimic_fairness.plots import plot_fairness_comparison
from mimic_fairness.thresholding import evaluate_model_thresholds
from mimic_fairness.train import train_model, train_model_with_weights


SPLIT_ORDER = ["train", "val", "test"]
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
KEY_COLUMNS = ["SUBJECT_ID", "HADM_ID"]
PIPELINE_STAGES = [
    "cohort",
    "splits",
    "baseline_training",
    "reweighted_training",
    "heldout_evaluation",
    "default_threshold_stats",
    "validation_threshold_sweep",
    "validation_threshold_stats",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cohort building, splitting, baseline training, reweighted training, "
            "held-out evaluation, paired tests, and validation-threshold analysis for "
            "each configured diagnosis label."
        )
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional subset of labels to run. Defaults to every label in config.yaml.",
    )
    parser.add_argument(
        "--skip-existing-models",
        action="store_true",
        help="Reuse disease-specific checkpoints when they already exist.",
    )
    parser.add_argument(
        "--rebuild-cohorts",
        action="store_true",
        help="Rebuild cohort parquet files even when they already exist.",
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        help="Regenerate train/val/test split files even when they already exist.",
    )
    return parser.parse_args()


def _stratify_or_none(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return labels


def _make_subject_splits(patient_labels: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    subject_ids = patient_labels["SUBJECT_ID"].to_numpy()
    labels = patient_labels["patient_label"]

    train_subjects, holdout_subjects = train_test_split(
        subject_ids,
        test_size=0.30,
        random_state=random_seed,
        stratify=_stratify_or_none(labels),
    )

    holdout_labels = patient_labels.set_index("SUBJECT_ID").loc[holdout_subjects, "patient_label"]
    val_subjects, test_subjects = train_test_split(
        holdout_subjects,
        test_size=0.50,
        random_state=random_seed,
        stratify=_stratify_or_none(holdout_labels),
    )

    return pd.concat(
        [
            pd.DataFrame({"SUBJECT_ID": train_subjects, "split": "train"}),
            pd.DataFrame({"SUBJECT_ID": val_subjects, "split": "val"}),
            pd.DataFrame({"SUBJECT_ID": test_subjects, "split": "test"}),
        ],
        ignore_index=True,
    )


def _summarize_splits(split_rows: pd.DataFrame, label: str) -> pd.DataFrame:
    base_summary = (
        split_rows.groupby("split")
        .agg(
            row_count=("HADM_ID", "count"),
            subject_count=("SUBJECT_ID", "nunique"),
            positive_count=(label, "sum"),
            positive_rate=(label, "mean"),
        )
        .reindex(SPLIT_ORDER)
    )

    group_counts = pd.crosstab(split_rows["split"], split_rows["fairness_group"])
    group_counts = group_counts.add_prefix("group_count_").reindex(SPLIT_ORDER).fillna(0).astype(int)
    return pd.concat([base_summary, group_counts], axis=1).reset_index()


def _build_or_load_cohort(cfg: dict, root, label: str, rebuild: bool) -> pd.DataFrame:
    interim_dir = root / cfg["paths"]["interim_dir"]
    interim_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = interim_dir / f"{label}_cohort.parquet"

    if cohort_path.exists() and not rebuild:
        print(f"[{label}] Reusing cohort: {cohort_path}")
        return pd.read_parquet(cohort_path)

    print(f"[{label}] Building cohort...")
    df = build_cohort(cfg, root)
    df.to_parquet(cohort_path, index=False)
    print(f"[{label}] Saved cohort to {cohort_path} rows={len(df)}")
    return df


def _build_or_load_splits(cfg: dict, root, label: str, cohort: pd.DataFrame, resplit: bool) -> pd.DataFrame:
    interim_dir = root / cfg["paths"]["interim_dir"]
    splits_path = interim_dir / f"{label}_splits.parquet"
    summary_path = interim_dir / f"{label}_split_summary.csv"

    if splits_path.exists() and not resplit:
        print(f"[{label}] Reusing splits: {splits_path}")
        return pd.read_parquet(splits_path)

    print(f"[{label}] Creating patient-disjoint train/val/test splits...")
    patient_labels = (
        cohort.groupby("SUBJECT_ID", as_index=False)[label]
        .max()
        .rename(columns={label: "patient_label"})
    )
    subject_splits = _make_subject_splits(patient_labels, cfg["model"]["random_seed"])
    split_rows = cohort.merge(subject_splits, on="SUBJECT_ID", how="left", validate="many_to_one")

    if split_rows["split"].isna().any():
        raise ValueError(f"[{label}] Some cohort rows were not assigned to a split.")
    split_counts = split_rows.groupby("SUBJECT_ID")["split"].nunique()
    if (split_counts > 1).any():
        examples = split_counts[split_counts > 1].index[:10].tolist()
        raise ValueError(f"[{label}] SUBJECT_ID leakage across splits. Examples: {examples}")

    splits = split_rows[["SUBJECT_ID", "HADM_ID", "split"]].copy()
    splits.to_parquet(splits_path, index=False)
    _summarize_splits(split_rows, label).to_csv(summary_path, index=False)
    print(f"[{label}] Saved splits to {splits_path}")
    return splits


def _train_baseline_model(
    cfg: dict,
    label: str,
    cohort_path,
    split_path,
    baseline_output_dir,
    skip_existing_model: bool,
):
    baseline_checkpoint = baseline_output_dir / "best_model"

    train_batch_size = cfg["model"].get("train_batch_size", cfg["model"]["batch_size"])
    gradient_accumulation_steps = cfg["model"].get("gradient_accumulation_steps", 1)
    optimizer_step_sleep_seconds = cfg["model"].get("optimizer_step_sleep_seconds", 0.0)

    if baseline_checkpoint.exists() and skip_existing_model:
        print(f"[{label}] Reusing baseline checkpoint: {baseline_checkpoint}")
        return baseline_checkpoint

    print(f"[{label}] Training baseline model...")
    train_model(
        cohort_path=str(cohort_path),
        split_path=str(split_path),
        model_name=cfg["model"]["base_model"],
        max_length=cfg["cohort"]["max_note_length"],
        label_column=label,
        output_dir=str(baseline_output_dir),
        num_epochs=cfg["model"]["num_epochs"],
        batch_size=train_batch_size,
        learning_rate=cfg["model"]["learning_rate"],
        random_seed=cfg["model"]["random_seed"],
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimizer_step_sleep_seconds=optimizer_step_sleep_seconds,
    )
    return baseline_checkpoint


def _train_reweighted_model(
    cfg: dict,
    label: str,
    cohort_path,
    split_path,
    reweighted_output_dir,
    skip_existing_model: bool,
):
    reweighted_checkpoint = reweighted_output_dir / "best_model"

    train_batch_size = cfg["model"].get("train_batch_size", cfg["model"]["batch_size"])
    gradient_accumulation_steps = cfg["model"].get("gradient_accumulation_steps", 1)
    optimizer_step_sleep_seconds = cfg["model"].get("optimizer_step_sleep_seconds", 0.0)

    if reweighted_checkpoint.exists() and skip_existing_model:
        print(f"[{label}] Reusing reweighted checkpoint: {reweighted_checkpoint}")
        return reweighted_checkpoint

    print(f"[{label}] Computing train-only subgroup weights...")
    cohort = pd.read_parquet(cohort_path)
    splits = pd.read_parquet(split_path)
    train_df = cohort.merge(
        splits[KEY_COLUMNS + ["split"]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    train_df = train_df[train_df["split"] == "train"].copy()
    sample_weights = compute_subgroup_weights(
        train_df,
        label_column=label,
        fairness_group_column="fairness_group",
        upweight_factor=2.0,
    )

    print(f"[{label}] Training reweighted mitigation model...")
    train_model_with_weights(
        cohort_path=str(cohort_path),
        split_path=str(split_path),
        model_name=cfg["model"]["base_model"],
        max_length=cfg["cohort"]["max_note_length"],
        label_column=label,
        output_dir=str(reweighted_output_dir),
        sample_weights=sample_weights,
        num_epochs=cfg["model"]["num_epochs"],
        batch_size=train_batch_size,
        learning_rate=cfg["model"]["learning_rate"],
        random_seed=cfg["model"]["random_seed"],
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimizer_step_sleep_seconds=optimizer_step_sleep_seconds,
    )
    return reweighted_checkpoint


def _validate_paired_prediction_files(baseline_path, reweighted_path) -> None:
    baseline = pd.read_parquet(baseline_path, columns=KEY_COLUMNS + ["split"])
    reweighted = pd.read_parquet(reweighted_path, columns=KEY_COLUMNS + ["split"])

    if set(baseline["split"].unique()) != {"test"}:
        raise ValueError("Baseline predictions include non-test rows.")
    if set(reweighted["split"].unique()) != {"test"}:
        raise ValueError("Reweighted predictions include non-test rows.")

    baseline_keys = baseline[KEY_COLUMNS]
    reweighted_keys = reweighted[KEY_COLUMNS]
    if baseline_keys.equals(reweighted_keys):
        print("Validated paired test predictions: identical SUBJECT_ID/HADM_ID order.")
        return

    paired = baseline_keys.merge(reweighted_keys, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(paired) != len(baseline_keys) or len(paired) != len(reweighted_keys):
        raise ValueError("Baseline and reweighted prediction files are not paired.")
    print("Validated paired test predictions: same SUBJECT_ID/HADM_ID set.")


def _save_heldout_plots(baseline_results: pd.DataFrame, reweighted_results: pd.DataFrame, plots_dir) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for metric in HELDOUT_PLOT_METRICS:
        if metric not in baseline_results.columns or metric not in reweighted_results.columns:
            continue
        plot_fairness_comparison(
            baseline_results,
            reweighted_results,
            str(plots_dir / f"heldout_test_{metric}_baseline_vs_reweighted.png"),
            metric=metric,
            title=f"Held-Out Test {metric.replace('_', ' ').title()}: Baseline vs Reweighted",
        )


def _evaluate_heldout(
    cfg: dict,
    label: str,
    cohort_path,
    split_path,
    baseline_checkpoint,
    reweighted_checkpoint,
    tables_dir,
    plots_dir,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_predictions_path = tables_dir / "heldout_test_baseline_predictions.parquet"
    reweighted_predictions_path = tables_dir / "heldout_test_reweighted_predictions.parquet"

    print(f"[{label}] Evaluating baseline on held-out test...")
    baseline_results = evaluate_fairness(
        cohort_path=str(cohort_path),
        checkpoint_path=str(baseline_checkpoint),
        max_length=cfg["cohort"]["max_note_length"],
        label_column=label,
        batch_size=cfg["model"]["batch_size"],
        split_path=str(split_path),
        split="test",
        predictions_output_path=str(baseline_predictions_path),
    )

    print(f"[{label}] Evaluating reweighted on held-out test...")
    reweighted_results = evaluate_fairness(
        cohort_path=str(cohort_path),
        checkpoint_path=str(reweighted_checkpoint),
        max_length=cfg["cohort"]["max_note_length"],
        label_column=label,
        batch_size=cfg["model"]["batch_size"],
        split_path=str(split_path),
        split="test",
        predictions_output_path=str(reweighted_predictions_path),
    )

    _validate_paired_prediction_files(baseline_predictions_path, reweighted_predictions_path)

    baseline_results.to_parquet(tables_dir / "heldout_test_baseline_fairness_results.parquet", index=False)
    reweighted_results.to_parquet(tables_dir / "heldout_test_reweighted_fairness_results.parquet", index=False)

    comparison = baseline_results.merge(
        reweighted_results,
        on="fairness_group",
        suffixes=("_baseline", "_reweighted"),
        validate="one_to_one",
    )
    for metric in HELDOUT_PLOT_METRICS:
        comparison[f"{metric}_delta"] = comparison[f"{metric}_reweighted"] - comparison[f"{metric}_baseline"]
    comparison = comparison.sort_values("fnr_delta")
    comparison.to_csv(tables_dir / "heldout_test_fairness_comparison.csv", index=False)

    baseline_long = baseline_results.copy()
    baseline_long.insert(0, "model", "baseline")
    reweighted_long = reweighted_results.copy()
    reweighted_long.insert(0, "model", "reweighted")
    pd.concat([baseline_long, reweighted_long], ignore_index=True).to_csv(
        tables_dir / "heldout_test_group_metrics_long.csv",
        index=False,
    )
    _save_heldout_plots(baseline_results, reweighted_results, plots_dir)

    baseline_predictions = pd.read_parquet(baseline_predictions_path)
    reweighted_predictions = pd.read_parquet(reweighted_predictions_path)
    return baseline_results, reweighted_results, baseline_predictions, reweighted_predictions


def _create_validation_predictions(
    cfg: dict,
    label: str,
    cohort_path,
    split_path,
    checkpoint_path,
    output_path,
) -> pd.DataFrame:
    evaluate_fairness(
        cohort_path=str(cohort_path),
        checkpoint_path=str(checkpoint_path),
        max_length=cfg["cohort"]["max_note_length"],
        label_column=label,
        batch_size=cfg["model"]["batch_size"],
        split_path=str(split_path),
        split="val",
        predictions_output_path=str(output_path),
    )
    predictions = pd.read_parquet(output_path)

    if set(predictions["split"].unique()) != {"val"}:
        raise ValueError(f"{output_path} does not contain only validation predictions.")

    required_columns = set(KEY_COLUMNS + ["fairness_group", "y_true", "y_prob", "split"])
    missing_columns = required_columns.difference(predictions.columns)
    if missing_columns:
        raise ValueError(f"{output_path} is missing columns: {sorted(missing_columns)}")

    return predictions


def _run_threshold_analysis(
    cfg: dict,
    label: str,
    cohort_path,
    split_path,
    baseline_checkpoint,
    reweighted_checkpoint,
    baseline_test_predictions: pd.DataFrame,
    reweighted_test_predictions: pd.DataFrame,
    tables_dir,
) -> None:
    val_predictions = {
        "baseline": _create_validation_predictions(
            cfg,
            label,
            cohort_path,
            split_path,
            baseline_checkpoint,
            tables_dir / "val_baseline_predictions.parquet",
        ),
        "reweighted": _create_validation_predictions(
            cfg,
            label,
            cohort_path,
            split_path,
            reweighted_checkpoint,
            tables_dir / "val_reweighted_predictions.parquet",
        ),
    }
    test_predictions = {
        "baseline": baseline_test_predictions,
        "reweighted": reweighted_test_predictions,
    }

    all_sweeps = []
    selected_thresholds = []
    selected_metrics = []
    test_group_metrics = []
    thresholded_dir = tables_dir / "val_threshold"
    thresholded_dir.mkdir(parents=True, exist_ok=True)

    for model_name in ["baseline", "reweighted"]:
        sweep, selected, metrics, group_metrics, thresholded_test = evaluate_model_thresholds(
            model_name,
            val_predictions[model_name],
            test_predictions[model_name],
        )
        all_sweeps.append(sweep)
        selected_thresholds.append(selected)
        selected_metrics.append(metrics)
        test_group_metrics.append(group_metrics)
        thresholded_test.to_parquet(
            thresholded_dir / f"heldout_test_{model_name}_predictions.parquet",
            index=False,
        )

    pd.concat(all_sweeps, ignore_index=True).to_csv(
        tables_dir / f"{label}_validation_threshold_sweep.csv",
        index=False,
    )
    pd.concat(selected_thresholds, ignore_index=True).to_csv(
        tables_dir / f"{label}_validation_selected_thresholds.csv",
        index=False,
    )
    pd.concat(selected_metrics, ignore_index=True).to_csv(
        tables_dir / f"{label}_validation_threshold_selected_metrics.csv",
        index=False,
    )
    pd.concat(test_group_metrics, ignore_index=True).to_csv(
        tables_dir / f"{label}_heldout_test_group_metrics_val_threshold.csv",
        index=False,
    )


def _load_stats_module(root):
    stats_path = root / "scripts" / "10_paired_statistical_tests.py"
    spec = importlib.util.spec_from_file_location("paired_statistical_tests", stats_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_paired_significance(root, results_dir) -> None:
    stats = _load_stats_module(root)
    paired = stats._load_paired_test_predictions(results_dir)

    rows = []
    stats._append_accuracy_result(rows, "overall", paired)
    stats._append_fnr_result(rows, "overall", paired)
    stats._append_fpr_result(rows, "overall", paired)

    for group, group_df in paired.groupby("fairness_group"):
        stats._append_accuracy_result(rows, group, group_df)
        stats._append_fnr_result(rows, group, group_df)
        stats._append_fpr_result(rows, group, group_df)

    results = pd.DataFrame(rows)
    results.to_csv(results_dir / "heldout_test_paired_significance.csv", index=False)

    fnr_ztests = stats._build_fnr_ztest_rows(paired)
    fnr_ztests.to_csv(results_dir / "heldout_test_fnr_group_z_tests.csv", index=False)
    fnr_ztests[
        [
            "fairness_group",
            "baseline_fnr",
            "baseline_fnr_ci_low",
            "baseline_fnr_ci_high",
            "reweighted_fnr",
            "reweighted_fnr_ci_low",
            "reweighted_fnr_ci_high",
        ]
    ].to_csv(results_dir / "heldout_test_fnr_confidence_intervals.csv", index=False)

    report = stats._build_statistical_report(results, fnr_ztests)
    (results_dir / "heldout_test_statistical_significance_report.txt").write_text(
        report,
        encoding="utf-8",
    )


def _run_label(base_cfg: dict, root, label: str, args: argparse.Namespace) -> None:
    cfg = deepcopy(base_cfg)
    cfg["active_label"] = label

    print("\n" + "=" * 80)
    print(f"RUNNING DIAGNOSIS: {label}")
    print("=" * 80)

    disease_models_dir = root / cfg["paths"]["models_dir"] / "by_disease" / label
    disease_outputs_dir = root / cfg["paths"]["outputs_dir"] / "by_disease" / label
    tables_dir = disease_outputs_dir / "tables"
    plots_dir = disease_outputs_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_output_dir = disease_models_dir / "baseline"
    reweighted_output_dir = disease_models_dir / "reweighted"
    cohort_path = root / cfg["paths"]["interim_dir"] / f"{label}_cohort.parquet"
    split_path = root / cfg["paths"]["interim_dir"] / f"{label}_splits.parquet"

    with tqdm(
        total=len(PIPELINE_STAGES),
        desc=f"{label} pipeline",
        unit="stage",
        leave=True,
    ) as stage_bar:
        stage_bar.set_postfix_str("building/loading cohort")
        cohort = _build_or_load_cohort(cfg, root, label, args.rebuild_cohorts)
        stage_bar.update(1)

        stage_bar.set_postfix_str("building/loading splits")
        _build_or_load_splits(cfg, root, label, cohort, args.resplit)
        stage_bar.update(1)

        stage_bar.set_postfix_str("training baseline")
        baseline_checkpoint = _train_baseline_model(
            cfg,
            label,
            cohort_path,
            split_path,
            baseline_output_dir,
            args.skip_existing_models,
        )
        stage_bar.update(1)

        stage_bar.set_postfix_str("training reweighted mitigation")
        reweighted_checkpoint = _train_reweighted_model(
            cfg,
            label,
            cohort_path,
            split_path,
            reweighted_output_dir,
            args.skip_existing_models,
        )
        stage_bar.update(1)

        stage_bar.set_postfix_str("held-out evaluation at threshold 0.5")
        _, _, baseline_test_predictions, reweighted_test_predictions = _evaluate_heldout(
            cfg,
            label,
            cohort_path,
            split_path,
            baseline_checkpoint,
            reweighted_checkpoint,
            tables_dir,
            plots_dir,
        )
        stage_bar.update(1)

        stage_bar.set_postfix_str("paired stats at threshold 0.5")
        print(f"[{label}] Running paired statistical tests at threshold 0.5...")
        _write_paired_significance(root, tables_dir)
        stage_bar.update(1)

        stage_bar.set_postfix_str("validation threshold sweep")
        print(f"[{label}] Running validation-selected threshold sweep...")
        _run_threshold_analysis(
            cfg,
            label,
            cohort_path,
            split_path,
            baseline_checkpoint,
            reweighted_checkpoint,
            baseline_test_predictions,
            reweighted_test_predictions,
            tables_dir,
        )
        stage_bar.update(1)

        stage_bar.set_postfix_str("paired stats with validation thresholds")
        print(f"[{label}] Running paired statistical tests with validation-selected thresholds...")
        _write_paired_significance(root, tables_dir / "val_threshold")
        stage_bar.update(1)

    print(f"[{label}] Complete. Results: {disease_outputs_dir}")


def main() -> None:
    args = _parse_args()
    root = project_root()
    cfg = load_config()

    labels = args.labels or list(cfg["labels"].keys())
    unknown_labels = sorted(set(labels).difference(cfg["labels"].keys()))
    if unknown_labels:
        raise ValueError(f"Unknown labels requested: {unknown_labels}")

    for label in tqdm(labels, desc="All diagnoses", unit="disease"):
        _run_label(cfg, root, label, args)

    print("\nAll requested diagnoses complete.")


if __name__ == "__main__":
    main()
