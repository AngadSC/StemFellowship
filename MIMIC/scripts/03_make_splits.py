import pandas as pd
from sklearn.model_selection import train_test_split

from mimic_fairness.paths import load_config, project_root


SPLIT_ORDER = ["train", "val", "test"]


def _stratify_or_none(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return labels


def _make_subject_splits(
    patient_labels: pd.DataFrame,
    random_seed: int,
) -> pd.DataFrame:
    subject_ids = patient_labels["SUBJECT_ID"].to_numpy()
    labels = patient_labels["patient_label"]

    train_subjects, holdout_subjects = train_test_split(
        subject_ids,
        test_size=0.30,
        random_state=random_seed,
        stratify=_stratify_or_none(labels),
    )

    holdout_labels = patient_labels.set_index("SUBJECT_ID").loc[holdout_subjects, "patient_label"]
    stratify_holdout = _stratify_or_none(holdout_labels)

    val_subjects, test_subjects = train_test_split(
        holdout_subjects,
        test_size=0.50,
        random_state=random_seed,
        stratify=stratify_holdout,
    )

    splits = pd.concat(
        [
            pd.DataFrame({"SUBJECT_ID": train_subjects, "split": "train"}),
            pd.DataFrame({"SUBJECT_ID": val_subjects, "split": "val"}),
            pd.DataFrame({"SUBJECT_ID": test_subjects, "split": "test"}),
        ],
        ignore_index=True,
    )

    return splits


def _validate_patient_disjointness(split_rows: pd.DataFrame) -> None:
    split_counts = split_rows.groupby("SUBJECT_ID")["split"].nunique()
    overlapping_subjects = split_counts[split_counts > 1]
    if not overlapping_subjects.empty:
        examples = overlapping_subjects.index[:10].tolist()
        raise ValueError(
            "SUBJECT_ID leakage detected across splits. "
            f"Example overlapping SUBJECT_ID values: {examples}"
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


def main() -> None:
    root = project_root()
    cfg = load_config()

    label = cfg["active_label"]
    interim_dir = root / cfg["paths"]["interim_dir"]
    cohort_path = interim_dir / f"{label}_cohort.parquet"
    splits_path = interim_dir / f"{label}_splits.parquet"
    summary_path = interim_dir / f"{label}_split_summary.csv"

    df = pd.read_parquet(cohort_path)
    required_columns = {"SUBJECT_ID", "HADM_ID", label, "fairness_group"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Cohort is missing required columns: {sorted(missing_columns)}")

    patient_labels = (
        df.groupby("SUBJECT_ID", as_index=False)[label]
        .max()
        .rename(columns={label: "patient_label"})
    )

    subject_splits = _make_subject_splits(
        patient_labels=patient_labels,
        random_seed=cfg["model"]["random_seed"],
    )

    split_rows = df.merge(subject_splits, on="SUBJECT_ID", how="left", validate="many_to_one")
    if split_rows["split"].isna().any():
        raise ValueError("Some cohort rows were not assigned to a split.")

    _validate_patient_disjointness(split_rows)

    output_splits = split_rows[["SUBJECT_ID", "HADM_ID", "split"]].copy()
    output_splits.to_parquet(splits_path, index=False)

    summary = _summarize_splits(split_rows, label)
    summary.to_csv(summary_path, index=False)

    print(f"Saved split labels to {splits_path}")
    print(f"Saved split summary to {summary_path}")
    print("\nSplit summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
