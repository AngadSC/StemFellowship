import pandas as pd
from mimic_fairness.paths import load_config, project_root
from mimic_fairness.train import train_model_with_weights
from mimic_fairness.mitigation import compute_subgroup_weights


def main():
    root = project_root()
    cfg = load_config()

    label_name = cfg["active_label"]
    cohort_path = root / cfg["paths"]["interim_dir"] / f"{label_name}_cohort.parquet"
    split_path = root / cfg["paths"]["interim_dir"] / f"{label_name}_splits.parquet"

    df = pd.read_parquet(cohort_path)
    splits = pd.read_parquet(split_path)
    df_with_splits = df.merge(
        splits[["SUBJECT_ID", "HADM_ID", "split"]],
        on=["SUBJECT_ID", "HADM_ID"],
        how="left",
        validate="one_to_one",
    )
    if df_with_splits["split"].isna().any():
        raise ValueError("Some cohort rows are missing from the split file.")
    train_df = df_with_splits[df_with_splits["split"] == "train"].copy()

    sample_weights = compute_subgroup_weights(
        train_df,
        label_column=label_name,
        fairness_group_column="fairness_group",
        upweight_factor=2.0,
    )

    output_dir = root / cfg["paths"]["models_dir"] / "reweighted"

    best_model_path = train_model_with_weights(
        cohort_path=str(cohort_path),
        split_path=str(split_path),
        model_name=cfg["model"]["base_model"],
        max_length=cfg["cohort"]["max_note_length"],
        label_column=label_name,
        output_dir=str(output_dir),
        sample_weights=sample_weights,
        num_epochs=cfg["model"]["num_epochs"],
        batch_size=cfg["model"]["batch_size"],
        learning_rate=cfg["model"]["learning_rate"],
        random_seed=cfg["model"]["random_seed"],
    )

    print(f"Training complete. Reweighted model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
