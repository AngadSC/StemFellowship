import pandas as pd
from mimic_fairness.paths import load_config, project_root
from mimic_fairness.train import train_model_with_weights
from mimic_fairness.mitigation import compute_subgroup_weights


def _factor_dir_name(upweight_factor: float) -> str:
    factor_str = f"{upweight_factor:.2f}".rstrip("0").rstrip(".")
    return f"factor_{factor_str.replace('.', '_')}"


def main():
    root = project_root()
    cfg = load_config()

    label_name = cfg["active_label"]
    cohort_path = root / cfg["paths"]["interim_dir"] / f"{label_name}_cohort.parquet"
    split_path = root / cfg["paths"]["interim_dir"] / f"{label_name}_splits.parquet"

    factor_grid = cfg["model"].get("reweighted_factor_grid", [2.0])
    if isinstance(factor_grid, (int, float)):
        factor_grid = [float(factor_grid)]
    factor_grid = [float(factor) for factor in factor_grid]

    default_factor = float(cfg["model"].get("reweighted_default_factor", 2.0))
    if not factor_grid:
        factor_grid = [default_factor]
    max_weight = float(cfg["model"].get("reweighted_max_weight", 10.0))
    normalize_weights = bool(cfg["model"].get("reweighted_normalize_weights", True))

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

    base_output_dir = root / cfg["paths"]["models_dir"] / "by_disease" / label_name / "reweighted"

    for upweight_factor in factor_grid:
        scope = "default" if upweight_factor == default_factor else _factor_dir_name(upweight_factor)
        output_dir = base_output_dir if scope == "default" else base_output_dir / scope
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Training reweighted model with upweight_factor={upweight_factor} -> {output_dir}")
        sample_weights = compute_subgroup_weights(
            train_df,
            label_column=label_name,
            fairness_group_column="fairness_group",
            upweight_factor=upweight_factor,
            max_weight=max_weight,
            normalize=normalize_weights,
        )

        summary_path = output_dir / "reweighted_weight_summary.csv"
        summary = (
            train_df.assign(weight=sample_weights)
            .groupby("fairness_group")
            .agg(
                n=("HADM_ID", "count"),
                positives=(label_name, "sum"),
                positive_rate=(label_name, "mean"),
                mean_weight=("weight", "mean"),
                min_weight=("weight", "min"),
                max_weight=("weight", "max"),
                total_weight=("weight", "sum"),
            )
            .reset_index()
            .sort_values("positive_rate")
        )
        summary.to_csv(summary_path, index=False)

        best_model_path = train_model_with_weights(
            cohort_path=str(cohort_path),
            split_path=str(split_path),
            model_name=cfg["model"]["base_model"],
            max_length=cfg["cohort"]["max_note_length"],
            label_column=label_name,
            output_dir=str(output_dir),
            sample_weights=sample_weights,
            num_epochs=cfg["model"]["num_epochs"],
            batch_size=cfg["model"].get("train_batch_size", cfg["model"]["batch_size"]),
            learning_rate=cfg["model"]["learning_rate"],
            random_seed=cfg["model"]["random_seed"],
            gradient_accumulation_steps=cfg["model"].get("gradient_accumulation_steps", 1),
            optimizer_step_sleep_seconds=cfg["model"].get("optimizer_step_sleep_seconds", 0.0),
        )

        print(f"Training complete for upweight_factor={upweight_factor}. Model saved to: {best_model_path}")
        print(f"Weight summary saved to: {summary_path}\n")
