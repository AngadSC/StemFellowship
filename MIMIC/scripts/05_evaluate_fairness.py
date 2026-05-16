from mimic_fairness.paths import load_config, project_root
from mimic_fairness.evaluate import evaluate_fairness


def main():
    root = project_root()
    cfg = load_config()

    label_name = cfg["active_label"]
    cohort_path = root / cfg["paths"]["interim_dir"] / f"{label_name}_cohort.parquet"

    models_dir = root / cfg["paths"]["models_dir"]
    best_model_path = models_dir / "best_model"

    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    output_dir = root / cfg["paths"]["outputs_dir"] / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_fairness_results.parquet"

    results_df = evaluate_fairness(
        cohort_path=str(cohort_path),
        checkpoint_path=str(best_model_path),
        max_length=cfg["cohort"]["max_note_length"],
        label_column=label_name,
        batch_size=cfg["model"]["batch_size"],
    )

    results_df.to_parquet(output_path, index=False)
    print(f"Saved baseline fairness results to {output_path}")
    print("\nBaseline Fairness Metrics:")
    print(results_df.to_string())


if __name__ == "__main__":
    main()
