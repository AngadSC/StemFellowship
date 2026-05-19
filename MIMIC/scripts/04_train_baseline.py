from mimic_fairness.paths import load_config, project_root
from mimic_fairness.train import train_model


def main():
    root = project_root()
    cfg = load_config()

    label_name = cfg["active_label"]
    cohort_path = root / cfg["paths"]["interim_dir"] / f"{label_name}_cohort.parquet"
    split_path = root / cfg["paths"]["interim_dir"] / f"{label_name}_splits.parquet"

    output_dir = root / cfg["paths"]["models_dir"] / "by_disease" / label_name / "baseline"

    best_model_path = train_model(
        cohort_path=str(cohort_path),
        split_path=str(split_path),
        model_name=cfg["model"]["base_model"],
        max_length=cfg["cohort"]["max_note_length"],
        label_column=label_name,
        output_dir=str(output_dir),
        num_epochs=cfg["model"]["num_epochs"],
        batch_size=cfg["model"].get("train_batch_size", cfg["model"]["batch_size"]),
        learning_rate=cfg["model"]["learning_rate"],
        random_seed=cfg["model"]["random_seed"],
        gradient_accumulation_steps=cfg["model"].get("gradient_accumulation_steps", 1),
        optimizer_step_sleep_seconds=cfg["model"].get("optimizer_step_sleep_seconds", 0.0),
    )

    print(f"Training complete. Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
