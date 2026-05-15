import pandas as pd
from pathlib import Path
from mimic_fairness.paths import load_config, project_root
from mimic_fairness.dataset import MIMICDataset
from mimic_fairness.preprocessing import load_tokenizer
from mimic_fairness.train import train_model_with_weights
from mimic_fairness.mitigation import compute_subgroup_weights


def main():
    root = project_root()
    cfg = load_config()

    label_name = cfg["active_label"]
    cohort_path = root / cfg["paths"]["interim_dir"] / f"{label_name}_cohort.parquet"

    df = pd.read_parquet(cohort_path)

    sample_weights = compute_subgroup_weights(
        df,
        label_column=label_name,
        fairness_group_column="fairness_group",
        upweight_factor=2.0,
    )

    output_dir = root / cfg["paths"]["models_dir"] / "reweighted"

    best_model_path = train_model_with_weights(
        cohort_path=str(cohort_path),
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
