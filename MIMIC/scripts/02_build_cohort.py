from pathlib import Path

from mimic_fairness.paths import load_config, project_root
from mimic_fairness.build_cohort import build_cohort


def main():
    root = project_root()
    cfg = load_config()

    out_dir = root / cfg["paths"]["interim_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_cohort(cfg, root)

    output_path = out_dir / f"{cfg['label']['task_name']}_cohort.parquet"
    df.to_parquet(output_path, index=False)

    print(f"Saved cohort to: {output_path}")
    print(df.shape)
    print(df[cfg["label"]["task_name"]].value_counts())
    print(df["fairness_group"].value_counts())


if __name__ == "__main__":
    main()
