import pandas as pd

from mimic_fairness.paths import load_config, project_root


def main():
    root = project_root()
    cfg = load_config()

    label = cfg["active_label"]
    cohort_path = root / cfg["paths"]["interim_dir"] / f"{label}_cohort.parquet"

    df = pd.read_parquet(cohort_path)

    summary = (
        df.groupby("fairness_group")
        .agg(
            n=("HADM_ID", "count"),
            positives=(label, "sum"),
            positive_rate=(label, "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )

    print(summary)

    out_dir = root / cfg["paths"]["outputs_dir"] / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / f"{label}_group_counts.csv", index=False)


if __name__ == "__main__":
    main()
