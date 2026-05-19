import numpy as np
import pandas as pd

from mimic_fairness.paths import load_config, project_root


def main() -> None:
    root = project_root()
    cfg = load_config()

    label_name = cfg["active_label"]
    results_dir = root / cfg["paths"]["outputs_dir"] / "by_disease" / label_name / "tables"
    baseline_path = results_dir / "baseline_fairness_results.parquet"
    reweighted_path = results_dir / "reweighted_fairness_results.parquet"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Exploratory baseline results not found at {baseline_path}")
    if not reweighted_path.exists():
        raise FileNotFoundError(f"Exploratory reweighted results not found at {reweighted_path}")

    baseline_df = pd.read_parquet(baseline_path)
    reweighted_df = pd.read_parquet(reweighted_path)

    merged = baseline_df.merge(
        reweighted_df,
        on="fairness_group",
        suffixes=("_baseline", "_reweighted"),
        validate="one_to_one",
    )

    merged["fnr_delta"] = merged["fnr_reweighted"] - merged["fnr_baseline"]
    merged["accuracy_delta"] = merged["accuracy_reweighted"] - merged["accuracy_baseline"]
    merged["auc_delta"] = merged["auc_reweighted"] - merged["auc_baseline"]

    output_path = results_dir / "exploratory_full_cohort_metric_deltas.csv"
    merged.to_csv(output_path, index=False)

    print("=" * 80)
    print("EXPLORATORY FULL-COHORT AGGREGATE COMPARISON")
    print("=" * 80)
    print(
        "These are descriptive aggregate deltas on full-cohort outputs. "
        "They are not paired significance tests and should not be used for "
        "final held-out claims. Use scripts/10_paired_statistical_tests.py "
        "after held-out test predictions are generated."
    )
    print(
        merged[
            [
                "fairness_group",
                "n_samples_baseline",
                "fnr_baseline",
                "fnr_reweighted",
                "fnr_delta",
                "accuracy_baseline",
                "accuracy_reweighted",
                "accuracy_delta",
                "auc_baseline",
                "auc_reweighted",
                "auc_delta",
            ]
        ].replace({np.nan: None}).to_string(index=False)
    )
    print(f"\nSaved exploratory aggregate deltas to {output_path}")


if __name__ == "__main__":
    main()
