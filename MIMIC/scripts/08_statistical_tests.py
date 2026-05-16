import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from pathlib import Path
from mimic_fairness.paths import load_config, project_root


def fnr_confidence_interval(fn_count, tp_count, confidence=0.95):
    """Compute Wilson score confidence interval for FNR."""
    from scipy.stats import binom
    n = fn_count + tp_count
    if n == 0:
        return (0, 0)
    p = fn_count / n
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0, center - margin), min(1, center + margin))


def fnr_z_test(fn1, tp1, fn2, tp2):
    """Z-test for difference in FNR between two groups."""
    n1 = fn1 + tp1
    n2 = fn2 + tp2
    if n1 == 0 or n2 == 0:
        return np.nan

    p1 = fn1 / n1
    p2 = fn2 / n2
    se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)

    if se == 0:
        return np.nan

    z = (p1 - p2) / se
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return p_value


def main():
    root = project_root()
    cfg = load_config()

    results_dir = root / cfg["paths"]["outputs_dir"] / "tables"

    baseline_path = results_dir / "baseline_fairness_results.parquet"
    reweighted_path = results_dir / "reweighted_fairness_results.parquet"

    baseline_df = pd.read_parquet(baseline_path)
    reweighted_df = pd.read_parquet(reweighted_path)

    merged = baseline_df.merge(
        reweighted_df,
        on="fairness_group",
        suffixes=("_baseline", "_reweighted"),
    )

    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    print("\n1. FNR Difference Per Group (Z-tests)")
    print("-" * 80)
    print(f"{'Group':<20} {'FNR Baseline':<15} {'FNR Reweighted':<15} {'p-value':<12}")
    print("-" * 80)

    for _, row in merged.iterrows():
        group = row["fairness_group"]
        fn_base = row["fn_count_baseline"]
        tp_base = row["tp_count_baseline"]
        fn_reweight = row["fn_count_reweighted"]
        tp_reweight = row["tp_count_reweighted"]

        p_val = fnr_z_test(fn_base, tp_base, fn_reweight, tp_reweight)

        print(f"{group:<20} {row['fnr_baseline']:<15.4f} {row['fnr_reweighted']:<15.4f} {p_val:<12.4f}")

    print("\n2. Confidence Intervals for FNR (95%)")
    print("-" * 80)
    print(f"{'Group':<20} {'Baseline FNR CI':<30} {'Reweighted FNR CI':<30}")
    print("-" * 80)

    for _, row in merged.iterrows():
        group = row["fairness_group"]
        fn_base = row["fn_count_baseline"]
        tp_base = row["tp_count_baseline"]
        fn_reweight = row["fn_count_reweighted"]
        tp_reweight = row["tp_count_reweighted"]

        ci_base = fnr_confidence_interval(fn_base, tp_base)
        ci_reweight = fnr_confidence_interval(fn_reweight, tp_reweight)

        print(f"{group:<20} ({ci_base[0]:.4f}, {ci_base[1]:.4f})          ({ci_reweight[0]:.4f}, {ci_reweight[1]:.4f})")

    print("\n3. Overall FNR Comparison (Paired Z-test)")
    print("-" * 80)
    total_fn_base = merged["fn_count_baseline"].sum()
    total_tp_base = merged["tp_count_baseline"].sum()
    total_fn_reweight = merged["fn_count_reweighted"].sum()
    total_tp_reweight = merged["tp_count_reweighted"].sum()

    overall_fnr_base = total_fn_base / (total_fn_base + total_tp_base)
    overall_fnr_reweight = total_fn_reweight / (total_fn_reweight + total_tp_reweight)

    p_val_overall = fnr_z_test(total_fn_base, total_tp_base, total_fn_reweight, total_tp_reweight)

    print(f"Baseline Mean FNR:     {overall_fnr_base:.4f}")
    print(f"Reweighted Mean FNR:   {overall_fnr_reweight:.4f}")
    print(f"FNR Reduction:         {overall_fnr_base - overall_fnr_reweight:.4f}")
    print(f"p-value:               {p_val_overall:.4e}")
    print(f"Significant (α=0.05):  {'Yes' if p_val_overall < 0.05 else 'No'}")

    print("\n4. Accuracy Comparison (Z-test)")
    print("-" * 80)

    # Accuracy = (TP + TN) / total, approximate as TP / (TP + FN) for positive class
    acc_base = baseline_df["accuracy"].mean()
    acc_reweight = reweighted_df["accuracy"].mean()

    # Z-test for accuracy difference
    n_samples = baseline_df["n_samples"].sum()
    se_acc = np.sqrt(acc_base*(1-acc_base)/n_samples + acc_reweight*(1-acc_reweight)/n_samples)
    z_acc = (acc_reweight - acc_base) / se_acc if se_acc > 0 else 0

    from scipy.stats import norm
    p_acc = 2 * (1 - norm.cdf(abs(z_acc))) if not np.isnan(z_acc) else np.nan

    print(f"Baseline Accuracy:     {acc_base:.4f}")
    print(f"Reweighted Accuracy:   {acc_reweight:.4f}")
    print(f"Accuracy Change:       {acc_reweight - acc_base:.4f}")
    print(f"p-value:               {p_acc:.4f}")
    print(f"Significant (α=0.05):  {'Yes' if p_acc < 0.05 else 'No (likely due to chance)'}")

    print("\n5. Fairness Metric: Ratio of Max to Min FNR")
    print("-" * 80)
    fnr_base_vals = baseline_df.set_index("fairness_group")["fnr"]
    fnr_reweight_vals = reweighted_df.set_index("fairness_group")["fnr"]

    ratio_base = fnr_base_vals.max() / fnr_base_vals.min()
    ratio_reweight = fnr_reweight_vals.max() / fnr_reweight_vals.min()

    print(f"Baseline FNR Ratio (max/min):     {ratio_base:.4f}")
    print(f"Reweighted FNR Ratio (max/min):   {ratio_reweight:.4f}")
    print(f"Improvement:                      {ratio_base - ratio_reweight:.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
