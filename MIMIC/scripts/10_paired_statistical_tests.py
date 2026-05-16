from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from mimic_fairness.paths import load_config, project_root


KEY_COLUMNS = ["SUBJECT_ID", "HADM_ID"]
ALPHA = 0.05


def _load_paired_test_predictions(results_dir) -> pd.DataFrame:
    baseline_path = results_dir / "heldout_test_baseline_predictions.parquet"
    reweighted_path = results_dir / "heldout_test_reweighted_predictions.parquet"

    if not baseline_path.exists():
        baseline_path = results_dir / "test_baseline_predictions.parquet"
    if not reweighted_path.exists():
        reweighted_path = results_dir / "test_reweighted_predictions.parquet"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline test predictions not found at {baseline_path}")
    if not reweighted_path.exists():
        raise FileNotFoundError(f"Reweighted test predictions not found at {reweighted_path}")

    baseline = pd.read_parquet(baseline_path).rename(
        columns={
            "y_true": "y_true_baseline",
            "y_pred": "y_pred_baseline",
            "y_prob": "y_prob_baseline",
        }
    )
    reweighted = pd.read_parquet(reweighted_path).rename(
        columns={
            "y_true": "y_true_reweighted",
            "y_pred": "y_pred_reweighted",
            "y_prob": "y_prob_reweighted",
        }
    )

    if "split" in baseline.columns and set(baseline["split"].unique()) != {"test"}:
        raise ValueError("Baseline predictions include non-test rows.")
    if "split" in reweighted.columns and set(reweighted["split"].unique()) != {"test"}:
        raise ValueError("Reweighted predictions include non-test rows.")

    if baseline["HADM_ID"].duplicated().any():
        raise ValueError("Baseline test predictions contain duplicate HADM_ID values.")
    if reweighted["HADM_ID"].duplicated().any():
        raise ValueError("Reweighted test predictions contain duplicate HADM_ID values.")

    baseline_keys = baseline[KEY_COLUMNS]
    reweighted_keys = reweighted[KEY_COLUMNS]
    same_order = baseline_keys.equals(reweighted_keys)

    paired = baseline.merge(
        reweighted[
            KEY_COLUMNS
            + ["y_true_reweighted", "y_pred_reweighted", "y_prob_reweighted"]
        ],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    if len(paired) != len(baseline) or len(paired) != len(reweighted):
        raise ValueError(
            "Baseline and reweighted prediction files are not paired on the same test rows. "
            f"baseline={len(baseline)}, reweighted={len(reweighted)}, paired={len(paired)}"
        )

    if not (paired["y_true_baseline"] == paired["y_true_reweighted"]).all():
        raise ValueError("Baseline and reweighted predictions disagree on true labels.")

    if same_order:
        print("Validated paired test predictions: identical SUBJECT_ID/HADM_ID order.")
    else:
        print(
            "Validated paired test predictions: same SUBJECT_ID/HADM_ID set; "
            "paired tests merged by SUBJECT_ID/HADM_ID."
        )

    paired["y_true"] = paired["y_true_baseline"]
    return paired


def _paired_binomial_test(
    baseline_event: pd.Series,
    reweighted_event: pd.Series,
) -> dict:
    baseline_event = baseline_event.astype(bool).to_numpy()
    reweighted_event = reweighted_event.astype(bool).to_numpy()

    baseline_only = int(np.sum(baseline_event & ~reweighted_event))
    reweighted_only = int(np.sum(~baseline_event & reweighted_event))
    discordant = baseline_only + reweighted_only

    p_value = np.nan
    if discordant > 0:
        p_value = binomtest(
            min(baseline_only, reweighted_only),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return {
        "baseline_only_discordant": baseline_only,
        "reweighted_only_discordant": reweighted_only,
        "discordant_count": discordant,
        "p_value": p_value,
    }


def _two_proportion_z_test(
    baseline_count: int,
    baseline_n: int,
    reweighted_count: int,
    reweighted_n: int,
) -> float:
    if baseline_n == 0 or reweighted_n == 0:
        return np.nan

    pooled = (baseline_count + reweighted_count) / (baseline_n + reweighted_n)
    se = sqrt(pooled * (1.0 - pooled) * ((1.0 / baseline_n) + (1.0 / reweighted_n)))
    if se == 0:
        return np.nan

    z_score = (baseline_count / baseline_n - reweighted_count / reweighted_n) / se
    return 2.0 * (1.0 - norm.cdf(abs(z_score)))


def _wilson_ci(count: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)

    z = norm.ppf(1.0 - alpha / 2.0)
    p_hat = count / n
    denominator = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2.0 * n)) / denominator
    margin = z * sqrt((p_hat * (1.0 - p_hat) / n) + (z**2 / (4.0 * n**2))) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def _format_p_value(p_value: float, scientific: bool = False) -> str:
    if pd.isna(p_value):
        return "nan"
    if scientific:
        return f"{p_value:.4e}"
    return f"{p_value:.4f}"


def _significance_label(p_value: float) -> str:
    return "Yes" if not pd.isna(p_value) and p_value < ALPHA else "No"


def _append_event_rate_result(
    rows: list[dict],
    metric: str,
    group: str,
    df: pd.DataFrame,
    baseline_event: pd.Series,
    reweighted_event: pd.Series,
    event_name: str,
    reported_rate_higher_is_better: bool,
) -> None:
    if df.empty:
        return

    test = _paired_binomial_test(baseline_event, reweighted_event)
    baseline_event_rate = baseline_event.mean()
    reweighted_event_rate = reweighted_event.mean()

    if reported_rate_higher_is_better:
        baseline_rate = 1.0 - baseline_event_rate
        reweighted_rate = 1.0 - reweighted_event_rate
    else:
        baseline_rate = baseline_event_rate
        reweighted_rate = reweighted_event_rate

    rows.append(
        {
            "metric": metric,
            "fairness_group": group,
            "n_eligible": len(df),
            "baseline_event": event_name,
            "reweighted_event": event_name,
            "baseline_event_count": int(baseline_event.sum()),
            "reweighted_event_count": int(reweighted_event.sum()),
            "baseline_rate": baseline_rate,
            "reweighted_rate": reweighted_rate,
            "delta_reweighted_minus_baseline": reweighted_rate - baseline_rate,
            **test,
        }
    )


def _append_accuracy_result(rows: list[dict], group: str, df: pd.DataFrame) -> None:
    _append_event_rate_result(
        rows=rows,
        metric="accuracy",
        group=group,
        df=df,
        baseline_event=df["y_pred_baseline"] != df["y_true"],
        reweighted_event=df["y_pred_reweighted"] != df["y_true"],
        event_name="incorrect_prediction",
        reported_rate_higher_is_better=True,
    )


def _append_fnr_result(rows: list[dict], group: str, df: pd.DataFrame) -> None:
    positives = df[df["y_true"] == 1]
    if positives.empty:
        return

    _append_event_rate_result(
        rows=rows,
        metric="fnr",
        group=group,
        df=positives,
        baseline_event=positives["y_pred_baseline"] == 0,
        reweighted_event=positives["y_pred_reweighted"] == 0,
        event_name="false_negative",
        reported_rate_higher_is_better=False,
    )


def _append_fpr_result(rows: list[dict], group: str, df: pd.DataFrame) -> None:
    negatives = df[df["y_true"] == 0]
    if negatives.empty:
        return

    _append_event_rate_result(
        rows=rows,
        metric="fpr",
        group=group,
        df=negatives,
        baseline_event=negatives["y_pred_baseline"] == 1,
        reweighted_event=negatives["y_pred_reweighted"] == 1,
        event_name="false_positive",
        reported_rate_higher_is_better=False,
    )


def _build_fnr_ztest_rows(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for group, group_df in paired.groupby("fairness_group"):
        positives = group_df[group_df["y_true"] == 1]
        n_positive = len(positives)
        baseline_fn = int((positives["y_pred_baseline"] == 0).sum())
        reweighted_fn = int((positives["y_pred_reweighted"] == 0).sum())

        baseline_fnr = baseline_fn / n_positive if n_positive else np.nan
        reweighted_fnr = reweighted_fn / n_positive if n_positive else np.nan
        p_value = _two_proportion_z_test(
            baseline_count=baseline_fn,
            baseline_n=n_positive,
            reweighted_count=reweighted_fn,
            reweighted_n=n_positive,
        )
        baseline_ci_low, baseline_ci_high = _wilson_ci(baseline_fn, n_positive)
        reweighted_ci_low, reweighted_ci_high = _wilson_ci(reweighted_fn, n_positive)

        rows.append(
            {
                "fairness_group": group,
                "n_positive": n_positive,
                "baseline_false_negatives": baseline_fn,
                "reweighted_false_negatives": reweighted_fn,
                "baseline_fnr": baseline_fnr,
                "reweighted_fnr": reweighted_fnr,
                "fnr_delta_reweighted_minus_baseline": reweighted_fnr - baseline_fnr,
                "p_value": p_value,
                "baseline_fnr_ci_low": baseline_ci_low,
                "baseline_fnr_ci_high": baseline_ci_high,
                "reweighted_fnr_ci_low": reweighted_ci_low,
                "reweighted_fnr_ci_high": reweighted_ci_high,
            }
        )

    return pd.DataFrame(rows).sort_values("fairness_group")


def _metric_row(results: pd.DataFrame, metric: str, group: str = "overall") -> pd.Series:
    matches = results[
        (results["metric"] == metric) & (results["fairness_group"] == group)
    ]
    if matches.empty:
        raise ValueError(f"Missing {metric!r} significance result for group {group!r}.")
    return matches.iloc[0]


def _build_statistical_report(
    results: pd.DataFrame,
    fnr_ztests: pd.DataFrame,
) -> str:
    overall_fnr = _metric_row(results, "fnr")
    overall_accuracy = _metric_row(results, "accuracy")

    baseline_fnr_ratio = (
        fnr_ztests["baseline_fnr"].max() / fnr_ztests["baseline_fnr"].min()
    )
    reweighted_fnr_ratio = (
        fnr_ztests["reweighted_fnr"].max() / fnr_ztests["reweighted_fnr"].min()
    )
    ratio_improvement = baseline_fnr_ratio - reweighted_fnr_ratio

    lines = []
    lines.append("=" * 80)
    lines.append("STATISTICAL SIGNIFICANCE TESTS")
    lines.append("=" * 80)
    lines.append("")

    lines.append("1. FNR Difference Per Group (Two-Proportion Z-tests)")
    lines.append("-" * 80)
    lines.append(f"{'Group':<20}{'FNR Baseline':<16}{'FNR Reweighted':<16}{'p-value':<12}")
    lines.append("-" * 80)
    for row in fnr_ztests.itertuples(index=False):
        lines.append(
            f"{row.fairness_group:<20}"
            f"{row.baseline_fnr:<16.4f}"
            f"{row.reweighted_fnr:<16.4f}"
            f"{_format_p_value(row.p_value):<12}"
        )
    lines.append("")

    lines.append("2. Confidence Intervals for FNR (95%, Wilson)")
    lines.append("-" * 80)
    lines.append(f"{'Group':<20}{'Baseline FNR CI':<32}{'Reweighted FNR CI':<32}")
    lines.append("-" * 80)
    for row in fnr_ztests.itertuples(index=False):
        baseline_ci = f"({row.baseline_fnr_ci_low:.4f}, {row.baseline_fnr_ci_high:.4f})"
        reweighted_ci = (
            f"({row.reweighted_fnr_ci_low:.4f}, {row.reweighted_fnr_ci_high:.4f})"
        )
        lines.append(f"{row.fairness_group:<20}{baseline_ci:<32}{reweighted_ci:<32}")
    lines.append("")

    lines.append("3. Overall FNR Comparison (Paired Exact Test)")
    lines.append("-" * 80)
    lines.append(f"Baseline FNR:          {overall_fnr['baseline_rate']:.4f}")
    lines.append(f"Reweighted FNR:        {overall_fnr['reweighted_rate']:.4f}")
    lines.append(
        "FNR Reduction:         "
        f"{overall_fnr['baseline_rate'] - overall_fnr['reweighted_rate']:.4f}"
    )
    lines.append(f"p-value:               {_format_p_value(overall_fnr['p_value'], scientific=True)}")
    lines.append(
        f"Significant (alpha=0.05):  {_significance_label(overall_fnr['p_value'])}"
    )
    lines.append("")

    lines.append("4. Accuracy Comparison (Paired Exact Test)")
    lines.append("-" * 80)
    lines.append(f"Baseline Accuracy:     {overall_accuracy['baseline_rate']:.4f}")
    lines.append(f"Reweighted Accuracy:   {overall_accuracy['reweighted_rate']:.4f}")
    lines.append(
        "Accuracy Change:       "
        f"{overall_accuracy['reweighted_rate'] - overall_accuracy['baseline_rate']:.4f}"
    )
    lines.append(
        f"p-value:               {_format_p_value(overall_accuracy['p_value'])}"
    )
    significance = _significance_label(overall_accuracy["p_value"])
    if significance == "No":
        significance = "No (likely due to chance)"
    lines.append(f"Significant (alpha=0.05):  {significance}")
    lines.append("")

    lines.append("5. Fairness Metric: Ratio of Max to Min FNR")
    lines.append("-" * 80)
    lines.append(f"Baseline FNR Ratio (max/min):     {baseline_fnr_ratio:.4f}")
    lines.append(f"Reweighted FNR Ratio (max/min):   {reweighted_fnr_ratio:.4f}")
    lines.append(f"Improvement:                      {ratio_improvement:.4f}")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def main() -> None:
    root = project_root()
    cfg = load_config()

    results_dir = root / cfg["paths"]["outputs_dir"] / "tables"
    paired = _load_paired_test_predictions(results_dir)

    rows = []
    _append_accuracy_result(rows, "overall", paired)
    _append_fnr_result(rows, "overall", paired)
    _append_fpr_result(rows, "overall", paired)

    for group, group_df in paired.groupby("fairness_group"):
        _append_accuracy_result(rows, group, group_df)
        _append_fnr_result(rows, group, group_df)
        _append_fpr_result(rows, group, group_df)

    results = pd.DataFrame(rows)
    output_path = results_dir / "heldout_test_paired_significance.csv"
    results.to_csv(output_path, index=False)
    results.to_csv(results_dir / "test_paired_significance.csv", index=False)

    fnr_ztests = _build_fnr_ztest_rows(paired)
    fnr_ztests.to_csv(results_dir / "heldout_test_fnr_group_z_tests.csv", index=False)
    fnr_ztests[
        [
            "fairness_group",
            "baseline_fnr",
            "baseline_fnr_ci_low",
            "baseline_fnr_ci_high",
            "reweighted_fnr",
            "reweighted_fnr_ci_low",
            "reweighted_fnr_ci_high",
        ]
    ].to_csv(results_dir / "heldout_test_fnr_confidence_intervals.csv", index=False)

    report = _build_statistical_report(results, fnr_ztests)
    report_path = results_dir / "heldout_test_statistical_significance_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved paired significance results to {output_path}")
    print(f"Saved FNR group z-tests to {results_dir / 'heldout_test_fnr_group_z_tests.csv'}")
    print(
        "Saved FNR confidence intervals to "
        f"{results_dir / 'heldout_test_fnr_confidence_intervals.csv'}"
    )
    print(f"Saved formatted report to {report_path}")


if __name__ == "__main__":
    main()
