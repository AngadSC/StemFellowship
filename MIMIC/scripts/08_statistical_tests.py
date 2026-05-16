import numpy as np
import pandas as pd
from scipy.stats import binomtest

from mimic_fairness.paths import load_config, project_root


KEY_COLUMNS = ["SUBJECT_ID", "HADM_ID"]


def mcnemar_exact(event_baseline: pd.Series, event_reweighted: pd.Series) -> dict:
    """
    Exact McNemar test for paired binary events on the same admissions.

    The event can be "incorrect prediction" for accuracy or "false negative"
    among positives for FNR. b counts baseline-only events; c counts
    reweighted-only events.
    """
    baseline = event_baseline.astype(bool).to_numpy()
    reweighted = event_reweighted.astype(bool).to_numpy()

    baseline_only = int(np.sum(baseline & ~reweighted))
    reweighted_only = int(np.sum(~baseline & reweighted))
    discordant = baseline_only + reweighted_only

    p_value = np.nan
    if discordant > 0:
        p_value = binomtest(
            min(baseline_only, reweighted_only),
            discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return {
        "baseline_only_events": baseline_only,
        "reweighted_only_events": reweighted_only,
        "discordant_pairs": discordant,
        "p_value": p_value,
    }


def load_paired_predictions(results_dir, label_name: str) -> pd.DataFrame:
    baseline_path = results_dir / "test_baseline_predictions.parquet"
    reweighted_path = results_dir / "test_reweighted_predictions.parquet"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline test predictions not found at {baseline_path}")
    if not reweighted_path.exists():
        raise FileNotFoundError(f"Reweighted test predictions not found at {reweighted_path}")

    baseline = pd.read_parquet(baseline_path)
    reweighted = pd.read_parquet(reweighted_path)

    baseline = baseline.rename(
        columns={
            "y_true": "y_true_baseline",
            "y_pred": "y_pred_baseline",
            "y_prob": "y_prob_baseline",
        }
    )
    reweighted = reweighted.rename(
        columns={
            "y_true": "y_true_reweighted",
            "y_pred": "y_pred_reweighted",
            "y_prob": "y_prob_reweighted",
        }
    )

    merged = baseline.merge(
        reweighted[
            KEY_COLUMNS
            + ["y_true_reweighted", "y_pred_reweighted", "y_prob_reweighted"]
        ],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(baseline) or len(merged) != len(reweighted):
        raise ValueError(
            "Prediction files do not contain the same paired admissions. "
            f"baseline={len(baseline)}, reweighted={len(reweighted)}, paired={len(merged)}"
        )

    if not (merged["y_true_baseline"] == merged["y_true_reweighted"]).all():
        raise ValueError("Baseline and reweighted prediction files disagree on y_true.")

    merged["y_true"] = merged["y_true_baseline"]
    if label_name in merged.columns:
        label_mismatch = merged[label_name] != merged["y_true"]
        if label_mismatch.any():
            raise ValueError("Stored label column disagrees with y_true.")

    return merged


def add_result(rows: list[dict], metric: str, group: str, subset: pd.DataFrame) -> None:
    if subset.empty:
        return

    if metric == "accuracy":
        baseline_event = subset["y_pred_baseline"] != subset["y_true"]
        reweighted_event = subset["y_pred_reweighted"] != subset["y_true"]
        baseline_value = 1.0 - baseline_event.mean()
        reweighted_value = 1.0 - reweighted_event.mean()
    elif metric == "fnr":
        positives = subset[subset["y_true"] == 1]
        if positives.empty:
            return
        baseline_event = positives["y_pred_baseline"] == 0
        reweighted_event = positives["y_pred_reweighted"] == 0
        baseline_value = baseline_event.mean()
        reweighted_value = reweighted_event.mean()
        subset = positives
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    test_result = mcnemar_exact(baseline_event, reweighted_event)
    rows.append(
        {
            "metric": metric,
            "fairness_group": group,
            "n_paired": len(subset),
            "baseline_value": baseline_value,
            "reweighted_value": reweighted_value,
            "delta_reweighted_minus_baseline": reweighted_value - baseline_value,
            **test_result,
        }
    )


def main() -> None:
    root = project_root()
    cfg = load_config()

    label_name = cfg["active_label"]
    results_dir = root / cfg["paths"]["outputs_dir"] / "tables"

    paired = load_paired_predictions(results_dir, label_name)

    rows = []
    add_result(rows, "accuracy", "overall", paired)
    add_result(rows, "fnr", "overall", paired)

    for group, group_df in paired.groupby("fairness_group"):
        add_result(rows, "accuracy", group, group_df)
        add_result(rows, "fnr", group, group_df)

    results = pd.DataFrame(rows)
    output_path = results_dir / "paired_statistical_tests.csv"
    results.to_csv(output_path, index=False)

    print("=" * 80)
    print("PAIRED STATISTICAL TESTS ON HELD-OUT TEST ADMISSIONS")
    print("=" * 80)
    print(
        results[
            [
                "metric",
                "fairness_group",
                "n_paired",
                "baseline_value",
                "reweighted_value",
                "delta_reweighted_minus_baseline",
                "baseline_only_events",
                "reweighted_only_events",
                "discordant_pairs",
                "p_value",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved paired statistical tests to {output_path}")


if __name__ == "__main__":
    main()
