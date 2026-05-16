import numpy as np
import pandas as pd
from scipy.stats import binomtest

from mimic_fairness.paths import load_config, project_root


KEY_COLUMNS = ["SUBJECT_ID", "HADM_ID"]


def _load_paired_test_predictions(results_dir) -> pd.DataFrame:
    baseline_path = results_dir / "test_baseline_predictions.parquet"
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


def _append_accuracy_result(rows: list[dict], group: str, df: pd.DataFrame) -> None:
    if df.empty:
        return

    baseline_incorrect = df["y_pred_baseline"] != df["y_true"]
    reweighted_incorrect = df["y_pred_reweighted"] != df["y_true"]
    test = _paired_binomial_test(baseline_incorrect, reweighted_incorrect)

    rows.append(
        {
            "metric": "accuracy",
            "fairness_group": group,
            "n_eligible": len(df),
            "baseline_event": "incorrect_prediction",
            "reweighted_event": "incorrect_prediction",
            "baseline_event_count": int(baseline_incorrect.sum()),
            "reweighted_event_count": int(reweighted_incorrect.sum()),
            "baseline_rate": 1.0 - baseline_incorrect.mean(),
            "reweighted_rate": 1.0 - reweighted_incorrect.mean(),
            "delta_reweighted_minus_baseline": (
                1.0 - reweighted_incorrect.mean()
            )
            - (1.0 - baseline_incorrect.mean()),
            **test,
        }
    )


def _append_fnr_result(rows: list[dict], group: str, df: pd.DataFrame) -> None:
    positives = df[df["y_true"] == 1]
    if positives.empty:
        return

    baseline_fn = positives["y_pred_baseline"] == 0
    reweighted_fn = positives["y_pred_reweighted"] == 0
    test = _paired_binomial_test(baseline_fn, reweighted_fn)

    rows.append(
        {
            "metric": "fnr",
            "fairness_group": group,
            "n_eligible": len(positives),
            "baseline_event": "false_negative",
            "reweighted_event": "false_negative",
            "baseline_event_count": int(baseline_fn.sum()),
            "reweighted_event_count": int(reweighted_fn.sum()),
            "baseline_rate": baseline_fn.mean(),
            "reweighted_rate": reweighted_fn.mean(),
            "delta_reweighted_minus_baseline": reweighted_fn.mean() - baseline_fn.mean(),
            **test,
        }
    )


def main() -> None:
    root = project_root()
    cfg = load_config()

    results_dir = root / cfg["paths"]["outputs_dir"] / "tables"
    paired = _load_paired_test_predictions(results_dir)

    rows = []
    _append_accuracy_result(rows, "overall", paired)
    _append_fnr_result(rows, "overall", paired)

    for group, group_df in paired.groupby("fairness_group"):
        _append_accuracy_result(rows, group, group_df)
        _append_fnr_result(rows, group, group_df)

    results = pd.DataFrame(rows)
    output_path = results_dir / "test_paired_significance.csv"
    results.to_csv(output_path, index=False)

    print("Paired significance tests on held-out test predictions:")
    print(results.to_string(index=False))
    print(f"\nSaved paired significance results to {output_path}")


if __name__ == "__main__":
    main()
