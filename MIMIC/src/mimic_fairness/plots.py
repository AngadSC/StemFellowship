import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_fairness_comparison(
    baseline_df: pd.DataFrame,
    reweighted_df: pd.DataFrame,
    output_path: str,
    metric: str = "fnr",
) -> None:
    """
    Plot grouped bar chart: baseline vs reweighted for each fairness group.

    metric: column name to plot (e.g., 'fnr', 'accuracy', 'auc')
    """
    merged = baseline_df.merge(
        reweighted_df,
        on="fairness_group",
        suffixes=("_baseline", "_reweighted"),
    )

    groups = merged["fairness_group"]
    baseline_vals = merged[f"{metric}_baseline"]
    reweighted_vals = merged[f"{metric}_reweighted"]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline", alpha=0.8)
    ax.bar(x + width / 2, reweighted_vals, width, label="Reweighted", alpha=0.8)

    ax.set_xlabel("Fairness Group")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Fairness Metric Comparison: {metric.upper()}")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {output_path}")
