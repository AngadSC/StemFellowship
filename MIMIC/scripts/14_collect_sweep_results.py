from pathlib import Path

import pandas as pd

from mimic_fairness.paths import load_config, project_root


def _factor_dir_name(upweight_factor: float) -> str:
    factor_str = f"{upweight_factor:.2f}".rstrip("0").rstrip(".")
    return f"factor_{factor_str.replace('.', '_')}"


def _factor_display(upweight_factor: float) -> str:
    return f"{upweight_factor:.2f}".rstrip("0").rstrip(".")


def _factor_grid(cfg: dict) -> list[float]:
    factors = cfg["model"].get("reweighted_factor_grid", [])
    if isinstance(factors, (int, float)):
        factors = [factors]
    return [float(factor) for factor in factors]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _overall_row(label: str, factor: float, factor_dir: Path) -> dict:
    selected = _read_csv(factor_dir / f"{label}_validation_selected_thresholds.csv")
    metrics = _read_csv(factor_dir / f"{label}_validation_threshold_selected_metrics.csv")
    paired = _read_csv(factor_dir / "val_threshold" / "heldout_test_paired_significance.csv")
    fnr_z = _read_csv(factor_dir / "val_threshold" / "heldout_test_fnr_group_z_tests.csv")

    test_metrics = metrics[metrics["split"].eq("test")].set_index("model")
    thresholds = selected.set_index("model")["threshold"]
    paired_overall = paired[paired["fairness_group"].eq("overall")].set_index("metric")

    baseline_fnr_ratio = fnr_z["baseline_fnr"].max() / fnr_z["baseline_fnr"].min()
    reweighted_fnr_ratio = fnr_z["reweighted_fnr"].max() / fnr_z["reweighted_fnr"].min()

    return {
        "disease": label,
        "upweight_factor": factor,
        "baseline_threshold": thresholds.get("baseline"),
        "reweighted_threshold": thresholds.get("reweighted"),
        "baseline_test_fnr": test_metrics.loc["baseline", "fnr"],
        "reweighted_test_fnr": test_metrics.loc["reweighted", "fnr"],
        "fnr_reduction": test_metrics.loc["baseline", "fnr"] - test_metrics.loc["reweighted", "fnr"],
        "fnr_p_value": paired_overall.loc["fnr", "p_value"],
        "baseline_test_accuracy": test_metrics.loc["baseline", "accuracy"],
        "reweighted_test_accuracy": test_metrics.loc["reweighted", "accuracy"],
        "accuracy_change": test_metrics.loc["reweighted", "accuracy"] - test_metrics.loc["baseline", "accuracy"],
        "accuracy_p_value": paired_overall.loc["accuracy", "p_value"],
        "baseline_fnr_ratio": baseline_fnr_ratio,
        "reweighted_fnr_ratio": reweighted_fnr_ratio,
        "fnr_ratio_improvement": baseline_fnr_ratio - reweighted_fnr_ratio,
        "results_dir": str(factor_dir),
    }


def _group_rows(label: str, factor: float, factor_dir: Path) -> pd.DataFrame:
    rows = _read_csv(factor_dir / "val_threshold" / "heldout_test_fnr_group_z_tests.csv")
    rows.insert(0, "upweight_factor", factor)
    rows.insert(0, "disease", label)
    return rows


def _append_report(label: str, factor: float, factor_dir: Path, report_parts: list[str]) -> None:
    report_path = factor_dir / "val_threshold" / "heldout_test_statistical_significance_report.txt"
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report_parts.append(report_path.read_text(encoding="utf-8").rstrip())


def main() -> None:
    root = project_root()
    cfg = load_config()
    factors = _factor_grid(cfg)
    labels = list(cfg["labels"].keys())
    summary_dir = root / cfg["paths"]["outputs_dir"] / "sweep_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    group_frames = []
    report_parts = []

    for label in labels:
        tables_dir = root / cfg["paths"]["outputs_dir"] / "by_disease" / label / "tables"
        for factor in factors:
            factor_dir = tables_dir / _factor_dir_name(factor)
            if not factor_dir.exists():
                raise FileNotFoundError(factor_dir)
            overall_rows.append(_overall_row(label, factor, factor_dir))
            group_frames.append(_group_rows(label, factor, factor_dir))
            _append_report(label, factor, factor_dir, report_parts)

    overall = pd.DataFrame(overall_rows)
    overall["upweight_factor_label"] = overall["upweight_factor"].map(_factor_display)
    overall.to_csv(summary_dir / "overall_factor_sweep_summary.csv", index=False)

    group = pd.concat(group_frames, ignore_index=True)
    group["upweight_factor_label"] = group["upweight_factor"].map(_factor_display)
    group.to_csv(summary_dir / "group_fnr_factor_sweep_summary.csv", index=False)

    (summary_dir / "ordered_statistical_reports.txt").write_text(
        "\n\n".join(report_parts) + "\n",
        encoding="utf-8",
    )

    best_fnr = overall.sort_values(
        ["disease", "fnr_reduction", "accuracy_change"],
        ascending=[True, False, False],
    ).groupby("disease", as_index=False).head(1)
    best_fnr.to_csv(summary_dir / "best_factor_by_fnr_reduction.csv", index=False)

    print(f"Saved ordered sweep summaries to {summary_dir}")
    print("\nBest factor by FNR reduction per disease:")
    print(
        best_fnr[
            [
                "disease",
                "upweight_factor_label",
                "fnr_reduction",
                "accuracy_change",
                "fnr_ratio_improvement",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
