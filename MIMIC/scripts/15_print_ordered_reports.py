from pathlib import Path

from mimic_fairness.paths import load_config, project_root


def _factor_dir_name(upweight_factor: float) -> str:
    factor_str = f"{upweight_factor:.2f}".rstrip("0").rstrip(".")
    return f"factor_{factor_str.replace('.', '_')}"


def _factor_grid(cfg: dict) -> list[float]:
    factors = cfg["model"].get("reweighted_factor_grid", [])
    if isinstance(factors, (int, float)):
        factors = [factors]
    return [float(factor) for factor in factors]


def _report_path(root: Path, cfg: dict, label: str, factor: float) -> Path:
    return (
        root
        / cfg["paths"]["outputs_dir"]
        / "by_disease"
        / label
        / "tables"
        / _factor_dir_name(factor)
        / "val_threshold"
        / "heldout_test_statistical_significance_report.txt"
    )


def main() -> None:
    root = project_root()
    cfg = load_config()
    labels = list(cfg["labels"].keys())
    factors = _factor_grid(cfg)

    for label_index, label in enumerate(labels):
        if label_index:
            print("\n" + "#" * 100 + "\n")
        print(f"{'#' * 36} {label} {'#' * 36}")

        for factor in factors:
            path = _report_path(root, cfg, label, factor)
            if not path.exists():
                print(f"\nMISSING REPORT: {path}\n")
                continue

            print()
            print(path.read_text(encoding="utf-8").rstrip())


if __name__ == "__main__":
    main()
