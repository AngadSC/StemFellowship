from pathlib import Path
from mimic_fairness.paths import load_config, project_root


def main():
    cfg = load_config()
    raw_dir = project_root() / cfg["paths"]["raw_dir"]

    required = [
        cfg["files"]["admissions"],
        cfg["files"]["patients"],
        cfg["files"]["notes"],
        cfg["files"]["diagnoses"],
        cfg["files"]["diagnosis_dictionary"],
    ]

    print(f"Checking raw directory: {raw_dir}")

    missing = []
    for filename in required:
        path = raw_dir / filename
        if path.exists():
            print(f"[OK] {filename}")
        else:
            print(f"[MISSING] {filename}")
            missing.append(filename)

    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    print("All required files found.")


if __name__ == "__main__":
    main()
