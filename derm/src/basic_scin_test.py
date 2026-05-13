from pathlib import Path
import ast
import pandas as pd

DATA_PATH = Path("data/raw/scin/scin_cases_and_labels.csv")

def parse_label_dict(value):
    """SCIN condition labels are stored as strings like Python dicts."""
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return list(parsed.keys())
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run: python src/download_scin.py"
        )

    df = pd.read_csv(DATA_PATH, dtype={"case_id": str})

    print("\n====================")
    print("BASIC DATASET INFO")
    print("====================")
    print("Rows/cases:", len(df))
    print("Columns:", len(df.columns))
    print("Duplicate case IDs:", df["case_id"].duplicated().sum())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\n====================")
    print("IMAGE PATH COUNTS")
    print("====================")
    for col in ["image_1_path", "image_2_path", "image_3_path"]:
        if col in df.columns:
            print(f"{col}: {df[col].notna().sum()}")

    print("\n====================")
    print("SELF-REPORTED FITZPATRICK")
    print("====================")
    print(df["fitzpatrick_skin_type"].value_counts(dropna=False))

    print("\n====================")
    print("DERMATOLOGIST FITZPATRICK LABEL 1")
    print("====================")
    print(df["dermatologist_fitzpatrick_skin_type_label_1"].value_counts(dropna=False))

    print("\n====================")
    print("MONK SKIN TONE US")
    print("====================")
    print(df["monk_skin_tone_label_us"].value_counts(dropna=False).sort_index())

    print("\n====================")
    print("RACE / ETHNICITY COLUMNS")
    print("====================")
    race_cols = [c for c in df.columns if c.startswith("race_ethnicity_")]
    for col in race_cols:
        yes_count = (df[col] == "YES").sum()
        print(f"{col}: {yes_count}")

    print("\n====================")
    print("AMERICAN INDIAN OR ALASKA NATIVE CHECK")
    print("====================")
    ai_an_col = "race_ethnicity_american_indian_or_alaska_native"
    if ai_an_col in df.columns:
        print(df[ai_an_col].value_counts(dropna=False))
        print("AI/AN YES count:", (df[ai_an_col] == "YES").sum())

    print("\n====================")
    print("TOP CONDITION LABELS")
    print("====================")
    all_conditions = []
    for value in df["weighted_skin_condition_label"]:
        all_conditions.extend(parse_label_dict(value))

    condition_counts = pd.Series(all_conditions).value_counts()
    print(condition_counts.head(30))

    print("\n====================")
    print("SAVE BASIC AUDIT TABLES")
    print("====================")
    out_dir = Path("results/audit_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    df["fitzpatrick_skin_type"].value_counts(dropna=False).to_csv(
        out_dir / "self_reported_fitzpatrick_counts.csv"
    )

    df["dermatologist_fitzpatrick_skin_type_label_1"].value_counts(dropna=False).to_csv(
        out_dir / "dermatologist_fitzpatrick_counts.csv"
    )

    df["monk_skin_tone_label_us"].value_counts(dropna=False).sort_index().to_csv(
        out_dir / "monk_skin_tone_us_counts.csv"
    )

    condition_counts.to_csv(out_dir / "condition_counts.csv")

    race_summary = {
        col: int((df[col] == "YES").sum())
        for col in race_cols
    }
    pd.Series(race_summary).sort_values(ascending=False).to_csv(
        out_dir / "race_ethnicity_yes_counts.csv"
    )

    print("Saved tables to results/audit_tables/")

if __name__ == "__main__":
    main()