from pathlib import Path
import pandas as pd

DATA_PATH = Path("external/fitzpatrick17k/fitzpatrick17k.csv")
OUT_DIR = Path("results/audit_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. "
            "Make sure you cloned the Fitzpatrick17k repo into external/"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n====================")
    print("BASIC DATASET INFO")
    print("====================")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Column names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\n====================")
    print("MISSING VALUES")
    print("====================")
    print(df.isna().sum().sort_values(ascending=False).head(30))

    print("\n====================")
    print("POSSIBLE FITZPATRICK COLUMNS")
    print("====================")
    fitz_cols = [c for c in df.columns if "fitz" in c.lower()]
    print(fitz_cols)

    for col in fitz_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).sort_index())

    print("\n====================")
    print("POSSIBLE LABEL COLUMNS")
    print("====================")
    possible_label_cols = [
        c for c in df.columns
        if any(word in c.lower() for word in [
            "label", "diagnosis", "disease", "partition", "condition"
        ])
    ]
    print(possible_label_cols)

    for col in possible_label_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).head(30))

    # Save basic summaries
    for col in fitz_cols:
        df[col].value_counts(dropna=False).sort_index().to_csv(
            OUT_DIR / f"fitzpatrick17k_{col}_counts.csv"
        )

    for col in possible_label_cols:
        df[col].value_counts(dropna=False).to_csv(
            OUT_DIR / f"fitzpatrick17k_{col}_counts.csv"
        )

    print("\nSaved audit tables to results/audit_tables/")

if __name__ == "__main__":
    main()