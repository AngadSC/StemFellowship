from pathlib import Path
import pandas as pd

DATA_PATH = Path("external/fitzpatrick17k/fitzpatrick17k.csv")
OUT_DIR = Path("results/audit_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FITZ_COL = "fitzpatrick_scale"
LABEL_COL = "three_partition_label"

def skin_group(x):
    try:
        x = int(x)
    except Exception:
        return "unknown"

    if x in [1, 2]:
        return "light_FST1_2"
    if x in [3, 4]:
        return "medium_FST3_4"
    if x in [5, 6]:
        return "dark_FST5_6"
    return "unknown"

def main():
    df = pd.read_csv(DATA_PATH)

    df["skin_group"] = df[FITZ_COL].apply(skin_group)
    df["is_malignant"] = (df[LABEL_COL] == "malignant").astype(int)

    print("\nThree-partition label distribution:")
    print(df[LABEL_COL].value_counts(dropna=False))

    print("\nSkin group distribution:")
    print(df["skin_group"].value_counts(dropna=False))

    print("\nSkin group x three_partition_label:")
    table = pd.crosstab(df["skin_group"], df[LABEL_COL])
    print(table)

    print("\nSkin group x malignant binary:")
    binary_table = pd.crosstab(df["skin_group"], df["is_malignant"])
    binary_table.columns = ["non_malignant", "malignant"]
    print(binary_table)

    print("\nMalignant rate by skin group:")
    print(df.groupby("skin_group")["is_malignant"].mean().sort_values(ascending=False))

    table.to_csv(OUT_DIR / "fitzpatrick17k_skin_group_by_three_partition.csv")
    binary_table.to_csv(OUT_DIR / "fitzpatrick17k_skin_group_by_malignant_binary.csv")

    print("\nSaved audit tables.")

if __name__ == "__main__":
    main()