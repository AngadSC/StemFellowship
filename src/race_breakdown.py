from pathlib import Path
import ast
import pandas as pd

DATA_PATH = Path("data/raw/scin/scin_cases_and_labels.csv")
OUT_DIR = Path("results/audit_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "weighted_skin_condition_label"
MONK_COL = "monk_skin_tone_label_us"

RACE_COLS = [
    "race_ethnicity_american_indian_or_alaska_native",
    "race_ethnicity_asian",
    "race_ethnicity_black_or_african_american",
    "race_ethnicity_hispanic_latino_or_spanish_origin",
    "race_ethnicity_middle_eastern_or_north_african",
    "race_ethnicity_native_hawaiian_or_pacific_islander",
    "race_ethnicity_white",
    "race_ethnicity_other_race",
    "race_ethnicity_prefer_not_to_answer",
    "race_ethnicity_two_or_more_after_mitigation",
]


def parse_condition_labels(value):
    """SCIN labels are stored as strings that look like Python dictionaries."""
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


def monk_group(value):
    if pd.isna(value):
        return "missing"

    value = float(value)

    if 1 <= value <= 3:
        return "light_1_3"
    elif 4 <= value <= 6:
        return "medium_4_6"
    elif 7 <= value <= 10:
        return "dark_7_10"
    else:
        return "missing"


def get_race_group(row):
    """
    Returns a combined race/ethnicity label.
    A case can have multiple race/ethnicity flags.
    """
    groups = []

    for col in RACE_COLS:
        if col in row.index and row[col] == "YES":
            clean_name = col.replace("race_ethnicity_", "")
            groups.append(clean_name)

    if not groups:
        return "unspecified"

    return ",".join(groups)


def make_binary_label(df, condition):
    return df[LABEL_COL].apply(
        lambda x: condition in parse_condition_labels(x)
    ).astype(int)


def condition_breakdown(df, condition):
    df = df.copy()

    df["target"] = make_binary_label(df, condition)
    df["race_group"] = df.apply(get_race_group, axis=1)
    df["monk_group"] = df[MONK_COL].apply(monk_group)

    positive_df = df[df["target"] == 1]
    negative_df = df[df["target"] == 0]

    print("\n==============================")
    print(f"CONDITION: {condition}")
    print("==============================")
    print(f"Positive cases: {len(positive_df)}")
    print(f"Negative cases: {len(negative_df)}")

    print("\nPositive cases by race/ethnicity:")
    race_pos = positive_df["race_group"].value_counts(dropna=False)
    print(race_pos)

    print("\nPositive cases by Monk skin-tone group:")
    monk_pos = positive_df["monk_group"].value_counts(dropna=False)
    print(monk_pos)

    print("\nRace x target table:")
    race_target = pd.crosstab(df["race_group"], df["target"])
    race_target.columns = ["non_condition", "condition"]
    print(race_target)

    print("\nMonk group x target table:")
    monk_target = pd.crosstab(df["monk_group"], df["target"])
    monk_target.columns = ["non_condition", "condition"]
    print(monk_target)

    # Save outputs
    safe_condition = condition.lower().replace(" ", "_").replace("/", "_")

    race_target.to_csv(OUT_DIR / f"{safe_condition}_race_target_table.csv")
    monk_target.to_csv(OUT_DIR / f"{safe_condition}_monk_target_table.csv")

    return race_target, monk_target


def main():
    df = pd.read_csv(DATA_PATH, dtype={"case_id": str})

    # Start with eczema
    condition_breakdown(df, "Eczema")

    # Optional: check more common labels too
    common_conditions = [
        "Eczema",
        "Allergic Contact Dermatitis",
        "Insect Bite",
        "Urticaria",
        "Psoriasis",
        "Folliculitis",
        "Irritant Contact Dermatitis",
        "Tinea",
    ]

    summary_rows = []

    df["race_group"] = df.apply(get_race_group, axis=1)
    df["monk_group"] = df[MONK_COL].apply(monk_group)

    for condition in common_conditions:
        df["target"] = make_binary_label(df, condition)
        pos = df[df["target"] == 1]

        row = {
            "condition": condition,
            "total_positive": int(df["target"].sum()),
            "light_positive": int((pos["monk_group"] == "light_1_3").sum()),
            "medium_positive": int((pos["monk_group"] == "medium_4_6").sum()),
            "dark_positive": int((pos["monk_group"] == "dark_7_10").sum()),
            "aian_positive": int(
                (pos["race_ethnicity_american_indian_or_alaska_native"] == "YES").sum()
            ),
            "white_positive": int(
                (pos["race_ethnicity_white"] == "YES").sum()
            ),
            "black_positive": int(
                (pos["race_ethnicity_black_or_african_american"] == "YES").sum()
            ),
            "asian_positive": int(
                (pos["race_ethnicity_asian"] == "YES").sum()
            ),
            "hispanic_positive": int(
                (pos["race_ethnicity_hispanic_latino_or_spanish_origin"] == "YES").sum()
            ),
        }

        # Simple usability rule
        if row["dark_positive"] >= 20 and row["medium_positive"] >= 20 and row["light_positive"] >= 20:
            row["skin_tone_status"] = "usable"
        else:
            row["skin_tone_status"] = "underpowered"

        if row["aian_positive"] >= 20:
            row["aian_status"] = "maybe_usable"
        else:
            row["aian_status"] = "underpowered"

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    print("\n==============================")
    print("COMMON CONDITION SUMMARY")
    print("==============================")
    print(summary)

    summary.to_csv(OUT_DIR / "common_condition_subgroup_summary.csv", index=False)


if __name__ == "__main__":
    main()