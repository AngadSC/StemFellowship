import pandas as pd


def normalize_ethnicity(value: str) -> str:
    if pd.isna(value):
        return "unknown"

    v = value.upper()

    if "WHITE" in v:
        return "white"
    if "BLACK" in v or "AFRICAN" in v:
        return "black"
    if "HISPANIC" in v or "LATINO" in v:
        return "hispanic"
    if "ASIAN" in v:
        return "asian"

    return "other"


def normalize_language(value: str) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "unknown"

    v = str(value).upper().strip()

    if v in {"ENGL", "ENGLISH"}:
        return "english"

    return "non_english"


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ethnicity_group"] = out["ETHNICITY"].apply(normalize_ethnicity)
    out["language_group"] = out["LANGUAGE"].apply(normalize_language)

    def combined_group(row):
        if row["ethnicity_group"] == "white" and row["language_group"] == "english":
            return "white_english"
        if row["ethnicity_group"] == "black":
            return "black"
        if row["ethnicity_group"] == "hispanic":
            return "hispanic"
        if row["language_group"] == "non_english":
            return "non_english"
        return "other"

    out["fairness_group"] = out.apply(combined_group, axis=1)

    return out
