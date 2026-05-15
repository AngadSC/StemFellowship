import pandas as pd


def make_binary_label(
    diagnoses: pd.DataFrame,
    label_name: str,
    positive_icd9_prefixes: list[str] | None = None,
    positive_icd9_codes: list[str] | None = None,
) -> pd.DataFrame:
    dx = diagnoses.dropna(subset=["ICD9_CODE"]).copy()
    dx["ICD9_CODE"] = dx["ICD9_CODE"].astype(str)

    match = pd.Series(False, index=dx.index)
    if positive_icd9_prefixes:
        match |= dx["ICD9_CODE"].str.startswith(tuple(positive_icd9_prefixes))
    if positive_icd9_codes:
        match |= dx["ICD9_CODE"].isin(positive_icd9_codes)

    dx[label_name] = match.astype(int)

    return dx.groupby("HADM_ID", as_index=False)[label_name].max()
