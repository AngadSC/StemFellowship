import pandas as pd


def make_binary_label(
    diagnoses: pd.DataFrame,
    positive_icd9_prefixes: list[str],
    label_name: str,
) -> pd.DataFrame:
    dx = diagnoses.dropna(subset=["ICD9_CODE"]).copy()
    dx["ICD9_CODE"] = dx["ICD9_CODE"].astype(str)

    pattern = tuple(positive_icd9_prefixes)
    dx[label_name] = dx["ICD9_CODE"].str.startswith(pattern).astype(int)

    labels = (
        dx.groupby("HADM_ID", as_index=False)[label_name]
        .max()
    )

    return labels
