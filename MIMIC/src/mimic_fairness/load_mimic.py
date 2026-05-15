import pandas as pd
from pathlib import Path


def read_admissions(raw_dir: Path, filename: str) -> pd.DataFrame:
    cols = ["SUBJECT_ID", "HADM_ID", "ETHNICITY", "LANGUAGE", "ADMITTIME", "DISCHTIME"]
    return pd.read_csv(raw_dir / filename, compression="gzip", usecols=cols)


def read_patients(raw_dir: Path, filename: str) -> pd.DataFrame:
    cols = ["SUBJECT_ID", "GENDER", "DOB", "DOD"]
    return pd.read_csv(raw_dir / filename, compression="gzip", usecols=cols)


def read_diagnoses(raw_dir: Path, filename: str) -> pd.DataFrame:
    cols = ["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]
    return pd.read_csv(raw_dir / filename, compression="gzip", usecols=cols, dtype={"ICD9_CODE": str})


def read_notes(raw_dir: Path, filename: str, category: str = "Discharge summary") -> pd.DataFrame:
    cols = ["SUBJECT_ID", "HADM_ID", "CATEGORY", "DESCRIPTION", "TEXT"]
    notes = pd.read_csv(raw_dir / filename, compression="gzip", usecols=cols)

    if category:
        notes = notes[notes["CATEGORY"] == category].copy()

    notes = notes.dropna(subset=["HADM_ID", "TEXT"])
    notes["HADM_ID"] = notes["HADM_ID"].astype(int)

    notes["text_length"] = notes["TEXT"].str.len()
    notes = notes.sort_values("text_length", ascending=False)
    notes = notes.drop_duplicates(subset=["HADM_ID"], keep="first")
    notes = notes.drop(columns=["text_length"])

    return notes
