from pathlib import Path
import pandas as pd

from mimic_fairness.load_mimic import read_admissions, read_patients, read_diagnoses, read_notes
from mimic_fairness.labels import make_binary_label
from mimic_fairness.groups import add_groups


def build_cohort(cfg: dict, root: Path) -> pd.DataFrame:
    raw_dir = root / cfg["paths"]["raw_dir"]

    admissions = read_admissions(raw_dir, cfg["files"]["admissions"])
    patients = read_patients(raw_dir, cfg["files"]["patients"])
    diagnoses = read_diagnoses(raw_dir, cfg["files"]["diagnoses"])
    notes = read_notes(
        raw_dir,
        cfg["files"]["notes"],
        category=cfg["cohort"]["note_category"],
    )

    label_name = cfg["label"]["task_name"]
    labels = make_binary_label(
        diagnoses,
        cfg["label"]["positive_icd9_prefixes"],
        label_name,
    )

    df = notes.merge(admissions, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    df = df.merge(patients, on="SUBJECT_ID", how="left")
    df = df.merge(labels, on="HADM_ID", how="left")

    df[label_name] = df[label_name].fillna(0).astype(int)

    df = add_groups(df)

    df = df[
        [
            "SUBJECT_ID",
            "HADM_ID",
            "TEXT",
            "ETHNICITY",
            "LANGUAGE",
            "ethnicity_group",
            "language_group",
            "fairness_group",
            label_name,
        ]
    ].copy()

    return df
