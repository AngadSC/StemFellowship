# src/build_fitz_renamed_manifest.py

from pathlib import Path
import re
import pandas as pd

CSV_PATH = Path(r"C:\Users\Angad\Desktop\StemFellowship\external\fitzpatrick17k\fitzpatrick17k.csv")
IMAGE_ROOT = Path(r"C:\Users\Angad\Desktop\StemFellowship\data\raw\fitzpatrick\Categorized_AbbrvName")
OUT_PATH = Path(r"C:\Users\Angad\Desktop\StemFellowship\data\processed\fitzpatrick_renamed_manifest.csv")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".webp"}


def abbrev_label(label):
    label = label.lower()
    label = re.sub(r"[^a-z0-9]+", " ", label)
    words = [w for w in label.split() if w]
    return "-".join(w[:2] for w in words)


def parse_folder(folder_name):
    # Example: ba-ce-ca_468
    base, count = folder_name.rsplit("_", 1)
    return base, int(count)


def parse_fitzpatrick(filename):
    # Example: ba-ce-ca_f3_22_abcd1234.jpg
    parts = Path(filename).stem.split("_")
    f_code = parts[1]  # f0, f1, f2...
    return int(f_code.replace("f", ""))


def skin_group_from_fst(fst):
    if fst in [1, 2]:
        return "light_FST1_2"
    elif fst in [3, 4]:
        return "medium_FST3_4"
    elif fst in [5, 6]:
        return "dark_FST5_6"
    else:
        return "unknown"


# Load original CSV labels
csv_df = pd.read_csv(CSV_PATH)

label_info = (
    csv_df.groupby(["label", "three_partition_label", "nine_partition_label"])
    .size()
    .reset_index(name="csv_count")
)

label_info["abbr"] = label_info["label"].apply(abbrev_label)

# Build image rows
rows = []

for p in IMAGE_ROOT.rglob("*"):
    if not (p.is_file() and p.suffix.lower() in image_exts):
        continue

    folder = p.parent.name
    folder_abbr, folder_count = parse_folder(folder)
    fst = parse_fitzpatrick(p.name)

    matches = label_info[
        (label_info["abbr"] == folder_abbr) &
        (label_info["csv_count"] == folder_count)
    ]

    if len(matches) == 1:
        match = matches.iloc[0]
        label = match["label"]
        three_partition_label = match["three_partition_label"]
        nine_partition_label = match["nine_partition_label"]
        matched = True
    else:
        label = None
        three_partition_label = None
        nine_partition_label = None
        matched = False

    rows.append({
        "image_path": str(p),
        "folder": folder,
        "folder_abbr": folder_abbr,
        "folder_count": folder_count,
        "filename": p.name,
        "fitzpatrick": fst,
        "skin_group": skin_group_from_fst(fst),
        "label": label,
        "nine_partition_label": nine_partition_label,
        "three_partition_label": three_partition_label,
        "is_malignant": 1 if three_partition_label == "malignant" else 0 if three_partition_label is not None else None,
        "matched_to_csv": matched,
    })

manifest = pd.DataFrame(rows)

print("\nTotal images:", len(manifest))
print("Matched images:", manifest["matched_to_csv"].sum())
print("Unmatched images:", (~manifest["matched_to_csv"]).sum())

print("\nFST counts:")
print(manifest["fitzpatrick"].value_counts().sort_index())

print("\nSkin group counts:")
print(manifest["skin_group"].value_counts())

print("\nThree-partition counts:")
print(manifest["three_partition_label"].value_counts(dropna=False))

print("\nUnmatched folders:")
print(manifest.loc[~manifest["matched_to_csv"], ["folder", "folder_abbr", "folder_count"]].drop_duplicates().sort_values("folder").to_string(index=False))

manifest.to_csv(OUT_PATH, index=False)
print("\nSaved manifest to:", OUT_PATH)