# src/inspect_renamed_fitz.py

from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path(r"C:\Users\Angad\Desktop\StemFellowship\data\raw\fitzpatrick\Categorized_AbbrvName")

image_exts = {".jpg", ".jpeg", ".png", ".webp"}
image_files = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in image_exts]

print("Total images:", len(image_files))

rows = []

for p in image_files:
    folder = p.parent.name
    stem = p.stem

    parts = stem.split("_")

    rows.append({
        "image_path": str(p),
        "folder": folder,
        "filename": p.name,
        "stem": stem,
        "parts": parts,
        "part0": parts[0] if len(parts) > 0 else None,
        "part1": parts[1] if len(parts) > 1 else None,
        "part2": parts[2] if len(parts) > 2 else None,
    })

df = pd.DataFrame(rows)

print("\nFirst 30 rows:")
print(df.head(30)[["folder", "filename", "part0", "part1", "part2"]])

print("\nFolder count:")
print(df["folder"].nunique())

print("\nF-code counts from filename part1:")
print(df["part1"].value_counts().sort_index())

print("\nFirst 30 folders:")
for folder in sorted(df["folder"].unique())[:30]:
    print(folder)

df.to_csv("results/renamed_fitz_manifest_debug.csv", index=False)
print("\nSaved: results/renamed_fitz_manifest_debug.csv")