from pathlib import Path
import requests
import pandas as pd

BASE_URL = "https://storage.googleapis.com/dx-scin-public-data"

OUT_DIR = Path("data/raw/scin")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "scin_cases.csv": f"{BASE_URL}/dataset/scin_cases.csv",
    "scin_labels.csv": f"{BASE_URL}/dataset/scin_labels.csv",
}

def download_file(url: str, output_path: Path):
    print(f"Downloading {url}")
    r = requests.get(url)
    r.raise_for_status()
    output_path.write_bytes(r.content)
    print(f"Saved to {output_path}")

def main():
    for filename, url in FILES.items():
        out_path = OUT_DIR / filename
        if not out_path.exists():
            download_file(url, out_path)
        else:
            print(f"Already exists: {out_path}")

    cases = pd.read_csv(OUT_DIR / "scin_cases.csv", dtype={"case_id": str})
    labels = pd.read_csv(OUT_DIR / "scin_labels.csv", dtype={"case_id": str})

    merged = cases.merge(labels, on="case_id")
    merged.to_csv(OUT_DIR / "scin_cases_and_labels.csv", index=False)

    print("Done.")
    print("Cases:", len(cases))
    print("Labels:", len(labels))
    print("Merged:", len(merged))
    print("Saved merged file to:", OUT_DIR / "scin_cases_and_labels.csv")

if __name__ == "__main__":
    main()