import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

from mimic_fairness.preprocessing import load_tokenizer, tokenize_text


class MIMICDataset(Dataset):
    def __init__(self, parquet_path: str, tokenizer, max_length: int, label_column: str):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_column = label_column

        parquet_file = Path(parquet_path)
        cache_path = parquet_file.parent / f".{parquet_file.stem}_tokens_cache.pt"

        cache_valid = (cache_path.exists() and
                      cache_path.stat().st_mtime > parquet_file.stat().st_mtime)

        if cache_valid:
            print(f"Loading cached tokens from {cache_path}...")
            cache = torch.load(cache_path)
            self.input_ids = cache["input_ids"]
            self.attention_masks = cache["attention_masks"]
            self.labels = cache["labels"]
            self.fairness_groups = cache["fairness_groups"]
        else:
            print("Pre-tokenizing dataset (this happens once)...")
            self.input_ids = []
            self.attention_masks = []
            self.labels = []
            self.fairness_groups = []

            for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Tokenizing"):
                text = row["TEXT"]
                label = int(row[self.label_column])
                fairness_group = row["fairness_group"]

                tokenized = tokenize_text(text, self.tokenizer, self.max_length)

                self.input_ids.append(torch.tensor(tokenized["input_ids"], dtype=torch.long))
                self.attention_masks.append(torch.tensor(tokenized["attention_mask"], dtype=torch.long))
                self.labels.append(torch.tensor(label, dtype=torch.long))
                self.fairness_groups.append(fairness_group)

            print(f"Caching tokens to {cache_path}...")
            torch.save({
                "input_ids": self.input_ids,
                "attention_masks": self.attention_masks,
                "labels": self.labels,
                "fairness_groups": self.fairness_groups,
            }, cache_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "label": self.labels[idx],
            "fairness_group": self.fairness_groups[idx],
        }
