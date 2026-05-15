import torch
import pandas as pd
from torch.utils.data import Dataset

from mimic_fairness.preprocessing import load_tokenizer, tokenize_text


class MIMICDataset(Dataset):
    def __init__(self, parquet_path: str, tokenizer, max_length: int, label_column: str):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = row["TEXT"]
        label = int(row[self.label_column])
        fairness_group = row["fairness_group"]

        tokenized = tokenize_text(text, self.tokenizer, self.max_length)

        return {
            "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "fairness_group": fairness_group,
        }
