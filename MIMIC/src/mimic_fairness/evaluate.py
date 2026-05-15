import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader

from mimic_fairness.dataset import MIMICDataset


def evaluate_fairness(
    cohort_path: str,
    checkpoint_path: str,
    max_length: int,
    label_column: str,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Evaluate model fairness across groups.

    Returns DataFrame with columns:
    - fairness_group
    - accuracy
    - auc
    - fnr (false negative rate)
    - fn_count
    - tp_count
    - n_samples
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    dataset = MIMICDataset(cohort_path, tokenizer, max_length, label_column)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probs = []
    all_labels = []
    all_groups = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_groups.extend(batch["fairness_group"])

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    results = []

    for group in sorted(set(all_groups)):
        mask = np.array(all_groups) == group
        group_preds = all_preds[mask]
        group_probs = all_probs[mask]
        group_labels = all_labels[mask]

        if len(group_labels) == 0:
            continue

        acc = accuracy_score(group_labels, group_preds)

        if len(np.unique(group_labels)) > 1:
            auc = roc_auc_score(group_labels, group_probs)
        else:
            auc = np.nan

        tn, fp, fn, tp = confusion_matrix(group_labels, group_preds).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        results.append({
            "fairness_group": group,
            "accuracy": acc,
            "auc": auc,
            "fnr": fnr,
            "fn_count": int(fn),
            "tp_count": int(tp),
            "n_samples": len(group_labels),
        })

    return pd.DataFrame(results)
