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
    split_path: str | None = None,
    split: str | None = None,
    predictions_output_path: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate model fairness across groups.

    Returns one row per fairness group with confusion counts and metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    dataset = MIMICDataset(cohort_path, tokenizer, max_length, label_column)

    indices = np.arange(len(dataset))
    if split_path is not None:
        if split is None:
            raise ValueError("split must be provided when split_path is provided.")

        splits = pd.read_parquet(split_path)
        required_columns = {"SUBJECT_ID", "HADM_ID", "split"}
        missing_columns = required_columns.difference(splits.columns)
        if missing_columns:
            raise ValueError(f"Split file is missing required columns: {sorted(missing_columns)}")

        split_counts = splits.groupby("SUBJECT_ID")["split"].nunique()
        overlapping_subjects = split_counts[split_counts > 1]
        if not overlapping_subjects.empty:
            examples = overlapping_subjects.index[:10].tolist()
            raise ValueError(
                "SUBJECT_ID leakage detected across splits. "
                f"Example overlapping SUBJECT_ID values: {examples}"
            )

        merged = dataset.df[["SUBJECT_ID", "HADM_ID"]].merge(
            splits[["SUBJECT_ID", "HADM_ID", "split"]],
            on=["SUBJECT_ID", "HADM_ID"],
            how="left",
            validate="one_to_one",
        )

        if merged["split"].isna().any():
            missing = merged.loc[merged["split"].isna(), ["SUBJECT_ID", "HADM_ID"]].head(10)
            raise ValueError(f"Some cohort rows are missing from split file:\n{missing}")

        indices = np.flatnonzero(merged["split"].eq(split).to_numpy())
        if len(indices) == 0:
            raise ValueError(f"No rows found for split='{split}'.")

        selected_rows = merged.iloc[indices]
        selected_splits = set(selected_rows["split"].unique())
        if selected_splits != {split}:
            raise ValueError(
                f"Evaluation split filter failed. Expected only {split!r}, "
                f"found {sorted(selected_splits)}."
            )
        if split == "test" and selected_rows["split"].isin(["train", "val"]).any():
            raise ValueError("Held-out test evaluation included train or val rows.")

    if split_path is not None:
        dataset_for_eval = torch.utils.data.Subset(dataset, indices)
        metadata_df = dataset.df.iloc[indices].reset_index(drop=True)
    else:
        dataset_for_eval = dataset
        metadata_df = dataset.df.reset_index(drop=True)

    dataloader = DataLoader(dataset_for_eval, batch_size=batch_size, shuffle=False)

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
    predictions_df = metadata_df[["SUBJECT_ID", "HADM_ID", "fairness_group", label_column]].copy()
    predictions_df["y_true"] = all_labels
    predictions_df["y_pred"] = all_preds
    predictions_df["y_prob"] = all_probs
    if split is not None:
        predictions_df["split"] = split

    if predictions_output_path is not None:
        output_path = Path(predictions_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_parquet(output_path, index=False)

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

        tn, fp, fn, tp = confusion_matrix(group_labels, group_preds, labels=[0, 1]).ravel()
        positives = int(np.sum(group_labels == 1))
        negatives = int(np.sum(group_labels == 0))
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        predicted_positive_rate = (tp + fp) / len(group_labels)

        results.append({
            "fairness_group": group,
            "n_samples": len(group_labels),
            "positives": positives,
            "negatives": negatives,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "accuracy": acc,
            "auc": auc,
            "fnr": fnr,
            "fpr": fpr,
            "recall": recall,
            "sensitivity": recall,
            "specificity": specificity,
            "precision": precision,
            "predicted_positive_rate": predicted_positive_rate,
        })

    return pd.DataFrame(results)
