# src/train_fitz_malignant_fairness.py

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# PATHS
# =========================

MANIFEST_PATH = Path(
    r"C:\Users\Angad\Desktop\StemFellowship\data\processed\fitzpatrick_renamed_manifest.csv"
)

RESULTS_DIR = Path(
    r"C:\Users\Angad\Desktop\StemFellowship\results\fitz_malignant_fairness"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CONFIG
# =========================

SEED = 42
EPOCHS = 3
BATCH_SIZE = 32
LR = 1e-4
TEST_SIZE = 0.2

# Baseline = False.
# Mitigation experiment later = True.
USE_WEIGHTED_SAMPLER = False

# Keep 0 on Windows/VS Code unless you know multiprocessing works.
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# REPRODUCIBILITY
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# DATA
# =========================

def prepare_dataframe():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)

    # Use only images successfully matched to disease labels.
    df = df[df["matched_to_csv"] == True].copy()

    # Drop unknown Fitzpatrick group f0.
    df = df[df["skin_group"] != "unknown"].copy()

    # Keep only rows with existing image paths.
    df["image_path"] = df["image_path"].astype(str)
    df = df[df["image_path"].apply(lambda p: Path(p).exists())].copy()

    # Binary label.
    df["is_malignant"] = df["is_malignant"].astype(int)

    print("\n====================")
    print("USABLE DATA")
    print("====================")
    print("Rows:", len(df))

    print("\nSkin group counts:")
    print(df["skin_group"].value_counts())

    print("\nSkin group x malignant:")
    print(pd.crosstab(df["skin_group"], df["is_malignant"]))

    if len(df) < 100:
        raise ValueError("Too few usable images. Check manifest/image paths.")

    # Stratify by both skin group and malignant label.
    df["stratify_col"] = df["skin_group"] + "_" + df["is_malignant"].astype(str)

    counts = df["stratify_col"].value_counts()
    valid_strata = counts[counts >= 2].index
    df = df[df["stratify_col"].isin(valid_strata)].copy()

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["stratify_col"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(RESULTS_DIR / "train_split.csv", index=False)
    test_df.to_csv(RESULTS_DIR / "test_split.csv", index=False)

    return train_df, test_df


class FitzMalignantDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(row["is_malignant"], dtype=torch.float32)

        return {
            "image": image,
            "label": label,
            "skin_group": row["skin_group"],
            "fitzpatrick": int(row["fitzpatrick"]),
            "filename": row["filename"],
            "image_path": row["image_path"],
        }


# =========================
# MODEL
# =========================

def build_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    return model, weights


# =========================
# METRICS
# =========================

def compute_group_metrics(pred_df: pd.DataFrame):
    rows = []

    for group, g in pred_df.groupby("skin_group"):
        y_true = g["true_label"].values
        y_pred = g["pred_label"].values
        y_prob = g["pred_prob"].values

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan

        rows.append({
            "skin_group": group,
            "n": len(g),
            "malignant_n": int(y_true.sum()),
            "non_malignant_n": int((y_true == 0).sum()),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "auc": auc,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "sensitivity_recall": sensitivity,
            "specificity": specificity,
            "false_negative_rate": fnr,
            "false_positive_rate": fpr,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        })

    return pd.DataFrame(rows)


def compute_overall_metrics(pred_df: pd.DataFrame):
    y_true = pred_df["true_label"].values
    y_pred = pred_df["pred_label"].values
    y_prob = pred_df["pred_prob"].values

    return pd.DataFrame([{
        "n": len(pred_df),
        "malignant_n": int(y_true.sum()),
        "non_malignant_n": int((y_true == 0).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
    }])


# =========================
# TRAIN / EVAL
# =========================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    all_true = []
    all_pred = []

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)

        all_true.extend(labels.detach().cpu().numpy().flatten().astype(int))
        all_pred.extend(preds)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_pred)
    bal_acc = balanced_accuracy_score(all_true, all_pred)

    return avg_loss, acc, bal_acc


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            true = labels.cpu().numpy().flatten().astype(int)

            for i in range(len(true)):
                rows.append({
                    "filename": batch["filename"][i],
                    "image_path": batch["image_path"][i],
                    "fitzpatrick": int(batch["fitzpatrick"][i]),
                    "skin_group": batch["skin_group"][i],
                    "true_label": int(true[i]),
                    "pred_label": int(preds[i]),
                    "pred_prob": float(probs[i]),
                })

    pred_df = pd.DataFrame(rows)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, pred_df


def make_weighted_sampler(train_df: pd.DataFrame):
    # Inverse-frequency weighting by skin_group x malignant label.
    combo = train_df["skin_group"] + "_" + train_df["is_malignant"].astype(str)
    counts = combo.value_counts()
    weights = combo.map(lambda x: 1.0 / counts[x]).values

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


def main():
    set_seed(SEED)

    print(f"\nUsing device: {DEVICE}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Results: {RESULTS_DIR}")

    train_df, test_df = prepare_dataframe()

    model, weights = build_model()
    transform = weights.transforms()

    train_dataset = FitzMalignantDataset(train_df, transform)
    test_dataset = FitzMalignantDataset(test_df, transform)

    if USE_WEIGHTED_SAMPLER:
        sampler = make_weighted_sampler(train_df)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = model.to(DEVICE)

    pos = train_df["is_malignant"].sum()
    neg = len(train_df) - pos

    if pos == 0:
        raise ValueError("No malignant examples found in training split.")

    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(DEVICE)

    print(f"\nTraining positives: {pos}")
    print(f"Training negatives: {neg}")
    print(f"pos_weight for BCE loss: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    best_bal_acc = -1.0
    best_path = RESULTS_DIR / "best_resnet18_malignant.pt"

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_acc, train_bal_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, pred_df = evaluate(model, test_loader, criterion)
        overall = compute_overall_metrics(pred_df)
        group_metrics = compute_group_metrics(pred_df)

        val_acc = overall.loc[0, "accuracy"]
        val_bal_acc = overall.loc[0, "balanced_accuracy"]
        val_auc = overall.loc[0, "auc"]

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Train bal acc: {train_bal_acc:.4f}"
        )

        print(
            f"Val loss:   {val_loss:.4f} | "
            f"Val acc:   {val_acc:.4f} | "
            f"Val bal acc: {val_bal_acc:.4f} | "
            f"AUC: {val_auc:.4f}"
        )

        print("\nGroup fairness metrics:")
        print(group_metrics[[
            "skin_group",
            "n",
            "malignant_n",
            "balanced_accuracy",
            "auc",
            "sensitivity_recall",
            "false_negative_rate",
            "false_positive_rate",
        ]].to_string(index=False))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_balanced_accuracy": train_bal_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_balanced_accuracy": val_bal_acc,
            "val_auc": val_auc,
        })

        # Save latest every epoch
        pred_df.to_csv(RESULTS_DIR / "latest_predictions.csv", index=False)
        overall.to_csv(RESULTS_DIR / "latest_overall_metrics.csv", index=False)
        group_metrics.to_csv(RESULTS_DIR / "latest_group_fairness_metrics.csv", index=False)

        # Save best by balanced accuracy
        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            torch.save(model.state_dict(), best_path)
            pred_df.to_csv(RESULTS_DIR / "best_predictions.csv", index=False)
            overall.to_csv(RESULTS_DIR / "best_overall_metrics.csv", index=False)
            group_metrics.to_csv(RESULTS_DIR / "best_group_fairness_metrics.csv", index=False)

    pd.DataFrame(history).to_csv(RESULTS_DIR / "training_history.csv", index=False)

    print("\nDone.")
    print(f"Saved best model to: {best_path}")
    print(f"Saved results to: {RESULTS_DIR}")
    print("Main file to check: best_group_fairness_metrics.csv")


if __name__ == "__main__":
    main()