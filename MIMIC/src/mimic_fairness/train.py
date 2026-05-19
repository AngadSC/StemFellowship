import time

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from transformers import AutoModelForSequenceClassification

from mimic_fairness.preprocessing import load_tokenizer
from mimic_fairness.dataset import MIMICDataset


def _load_train_val_indices(df: pd.DataFrame, split_path: str) -> tuple[np.ndarray, np.ndarray]:
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

    merged = df[["SUBJECT_ID", "HADM_ID"]].merge(
        splits[["SUBJECT_ID", "HADM_ID", "split"]],
        on=["SUBJECT_ID", "HADM_ID"],
        how="left",
        validate="one_to_one",
    )

    if merged["split"].isna().any():
        missing = merged.loc[merged["split"].isna(), ["SUBJECT_ID", "HADM_ID"]].head(10)
        raise ValueError(f"Some cohort rows are missing from split file:\n{missing}")

    train_idx = np.flatnonzero(merged["split"].eq("train").to_numpy())
    val_idx = np.flatnonzero(merged["split"].eq("val").to_numpy())

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"Split file must contain non-empty train and val rows. "
            f"Found train={len(train_idx)}, val={len(val_idx)}."
        )

    return train_idx, val_idx


def _copy_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _validate_training_controls(
    batch_size: int,
    gradient_accumulation_steps: int,
    optimizer_step_sleep_seconds: float,
) -> None:
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1. Received {batch_size}.")
    if gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps must be at least 1. "
            f"Received {gradient_accumulation_steps}."
        )
    if optimizer_step_sleep_seconds < 0:
        raise ValueError(
            "optimizer_step_sleep_seconds must be non-negative. "
            f"Received {optimizer_step_sleep_seconds}."
        )


def _accumulation_window_size(
    batch_idx: int,
    num_batches: int,
    gradient_accumulation_steps: int,
) -> int:
    window_start = (batch_idx // gradient_accumulation_steps) * gradient_accumulation_steps
    return min(gradient_accumulation_steps, num_batches - window_start)


def train_model(
    cohort_path: str,
    split_path: str,
    model_name: str,
    max_length: int,
    label_column: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    random_seed: int = 42,
    train_sampler=None,
    gradient_accumulation_steps: int = 1,
    optimizer_step_sleep_seconds: float = 0.0,
) -> str:
    _validate_training_controls(
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimizer_step_sleep_seconds=optimizer_step_sleep_seconds,
    )

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    tokenizer = load_tokenizer(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    dataset = MIMICDataset(cohort_path, tokenizer, max_length, label_column)
    train_idx, val_idx = _load_train_val_indices(dataset.df, split_path)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    effective_batch_size = batch_size * gradient_accumulation_steps
    print(
        "Training controls: "
        f"device={device}, micro_batch_size={batch_size}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}, "
        f"effective_batch_size={effective_batch_size}, "
        f"optimizer_step_sleep_seconds={optimizer_step_sleep_seconds}"
    )

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)

            epoch_loss += loss.item()
            accumulation_window_size = _accumulation_window_size(
                batch_idx,
                len(train_loader),
                gradient_accumulation_steps,
            )
            scaler.scale(loss / accumulation_window_size).backward()

            should_step = (batch_idx + 1) % gradient_accumulation_steps == 0
            is_last_batch = batch_idx + 1 == len(train_loader)
            if should_step or is_last_batch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if optimizer_step_sleep_seconds:
                    time.sleep(optimizer_step_sleep_seconds)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        epoch_loss /= len(train_loader)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = _copy_model_state(model)

    if best_model_state:
        model.load_state_dict(best_model_state)

    best_checkpoint_path = str(output_path / "best_model")
    model.save_pretrained(best_checkpoint_path)
    tokenizer.save_pretrained(best_checkpoint_path)

    return best_checkpoint_path


def train_model_with_weights(
    cohort_path: str,
    split_path: str,
    model_name: str,
    max_length: int,
    label_column: str,
    output_dir: str,
    sample_weights: np.ndarray,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    random_seed: int = 42,
    gradient_accumulation_steps: int = 1,
    optimizer_step_sleep_seconds: float = 0.0,
) -> str:
    """Train with sample weights for fairness mitigation."""
    _validate_training_controls(
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimizer_step_sleep_seconds=optimizer_step_sleep_seconds,
    )

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    tokenizer = load_tokenizer(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    dataset = MIMICDataset(cohort_path, tokenizer, max_length, label_column)
    train_idx, val_idx = _load_train_val_indices(dataset.df, split_path)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    if len(sample_weights) == len(dataset):
        train_weights = sample_weights[train_idx]
    elif len(sample_weights) == len(train_idx):
        train_weights = sample_weights
    else:
        raise ValueError(
            "sample_weights must align to either the full cohort or the train split. "
            f"Received {len(sample_weights)} weights for {len(dataset)} cohort rows "
            f"and {len(train_idx)} train rows."
        )

    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    effective_batch_size = batch_size * gradient_accumulation_steps
    print(
        "Training controls: "
        f"device={device}, micro_batch_size={batch_size}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}, "
        f"effective_batch_size={effective_batch_size}, "
        f"optimizer_step_sleep_seconds={optimizer_step_sleep_seconds}"
    )

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)

            epoch_loss += loss.item()
            accumulation_window_size = _accumulation_window_size(
                batch_idx,
                len(train_loader),
                gradient_accumulation_steps,
            )
            scaler.scale(loss / accumulation_window_size).backward()

            should_step = (batch_idx + 1) % gradient_accumulation_steps == 0
            is_last_batch = batch_idx + 1 == len(train_loader)
            if should_step or is_last_batch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if optimizer_step_sleep_seconds:
                    time.sleep(optimizer_step_sleep_seconds)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        epoch_loss /= len(train_loader)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = _copy_model_state(model)

    if best_model_state:
        model.load_state_dict(best_model_state)

    best_checkpoint_path = str(output_path / "best_model")
    model.save_pretrained(best_checkpoint_path)
    tokenizer.save_pretrained(best_checkpoint_path)

    return best_checkpoint_path
