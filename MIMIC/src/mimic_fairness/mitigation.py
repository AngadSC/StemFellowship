import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """Compute class weights to balance label distribution."""
    unique, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    weights = n / (len(unique) * counts)
    weight_map = {label: weight for label, weight in zip(unique, weights)}
    return np.array([weight_map[label] for label in labels])


def compute_subgroup_weights(
    df: pd.DataFrame,
    label_column: str,
    fairness_group_column: str = "fairness_group",
    upweight_factor: float = 2.0,
    max_weight: float = 10.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Upweight positive cases from underrepresented groups.

    Strategy: identify groups with lower positive label rates, upweight their positives.
    """
    weights = np.ones(len(df), dtype=float)

    group_positive_rates = df.groupby(fairness_group_column)[label_column].mean()
    mean_rate = group_positive_rates.mean()

    for group, group_rate in group_positive_rates.items():
        if group_rate < mean_rate:
            positive_mask = (df[fairness_group_column] == group) & (df[label_column] == 1)
            weights[positive_mask] *= upweight_factor

    if max_weight is not None:
        weights = np.minimum(weights, max_weight)

    if normalize:
        weights = weights / float(np.mean(weights))

    return weights


def apply_class_reweighting(
    dataset,
    label_column: str,
) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that balances class distribution."""
    labels = dataset.df[label_column].values
    weights = compute_class_weights(labels)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    return sampler


def apply_subgroup_reweighting(
    dataset,
    label_column: str,
    fairness_group_column: str = "fairness_group",
    upweight_factor: float = 2.0,
) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that upweights underrepresented subgroup positives."""
    weights = compute_subgroup_weights(
        dataset.df,
        label_column,
        fairness_group_column,
        upweight_factor,
    )
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    return sampler
