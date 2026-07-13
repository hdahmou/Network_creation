"""Deterministic row splitting for model inversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import InversionConfig
from .dataset import InversionRowIndex, ZarrInversionDataset


@dataclass(frozen=True)
class InversionSplit:
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    diagnostic_indices: NDArray[np.int64]
    calibration_indices: NDArray[np.int64]


@dataclass(frozen=True)
class PriorSamples:
    theta: NDArray[np.float32]
    source_indices: NDArray[np.int64]
    seed: int


def split_inversion_rows(
    row_index: InversionRowIndex,
    dataset: ZarrInversionDataset,
    config: InversionConfig,
) -> InversionSplit:
    rng = np.random.default_rng(int(config.training.seed))
    diagnostic_indices = _diagnostic_indices(row_index, config=config, rng=rng)
    all_indices = np.arange(len(dataset), dtype=np.int64)
    diagnostic_mask = np.zeros(len(dataset), dtype=np.bool_)
    diagnostic_mask[diagnostic_indices] = True
    remaining = all_indices[~diagnostic_mask]
    if remaining.size <= 1:
        raise ValueError("diagnostic split must leave training and validation rows")

    permutation = rng.permutation(remaining)
    train_count = int((1.0 - config.training.validation_fraction) * remaining.size)
    validation_count = int(remaining.size) - train_count
    if train_count <= 0 or validation_count <= 0:
        raise ValueError(
            "validation_fraction must leave at least one training and validation row"
        )
    validation_indices = np.sort(permutation[:validation_count].astype(np.int64))
    train_indices = np.sort(permutation[validation_count:].astype(np.int64))
    _validate_category_coverage(dataset, train_indices, split_name="training split")
    _validate_category_coverage(
        dataset,
        validation_indices,
        split_name="validation split",
    )
    calibration_indices = _calibration_indices(dataset, train_indices, config=config, rng=rng)
    _validate_category_coverage(
        dataset,
        calibration_indices,
        split_name="calibration split",
    )
    return InversionSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        diagnostic_indices=np.sort(diagnostic_indices),
        calibration_indices=np.sort(calibration_indices),
    )


def select_prior_samples(
    dataset: ZarrInversionDataset,
    split: InversionSplit,
    config: InversionConfig,
) -> PriorSamples:
    max_samples = int(config.output.prior_sample_count)
    n_keep = min(max_samples, int(split.train_indices.size))
    rng = np.random.default_rng(int(config.training.seed))
    if n_keep == int(split.train_indices.size):
        selected = np.asarray(split.train_indices, dtype=np.int64)
    else:
        selected = np.sort(
            rng.choice(split.train_indices, size=n_keep, replace=False).astype(
                np.int64
            )
        )
    theta = np.stack([dataset.theta(int(index)) for index in selected], axis=0).astype(
        np.float32,
    )
    return PriorSamples(
        theta=theta,
        source_indices=dataset.source_sample_ids_for_indices(selected),
        seed=int(config.training.seed),
    )


def _diagnostic_indices(
    row_index: InversionRowIndex,
    *,
    config: InversionConfig,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, group in enumerate(row_index.diagnostic_group_ids):
        groups.setdefault(tuple(group), []).append(index)
    selected: list[int] = []
    count = int(config.diagnostic_holdout_count_per_group)
    for group in sorted(groups):
        values = np.asarray(groups[group], dtype=np.int64)
        if values.size <= count:
            raise ValueError(
                "diagnostic_holdout_count_per_group must leave at least one "
                f"training row in group {group}"
            )
        selected.extend(rng.choice(values, size=count, replace=False).astype(np.int64))
    if not selected:
        raise ValueError("diagnostic split is empty")
    return np.asarray(selected, dtype=np.int64)


def _calibration_indices(
    dataset: ZarrInversionDataset,
    train_indices: NDArray[np.int64],
    *,
    config: InversionConfig,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    max_count = min(
        int(config.training.calibration_sample_count),
        int(train_indices.shape[0]),
    )
    if max_count <= 0:
        raise ValueError("calibration split is empty")
    required = _indices_covering_discrete_values(dataset, train_indices)
    if required.size > max_count:
        raise ValueError(
            "calibration_sample_count is too small to cover discrete theta categories"
        )
    selected = [int(index) for index in required.tolist()]
    selected_set = set(selected)
    for index in rng.permutation(train_indices):
        item = int(index)
        if item in selected_set:
            continue
        selected.append(item)
        selected_set.add(item)
        if len(selected) >= max_count:
            break
    return np.asarray(selected, dtype=np.int64)


def _validate_category_coverage(
    dataset: ZarrInversionDataset,
    indices: NDArray[np.int64],
    *,
    split_name: str,
) -> None:
    if not dataset.schema.discrete_theta_names:
        return
    if indices.size == 0:
        raise ValueError(f"{split_name} is empty")
    theta = np.stack([dataset.theta(int(index)) for index in indices], axis=0)
    discrete_start = len(dataset.schema.continuous_theta_names)
    for offset, name in enumerate(dataset.schema.discrete_theta_names):
        expected = set(dataset.schema.discrete_theta_values[name])
        observed = set(
            int(value)
            for value in np.rint(theta[:, discrete_start + offset]).astype(np.int64)
        )
        if observed != expected:
            raise ValueError(
                f"{split_name} must contain all discrete theta categories; "
                "change seed, add data, or adjust split settings"
            )


def _indices_covering_discrete_values(
    dataset: ZarrInversionDataset,
    indices: NDArray[np.int64],
) -> NDArray[np.int64]:
    if not dataset.schema.discrete_theta_names:
        return np.empty((0,), dtype=np.int64)
    required: set[int] = set()
    theta = np.stack([dataset.theta(int(index)) for index in indices], axis=0)
    discrete_start = len(dataset.schema.continuous_theta_names)
    for offset, name in enumerate(dataset.schema.discrete_theta_names):
        for category in dataset.schema.discrete_theta_values[name]:
            matches = np.nonzero(
                np.rint(theta[:, discrete_start + offset]).astype(np.int64)
                == int(category)
            )[0]
            if matches.size == 0:
                raise ValueError(
                    "training split must contain all discrete theta categories"
                )
            required.add(int(indices[int(matches[0])]))
    return np.asarray(sorted(required), dtype=np.int64)
