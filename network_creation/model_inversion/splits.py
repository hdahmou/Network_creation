"""Deterministic parameter-sample splitting for model inversion."""

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
    rows_by_parameter_row = _rows_by_parameter_row(row_index)
    diagnostic_parameter_rows = _diagnostic_parameter_rows(
        row_index,
        rows_by_parameter_row,
        config=config,
        rng=rng,
    )
    diagnostic_set = set(int(value) for value in diagnostic_parameter_rows)
    remaining_parameter_rows = np.asarray(
        [
            parameter_row
            for parameter_row in sorted(rows_by_parameter_row)
            if parameter_row not in diagnostic_set
        ],
        dtype=np.int64,
    )
    if remaining_parameter_rows.size <= 1:
        raise ValueError(
            "diagnostic split must leave training and validation parameter rows"
        )

    permutation = rng.permutation(remaining_parameter_rows)
    train_count = int(
        (1.0 - config.training.validation_fraction)
        * remaining_parameter_rows.size
    )
    validation_count = int(remaining_parameter_rows.size) - train_count
    if train_count <= 0 or validation_count <= 0:
        raise ValueError(
            "validation_fraction must leave at least one training and validation "
            "parameter row"
        )
    validation_parameter_rows = permutation[:validation_count]
    train_parameter_rows = permutation[validation_count:]
    diagnostic_indices = _rows_for_parameter_rows(
        rows_by_parameter_row,
        diagnostic_parameter_rows,
    )
    validation_indices = _rows_for_parameter_rows(
        rows_by_parameter_row,
        validation_parameter_rows,
    )
    train_indices = _rows_for_parameter_rows(
        rows_by_parameter_row,
        train_parameter_rows,
    )
    for split_name, indices in (
        ("training split", train_indices),
        ("validation split", validation_indices),
        ("diagnostic split", diagnostic_indices),
    ):
        _validate_n_agents_coverage(dataset, indices, split_name=split_name)
        _validate_category_coverage(dataset, indices, split_name=split_name)
    calibration_indices = _calibration_indices(
        dataset,
        train_indices,
        config=config,
        rng=rng,
    )
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


def _rows_by_parameter_row(
    row_index: InversionRowIndex,
) -> dict[int, NDArray[np.int64]]:
    grouped: dict[int, list[int]] = {}
    for local_index, parameter_row in enumerate(row_index.parameter_row_indices):
        grouped.setdefault(int(parameter_row), []).append(local_index)
    return {
        parameter_row: np.asarray(indices, dtype=np.int64)
        for parameter_row, indices in grouped.items()
    }


def _diagnostic_parameter_rows(
    row_index: InversionRowIndex,
    rows_by_parameter_row: dict[int, NDArray[np.int64]],
    *,
    config: InversionConfig,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for parameter_row, indices in rows_by_parameter_row.items():
        parameter_groups = {
            row_index.diagnostic_group_ids[int(index)] for index in indices
        }
        if len(parameter_groups) != 1:
            raise ValueError(
                f"parameter row {parameter_row} spans multiple diagnostic groups"
            )
        group = next(iter(parameter_groups))
        groups.setdefault(group, []).append(parameter_row)
    selected: list[int] = []
    count = int(config.diagnostic_holdout_count_per_group)
    for group in sorted(groups):
        values = np.asarray(sorted(groups[group]), dtype=np.int64)
        if values.size <= count:
            raise ValueError(
                "diagnostic_holdout_count_per_group must leave at least one "
                f"non-diagnostic parameter row in group {group}"
            )
        selected.extend(rng.choice(values, size=count, replace=False).astype(np.int64))
    if not selected:
        raise ValueError("diagnostic split is empty")
    return np.asarray(selected, dtype=np.int64)


def _rows_for_parameter_rows(
    rows_by_parameter_row: dict[int, NDArray[np.int64]],
    parameter_rows: NDArray[np.int64],
) -> NDArray[np.int64]:
    rows = [rows_by_parameter_row[int(parameter_row)] for parameter_row in parameter_rows]
    if not rows:
        return np.empty((0,), dtype=np.int64)
    return np.sort(np.concatenate(rows).astype(np.int64, copy=False))


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


def _validate_n_agents_coverage(
    dataset: ZarrInversionDataset,
    indices: NDArray[np.int64],
    *,
    split_name: str,
) -> None:
    if indices.size == 0:
        raise ValueError(f"{split_name} is empty")
    source_positions = dataset.row_index.kept_indices[indices]
    population_indices = dataset.source.population_indices[source_positions]
    expected = {int(value) for value in dataset.schema.n_agents}
    observed = {
        int(dataset.schema.n_agents[int(population_index)])
        for population_index in population_indices
    }
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(
            f"{split_name} must contain all n_agents values; missing {missing} after "
            "successful/non-finite row filtering; change seed, add data, or adjust "
            "split settings"
        )


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
