"""Lazy zarr-backed PyTorch dataset for inversion training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset

from .encoding import ObservationEncoder
from .schema import (
    InversionSchema,
    parameter_values_for_indices,
)
from .source import InversionSource


@dataclass(frozen=True)
class InversionRowIndex:
    kept_indices: NDArray[np.int64]
    source_sample_ids: NDArray[np.int64]
    diagnostic_group_ids: tuple[tuple[int, ...], ...]

    @property
    def row_count(self) -> int:
        return int(self.kept_indices.shape[0])


class ZarrInversionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        *,
        source: InversionSource,
        row_index: InversionRowIndex,
        schema: InversionSchema,
    ) -> None:
        self.source = source
        self.row_index = row_index
        self.schema = schema
        self.encoder = ObservationEncoder.from_schema(schema)

    def __len__(self) -> int:
        return int(self.row_index.kept_indices.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        theta, x = self.item_arrays(index)
        return (
            torch.as_tensor(theta, dtype=torch.float32),
            torch.as_tensor(x, dtype=torch.float32),
        )

    def item_arrays(self, index: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        source_position = int(self.row_index.kept_indices[int(index)])
        return self.item_arrays_from_source_position(source_position)

    def item_arrays_from_source_position(
        self,
        source_position: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        theta = self.theta_from_source_position(source_position)
        x = self.x_from_source_position(source_position)
        return theta, x

    def theta(self, index: int) -> NDArray[np.float32]:
        source_position = int(self.row_index.kept_indices[int(index)])
        return self.theta_from_source_position(source_position)

    def theta_from_source_position(self, source_position: int) -> NDArray[np.float32]:
        values = parameter_values_for_indices(
            self.source,
            self.schema.theta_order,
            source_positions=np.asarray([source_position], dtype=np.int64),
        )[0]
        return np.asarray(values, dtype=np.float32)

    def x_from_source_position(self, source_position: int) -> NDArray[np.float32]:
        zarr_row = int(self.source.result_rows[int(source_position)])
        input_values = self.input_values_from_source_position(source_position)
        n_agents_value = float(
            parameter_values_for_indices(
                self.source,
                ("n_agents",),
                source_positions=np.asarray([source_position], dtype=np.int64),
            )[0, 0]
        )
        observed = {
            name: np.asarray(
                self.source.observed_output_arrays[name][zarr_row],
                dtype=np.float32,
            )
            for name in self.schema.observed_outputs
        }
        return self.encoder.encode(
            observed,
            input_values,
            n_agents=n_agents_value,
        )

    def input_values_from_source_position(
        self,
        source_position: int,
    ) -> dict[str, float]:
        names = self.schema.continuous_input_names + self.schema.discrete_input_names
        values = parameter_values_for_indices(
            self.source,
            names,
            source_positions=np.asarray([source_position], dtype=np.int64),
        )[0]
        return {name: float(values[index]) for index, name in enumerate(names)}

    def source_sample_ids_for_indices(
        self,
        indices: NDArray[np.integer[Any]],
    ) -> NDArray[np.int64]:
        local = np.asarray(indices, dtype=np.int64)
        return np.asarray(self.row_index.source_sample_ids[local], dtype=np.int64)


def build_row_index(
    source: InversionSource,
    schema: InversionSchema,
) -> InversionRowIndex:
    row_count = int(source.input_index.shape[0])
    valid_mask = np.ones(row_count, dtype=np.bool_)
    names = (
        schema.continuous_theta_names
        + schema.discrete_theta_names
        + schema.continuous_input_names
        + schema.discrete_input_names
    )
    parameter_values = parameter_values_for_indices(source, names)
    if parameter_values.size:
        valid_mask &= np.isfinite(parameter_values).all(axis=1)

    for source_position in range(row_count):
        if not bool(valid_mask[source_position]):
            continue
        zarr_row = int(source.result_rows[source_position])
        for name in schema.observed_outputs:
            values = np.asarray(
                source.observed_output_arrays[name][zarr_row],
                dtype=np.float32,
            )
            if not np.isfinite(values).all():
                valid_mask[source_position] = False
                break

    kept_indices = np.nonzero(valid_mask)[0].astype(np.int64)
    if kept_indices.size == 0:
        raise ValueError("no finite inversion rows remain after filtering")
    diagnostic_group_ids = _diagnostic_groups(
        source,
        schema,
        kept_indices=kept_indices,
    )
    return InversionRowIndex(
        kept_indices=kept_indices,
        source_sample_ids=np.asarray(source.input_index[kept_indices], dtype=np.int64),
        diagnostic_group_ids=diagnostic_group_ids,
    )


def make_zarr_inversion_dataset(
    source: InversionSource,
    row_index: InversionRowIndex,
    schema: InversionSchema,
) -> ZarrInversionDataset:
    return ZarrInversionDataset(source=source, row_index=row_index, schema=schema)


def _diagnostic_groups(
    source: InversionSource,
    schema: InversionSchema,
    *,
    kept_indices: NDArray[np.int64],
) -> tuple[tuple[int, ...], ...]:
    group_names = schema.discrete_input_names + schema.discrete_theta_names
    if not group_names:
        return tuple((0,) for _ in kept_indices)
    values = parameter_values_for_indices(source, group_names, source_positions=kept_indices)
    groups: list[tuple[int, ...]] = []
    for row in values:
        rounded = np.rint(row)
        if not np.allclose(row, rounded, rtol=0.0, atol=1e-6):
            raise ValueError("diagnostic group values must be integer-coded")
        groups.append(tuple(int(value) for value in rounded.astype(np.int64).tolist()))
    return tuple(groups)
