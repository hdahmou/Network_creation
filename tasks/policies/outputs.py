"""Policy result tensor assembly."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .results import PolicyOutput


def _collect_timeseries_outputs(
    outputs: Mapping[str, NDArray[np.generic]],
) -> tuple[tuple[str, ...], list[NDArray[np.float32]]]:
    series_names: list[str] = []
    series_arrays: list[NDArray[np.float32]] = []
    expected_steps: int | None = None

    for name, values in outputs.items():
        array = np.asarray(values)
        if array.ndim != 2:
            continue
        if expected_steps is None:
            expected_steps = array.shape[1]
        elif array.shape[1] != expected_steps:
            raise ValueError("time-series outputs must share the same step axis length")
        series_names.append(name)
        series_arrays.append(np.asarray(array, dtype=np.float32))

    return tuple(series_names), series_arrays


def stack_result_tensors(
    outputs: Mapping[str, NDArray[np.generic]],
) -> tuple[NDArray[np.float32], tuple[str, ...]]:
    series_names, series_arrays = _collect_timeseries_outputs(outputs)
    if series_arrays:
        return (
            np.stack(series_arrays, axis=2).astype(np.float32, copy=False),
            series_names,
        )

    scalar_names: list[str] = []
    scalar_arrays: list[NDArray[np.float32]] = []
    for name, values in outputs.items():
        array = np.asarray(values)
        if array.ndim == 1:
            scalar_names.append(name)
            scalar_arrays.append(np.asarray(array, dtype=np.float32))

    if scalar_arrays:
        scalar_values = np.stack(scalar_arrays, axis=1).astype(
            np.float32,
            copy=False,
        )
        return scalar_values[:, None, :], tuple(scalar_names)

    raise ValueError("results must contain scalar or time-series outputs")


def build_policy_outputs(outputs: dict[str, NDArray[np.generic]]) -> PolicyOutput:
    from .results import PolicyOutput

    timeseries, observable_names = stack_result_tensors(outputs)
    return PolicyOutput(
        timeseries=timeseries,
        observable_names=observable_names,
        step_labels=np.arange(timeseries.shape[1], dtype=np.int64),
    )
