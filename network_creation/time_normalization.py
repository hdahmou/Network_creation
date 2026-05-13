"""Tensor helpers for normalized simulation time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

TIME_AXIS_ATTR = "activation_opportunities_per_agent"


@dataclass(frozen=True)
class NormalizedTimeGrid:
    raw_step_labels: NDArray[np.float64]
    step_labels: NDArray[np.float64]
    source_times: NDArray[np.float64]
    common_normalized_time_max: float
    normalized_time_resolution: float | None

    def attrs(self) -> dict[str, Any]:
        return {
            "normalize_time": True,
            "time_axis": TIME_AXIS_ATTR,
            "raw_step_labels": self.raw_step_labels.tolist(),
            "common_normalized_time_max": self.common_normalized_time_max,
            "normalized_time_resolution": self.normalized_time_resolution,
        }


@dataclass(frozen=True)
class NormalizedTimeTensor:
    values: NDArray[np.float32]
    grid: NormalizedTimeGrid


def normalized_source_times(
    raw_step_labels: ArrayLike,
    *,
    n_agents: ArrayLike,
    n_actors: int,
) -> NDArray[np.float64]:
    """Convert raw simulator steps to activation opportunities per agent."""

    raw_labels = _time_vector(raw_step_labels, name="raw_step_labels")
    agents = _n_agents_array(n_agents)
    actors = _positive_float(n_actors, name="n_actors")
    effective_actors = np.minimum(actors, agents)
    scale = effective_actors / agents
    return np.asarray(scale[..., None] * raw_labels, dtype=np.float64)


def common_normalized_time_grid(
    raw_step_labels: ArrayLike,
    *,
    n_agents_values: ArrayLike,
    n_actors: int,
) -> NormalizedTimeGrid:
    """Build the shared normalized-time grid for several population sizes."""

    raw_labels = _time_vector(raw_step_labels, name="raw_step_labels")
    agents = _n_agents_vector(n_agents_values)
    source_times = normalized_source_times(
        raw_labels,
        n_agents=agents,
        n_actors=n_actors,
    )
    common_max = float(np.min(source_times[:, -1]))

    # Use the coarsest normalized spacing so every population size can be
    # represented on the shared grid without extrapolating past common_max.
    step_deltas = np.diff(source_times, axis=1)
    positive_deltas = step_deltas[step_deltas > 0.0]
    if positive_deltas.size == 0:
        target_labels = np.asarray([common_max], dtype=np.float64)
        resolution: float | None = None
    else:
        resolution = float(np.max(positive_deltas))
        count = int(np.floor((common_max / resolution) + 1e-12))
        target_labels = np.arange(count + 1, dtype=np.float64) * resolution
        if not np.isclose(target_labels[-1], common_max, rtol=0.0, atol=1e-12):
            target_labels = np.concatenate(
                [target_labels, np.asarray([common_max], dtype=np.float64)]
            )

    return NormalizedTimeGrid(
        raw_step_labels=raw_labels,
        step_labels=target_labels,
        source_times=source_times,
        common_normalized_time_max=common_max,
        normalized_time_resolution=resolution,
    )


def normalize_time_tensor(
    values: ArrayLike,
    *,
    raw_step_labels: ArrayLike,
    n_agents: ArrayLike,
    n_actors: int,
    step_axis: int,
    n_agents_axis: int | None = None,
    target_times: ArrayLike | None = None,
) -> NormalizedTimeTensor:
    """Normalize one tensor or align many population slices on normalized time."""

    array = np.asarray(values, dtype=np.float32)
    step_axis = _normalize_axis(step_axis, array.ndim, name="step_axis")
    raw_labels = _time_vector(raw_step_labels, name="raw_step_labels")
    if array.shape[step_axis] != raw_labels.shape[0]:
        raise ValueError("raw_step_labels length must match the tensor step axis")

    if n_agents_axis is None:
        source_times = normalized_source_times(
            raw_labels,
            n_agents=_n_agents_scalar(n_agents),
            n_actors=n_actors,
        )
        step_labels = (
            source_times
            if target_times is None
            else _time_vector(target_times, name="target_times")
        )
        grid = _time_grid_from_target(raw_labels, source_times, step_labels)
        return NormalizedTimeTensor(
            values=interpolate_time_axis(
                array,
                source_times=source_times,
                target_times=grid.step_labels,
                step_axis=step_axis,
            ),
            grid=grid,
        )

    n_agents_axis = _normalize_axis(n_agents_axis, array.ndim, name="n_agents_axis")
    if n_agents_axis == step_axis:
        raise ValueError("n_agents_axis and step_axis must be different")

    agents = _n_agents_vector(n_agents)
    if array.shape[n_agents_axis] != agents.shape[0]:
        raise ValueError("n_agents length must match the tensor n_agents axis")

    grid = (
        common_normalized_time_grid(
            raw_labels,
            n_agents_values=agents,
            n_actors=n_actors,
        )
        if target_times is None
        else _time_grid_from_target(
            raw_labels,
            normalized_source_times(
                raw_labels,
                n_agents=agents,
                n_actors=n_actors,
            ),
            _time_vector(target_times, name="target_times"),
        )
    )

    moved = np.moveaxis(array, (n_agents_axis, step_axis), (0, 1))
    interpolated = np.empty(
        (moved.shape[0], grid.step_labels.shape[0], *moved.shape[2:]),
        dtype=np.float32,
    )
    for n_agents_index in range(moved.shape[0]):
        interpolated[n_agents_index] = interpolate_time_axis(
            moved[n_agents_index],
            source_times=grid.source_times[n_agents_index],
            target_times=grid.step_labels,
            step_axis=0,
        )

    return NormalizedTimeTensor(
        values=np.moveaxis(interpolated, (0, 1), (n_agents_axis, step_axis)),
        grid=grid,
    )


def interpolate_time_axis(
    values: ArrayLike,
    *,
    source_times: ArrayLike,
    target_times: ArrayLike,
    step_axis: int,
) -> NDArray[np.float32]:
    """Linearly interpolate a tensor along one time axis."""

    array = np.asarray(values, dtype=np.float32)
    step_axis = _normalize_axis(step_axis, array.ndim, name="step_axis")
    source = _time_vector(source_times, name="source_times")
    target = _time_vector(target_times, name="target_times")
    if array.shape[step_axis] != source.shape[0]:
        raise ValueError("source_times length must match the tensor step axis")
    if target[0] < source[0] - 1e-12 or target[-1] > source[-1] + 1e-12:
        raise ValueError("target_times extend beyond the source time horizon")

    moved = np.moveaxis(array, step_axis, 0)
    interpolated = _interpolate_leading_time_axis(
        moved,
        source_times=source,
        target_times=target,
    )
    return np.moveaxis(interpolated, 0, step_axis)


def _interpolate_leading_time_axis(
    values: NDArray[np.float32],
    *,
    source_times: NDArray[np.float64],
    target_times: NDArray[np.float64],
) -> NDArray[np.float32]:
    if source_times.shape[0] == 1:
        return np.repeat(values[:1], target_times.shape[0], axis=0)

    lower = np.searchsorted(source_times, target_times, side="right") - 1
    lower = np.clip(lower, 0, source_times.shape[0] - 1)
    upper = np.clip(lower + 1, 0, source_times.shape[0] - 1)

    lower_times = source_times[lower]
    upper_times = source_times[upper]
    denom = upper_times - lower_times
    weights = np.zeros(target_times.shape, dtype=np.float32)
    valid = denom > 0.0
    weights[valid] = (
        (target_times[valid] - lower_times[valid]) / denom[valid]
    ).astype(np.float32)
    weight_shape = (target_times.shape[0], *([1] * (values.ndim - 1)))
    weights = weights.reshape(weight_shape)

    lower_values = np.take(values, lower, axis=0)
    upper_values = np.take(values, upper, axis=0)
    return np.asarray(
        lower_values + weights * (upper_values - lower_values),
        dtype=np.float32,
    )


def _time_grid_from_target(
    raw_step_labels: NDArray[np.float64],
    source_times: NDArray[np.float64],
    target_times: NDArray[np.float64],
) -> NormalizedTimeGrid:
    return NormalizedTimeGrid(
        raw_step_labels=raw_step_labels,
        step_labels=target_times,
        source_times=np.asarray(source_times, dtype=np.float64),
        common_normalized_time_max=float(np.min(np.asarray(source_times)[..., -1])),
        normalized_time_resolution=_grid_resolution(target_times),
    )


def _time_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if array.shape[0] > 1 and not np.all(np.diff(array) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return array


def _n_agents_array(values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or not np.all(array > 0.0):
        raise ValueError("n_agents must contain only positive finite values")
    return array


def _n_agents_vector(values: ArrayLike) -> NDArray[np.float64]:
    array = _n_agents_array(values)
    if array.ndim != 1 or array.shape[0] == 0:
        raise ValueError("n_agents must be a non-empty one-dimensional array")
    return array


def _n_agents_scalar(value: ArrayLike) -> float:
    array = _n_agents_array(value)
    if array.ndim != 0:
        raise ValueError("n_agents must be scalar when n_agents_axis is not set")
    return float(array)


def _positive_float(value: int | float, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return parsed


def _normalize_axis(axis: int, ndim: int, *, name: str) -> int:
    if ndim == 0:
        raise ValueError("values must have at least one dimension")
    parsed = int(axis)
    if parsed < 0:
        parsed += ndim
    if parsed < 0 or parsed >= ndim:
        raise ValueError(f"{name} out of bounds for tensor with {ndim} dimensions")
    return parsed


def _grid_resolution(step_labels: NDArray[np.float64]) -> float | None:
    deltas = np.diff(step_labels)
    positive_deltas = deltas[deltas > 0.0]
    if positive_deltas.size == 0:
        return None
    return float(np.max(positive_deltas))


__all__ = [
    "NormalizedTimeGrid",
    "NormalizedTimeTensor",
    "TIME_AXIS_ATTR",
    "common_normalized_time_grid",
    "interpolate_time_axis",
    "normalize_time_tensor",
    "normalized_source_times",
]
