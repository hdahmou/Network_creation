"""Policy result loading as labeled xarray datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from network_creation.policy_result_metadata import (
    configured_n_actors as _configured_n_actors,
    configured_output_step_labels as _configured_output_step_labels,
    policy_composition_axis_name as _policy_composition_axis_name,
    theta_parameter_columns as _theta_parameter_columns,
)
from network_creation.time_normalization import (
    common_normalized_time_grid,
    normalize_time_tensor,
)

from ..execution.zarr_store import load_zarr_results
from .outputs import (
    PolicyOutput,
    SAMPLED_AGENT_PARAMETER_NAMES_ATTR,
    SAMPLED_AGENT_PARAMETERS_FIELD,
    build_policy_outputs,
)

AGENT_PAYOFF_OBSERVER_NAME = "agent_payoff"
POLICY_WELFARE_OBSERVER_NAME = "policy_welfare"


@dataclass(frozen=True)
class _ResultCoordinates:
    n_agents_index: NDArray[np.int64]
    policy_proportion_index: NDArray[np.int64]
    parameter_row: NDArray[np.int64]


def load_policy_results(
    zarr_path: str | Path,
    *,
    stats: str | Sequence[str] | None = None,
    load_sampled_agent_parameters: bool = False,
    normalize_time: bool = False,
) -> xr.Dataset:
    requested_stats = _requested_stats(stats)
    if (
        requested_stats is not None
        and SAMPLED_AGENT_PARAMETERS_FIELD in requested_stats
    ):
        raise ValueError(
            f"{SAMPLED_AGENT_PARAMETERS_FIELD!r} is controlled by "
            "load_sampled_agent_parameters; set "
            "load_sampled_agent_parameters=True instead of including it in stats"
        )
    load_outputs = requested_stats
    if load_sampled_agent_parameters and load_outputs is not None:
        load_outputs = (*load_outputs, SAMPLED_AGENT_PARAMETERS_FIELD)
    skip_outputs = (
        ()
        if load_sampled_agent_parameters
        else (SAMPLED_AGENT_PARAMETERS_FIELD,)
    )
    raw = load_zarr_results(
        zarr_path,
        skip_outputs=skip_outputs,
        load_outputs=load_outputs,
    )
    result_store_path = Path(raw.attrs["path"])
    metadata = raw.attrs["metadata"]
    attrs = raw.attrs

    output_schema = list(attrs["output_schema"])
    raw_parameters = np.asarray(metadata["parameters"], dtype=np.float32)
    parameter_names = list(attrs["parameter_names"])
    policy_config = attrs.get("policy_config", {})
    result_coordinates = _result_coordinates(raw, metadata)
    grid_shape = _parameter_grid_shape(raw_parameters)
    n_agents_values = np.asarray(metadata["n_agents"], dtype=np.int64)
    parameter_rows: NDArray[np.int64] = np.arange(grid_shape[2], dtype=np.int64)
    raw_outputs = _raw_outputs(raw, output_schema)
    sampled_agent_parameters = raw_outputs.get(SAMPLED_AGENT_PARAMETERS_FIELD)
    agent_payoff_outputs = _agent_payoff_outputs(raw_outputs)
    policy_welfare_outputs = _policy_welfare_outputs(raw_outputs)
    regular_outputs = {
        name: values
        for name, values in raw_outputs.items()
        if name not in agent_payoff_outputs
        and name not in policy_welfare_outputs
        and name != SAMPLED_AGENT_PARAMETERS_FIELD
    }
    step_labels = _shared_step_labels(policy_config, raw_outputs)
    output = _build_policy_output_or_empty(
        regular_outputs,
        all_outputs=raw_outputs,
        step_labels=step_labels,
        result_count=int(raw.sizes["result"]),
    )

    composition_axis_name = _policy_composition_axis_name(
        policy_config,
        parameter_names,
    )
    keep_parameter_names, keep_parameter_indices = _theta_parameter_columns(
        parameter_names,
        policy_config=policy_config,
        composition_axis_name=composition_axis_name,
    )
    composition_axis_values = _composition_axis_values(
        raw_parameters,
        parameter_names,
        composition_axis_name=composition_axis_name,
    )
    theta, prior_low, prior_high = _theta_arrays(
        metadata,
        raw_parameters=raw_parameters,
        grid_shape=grid_shape,
        n_agents_values=n_agents_values,
        keep_parameter_indices=keep_parameter_indices,
    )
    x = _dense_output_grid(
        grid_shape=grid_shape,
        result_coordinates=result_coordinates,
        output=output,
    )
    input_index_grid = np.arange(np.prod(grid_shape), dtype=np.int64).reshape(
        grid_shape,
    )
    dense_dims = ("n_agents", composition_axis_name, "parameter_row")
    coords: dict[str, Any] = {
        "n_agents": ("n_agents", n_agents_values),
        composition_axis_name: (composition_axis_name, composition_axis_values),
        "parameter_row": ("parameter_row", parameter_rows),
        "input_index": (dense_dims, input_index_grid),
        "theta_parameter": (
            "theta_parameter",
            ["n_agents", *keep_parameter_names],
        ),
        "observable": ("observable", list(output.observable_names)),
        "step": ("step", output.step_labels),
    }
    agent_slot_count = _agent_slot_count(agent_payoff_outputs)
    if agent_slot_count is not None:
        coords["agent_slot"] = (
            "agent_slot",
            np.arange(agent_slot_count, dtype=np.int64),
        )
    policy_class_count = _policy_class_count(policy_welfare_outputs)
    if policy_class_count is not None:
        coords["policy_class"] = (
            "policy_class",
            _policy_class_labels(policy_config, policy_class_count),
        )
    episode_count = _tensor_episode_count(
        {**agent_payoff_outputs, **policy_welfare_outputs}
    )
    if episode_count is not None:
        coords["episode"] = ("episode", np.arange(episode_count, dtype=np.int64))
    if sampled_agent_parameters is not None:
        agent_parameter_names = _sampled_agent_parameter_names(attrs)
        _validate_sampled_agent_parameter_names(
            sampled_agent_parameters,
            agent_parameter_names,
        )
        coords["agent_index"] = (
            "agent_index",
            np.arange(np.asarray(sampled_agent_parameters).shape[1], dtype=np.int64),
        )
        coords["agent_parameter"] = (
            "agent_parameter",
            list(agent_parameter_names),
        )

    data_vars: dict[str, Any] = {
        "theta": ((*dense_dims, "theta_parameter"), theta),
        "prior_low": (("theta_parameter",), prior_low),
        "prior_high": (("theta_parameter",), prior_high),
        "x": (
            (*dense_dims, "step", "observable"),
            x,
        ),
    }
    data_vars.update(
        _dense_agent_payoff_data_vars(
            grid_shape=grid_shape,
            dense_dims=dense_dims,
            result_coordinates=result_coordinates,
            outputs=agent_payoff_outputs,
        )
    )
    data_vars.update(
        _dense_policy_welfare_data_vars(
            grid_shape=grid_shape,
            dense_dims=dense_dims,
            result_coordinates=result_coordinates,
            outputs=policy_welfare_outputs,
        )
    )
    if sampled_agent_parameters is not None:
        data_vars[SAMPLED_AGENT_PARAMETERS_FIELD] = (
            (*dense_dims, "agent_index", "agent_parameter"),
            _dense_result_grid(
                grid_shape=grid_shape,
                result_coordinates=result_coordinates,
                values=sampled_agent_parameters,
            ),
        )

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "result_store_path": str(result_store_path),
            "config_path": str(attrs["config_path"]),
            "has_failures": attrs["has_failures"],
        },
    )
    if not normalize_time:
        return dataset
    return _normalize_time_axis(dataset, policy_config=policy_config)


def _requested_stats(stats: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if stats is None:
        return None
    if isinstance(stats, str):
        return (stats,)
    return tuple(str(name) for name in stats)


def _result_coordinates(
    raw: xr.Dataset,
    metadata: dict[str, Any],
) -> _ResultCoordinates:
    input_index = np.asarray(raw.coords["input_index"].values, dtype=np.int64)
    input_coordinates = np.asarray(metadata["input_coordinates"], dtype=np.int64)
    row_coordinates = input_coordinates[input_index]
    return _ResultCoordinates(
        n_agents_index=row_coordinates[:, 0],
        policy_proportion_index=row_coordinates[:, 1],
        parameter_row=row_coordinates[:, 2],
    )


def _parameter_grid_shape(
    raw_parameters: NDArray[np.float32],
) -> tuple[int, int, int]:
    grid_shape = raw_parameters.shape[:3]
    if len(grid_shape) != 3:
        raise ValueError("metadata parameters must have n_agents/proportion/row axes")
    return int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])


def _raw_outputs(
    raw: xr.Dataset,
    output_schema: list[Any],
) -> dict[str, NDArray[np.generic]]:
    outputs: dict[str, NDArray[np.generic]] = {}
    for field in output_schema:
        name = str(field["name"])
        if name not in raw.data_vars:
            continue
        outputs[name] = np.asarray(raw[name].values)
    return outputs


def _agent_payoff_outputs(
    outputs: Mapping[str, NDArray[np.generic]],
) -> dict[str, NDArray[np.generic]]:
    return {
        name: values
        for name, values in outputs.items()
        if _is_agent_payoff_output(name)
    }


def _policy_welfare_outputs(
    outputs: Mapping[str, NDArray[np.generic]],
) -> dict[str, NDArray[np.generic]]:
    return {
        name: values
        for name, values in outputs.items()
        if _is_policy_welfare_output(name)
    }


def _validate_sampled_agent_parameter_names(
    values: NDArray[np.generic],
    names: Sequence[str],
) -> None:
    array = np.asarray(values)
    if array.ndim != 3:
        raise ValueError(
            f"{SAMPLED_AGENT_PARAMETERS_FIELD} outputs must have rank 3"
        )
    if len(names) != array.shape[2]:
        raise ValueError(
            f"{SAMPLED_AGENT_PARAMETER_NAMES_ATTR} length must match "
            f"{SAMPLED_AGENT_PARAMETERS_FIELD} parameter axis"
        )


def _sampled_agent_parameter_names(attrs: Mapping[str, Any]) -> tuple[str, ...]:
    raw_names = attrs.get(SAMPLED_AGENT_PARAMETER_NAMES_ATTR, ())
    return tuple(str(name) for name in cast(Sequence[Any], raw_names))


def _is_agent_payoff_output(name: str) -> bool:
    return str(name).split(".", maxsplit=1)[0] == AGENT_PAYOFF_OBSERVER_NAME


def _is_policy_welfare_output(name: str) -> bool:
    return str(name).split(".", maxsplit=1)[0] == POLICY_WELFARE_OBSERVER_NAME


def _agent_slot_count(
    outputs: Mapping[str, NDArray[np.generic]],
) -> int | None:
    if not outputs:
        return None
    counts = {int(np.asarray(values).shape[-1]) for values in outputs.values()}
    if len(counts) != 1:
        raise ValueError("agent_payoff outputs must share the same agent axis length")
    return counts.pop()


def _policy_class_count(
    outputs: Mapping[str, NDArray[np.generic]],
) -> int | None:
    if not outputs:
        return None
    counts = {int(np.asarray(values).shape[-1]) for values in outputs.values()}
    if len(counts) != 1:
        raise ValueError("policy_welfare outputs must share the same policy axis length")
    return counts.pop()


def _policy_class_labels(
    policy_config: Any,
    policy_class_count: int,
) -> NDArray[np.str_]:
    policy_ids: list[str] = []
    if isinstance(policy_config, Mapping):
        population = policy_config.get("population")
        if isinstance(population, Mapping):
            policies = population.get("policies")
            if isinstance(policies, Sequence):
                for policy in policies:
                    if isinstance(policy, Mapping) and "id" in policy:
                        policy_ids.append(str(policy["id"]))
    if len(policy_ids) != policy_class_count:
        policy_ids = [f"policy_{index}" for index in range(policy_class_count)]
    return np.asarray(policy_ids, dtype=np.str_)


def _tensor_episode_count(
    outputs: Mapping[str, NDArray[np.generic]],
) -> int | None:
    counts = {
        int(np.asarray(values).shape[1])
        for values in outputs.values()
        if np.asarray(values).ndim == 4
    }
    if not counts:
        return None
    if len(counts) != 1:
        raise ValueError("agent_payoff runs outputs must share the same episode count")
    return counts.pop()


def _infer_result_count(outputs: Mapping[str, NDArray[np.generic]]) -> int:
    first = next(iter(outputs.values()), None)
    if first is None:
        return 0
    return int(np.asarray(first).shape[0])


def _infer_step_count(outputs: Mapping[str, NDArray[np.generic]]) -> int:
    for name, values in outputs.items():
        step_axis_length = _step_axis_length(name, values)
        if step_axis_length is not None:
            return step_axis_length
    return 1


def _build_policy_output_or_empty(
    outputs: Mapping[str, NDArray[np.generic]],
    *,
    all_outputs: Mapping[str, NDArray[np.generic]],
    step_labels: NDArray[np.int64] | None,
    result_count: int | None = None,
) -> PolicyOutput:
    if outputs:
        labels = step_labels if _has_step_axis(outputs) else None
        return build_policy_outputs(dict(outputs), step_labels=labels)

    labels = (
        np.arange(_infer_step_count(all_outputs), dtype=np.int64)
        if step_labels is None
        else np.asarray(step_labels, dtype=np.int64)
    )
    if result_count is None:
        result_count = _infer_result_count(all_outputs)
    return PolicyOutput(
        timeseries=np.empty((result_count, labels.shape[0], 0), dtype=np.float32),
        observable_names=(),
        step_labels=labels,
    )


def _shared_step_labels(
    policy_config: Any,
    outputs: Mapping[str, NDArray[np.generic]],
) -> NDArray[np.int64] | None:
    step_axis_lengths = {
        name: step_axis_length
        for name, values in outputs.items()
        if (step_axis_length := _step_axis_length(name, values)) is not None
    }
    if not step_axis_lengths or not isinstance(policy_config, Mapping):
        return None

    labels_by_output = _configured_output_step_labels(policy_config)
    shared: NDArray[np.int64] | None = None
    for name, step_axis_length in step_axis_lengths.items():
        labels = labels_by_output.get(name)
        if labels is None:
            return None
        if labels.shape != (step_axis_length,):
            raise ValueError(
                f"configured step labels for {name!r} do not match output shape"
            )
        if shared is None:
            shared = labels
        elif not np.array_equal(shared, labels):
            raise ValueError("time-series outputs must share the same step labels")
    return shared


def _has_step_axis(outputs: Mapping[str, NDArray[np.generic]]) -> bool:
    return any(np.asarray(values).ndim == 2 for values in outputs.values())


def _step_axis_length(
    name: str,
    values: NDArray[np.generic],
) -> int | None:
    array = np.asarray(values)
    if array.ndim == 2:
        return int(array.shape[1])
    if _is_agent_payoff_output(name):
        if array.ndim == 3:
            return int(array.shape[1])
        if array.ndim == 4:
            return int(array.shape[2])
    if _is_policy_welfare_output(name):
        if array.ndim == 3:
            return int(array.shape[1])
        if array.ndim == 4:
            return int(array.shape[2])
    return None


def _composition_axis_values(
    raw_parameters: NDArray[np.float32],
    parameter_names: list[str],
    *,
    composition_axis_name: str,
) -> NDArray[np.float32]:
    composition_column = parameter_names.index(composition_axis_name)
    return raw_parameters[0, :, 0, composition_column].astype(
        np.float32,
        copy=False,
    )


def _theta_arrays(
    metadata: dict[str, Any],
    *,
    raw_parameters: NDArray[np.float32],
    grid_shape: tuple[int, int, int],
    n_agents_values: NDArray[np.int64],
    keep_parameter_indices: list[int],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    n_agents_float = n_agents_values.astype(np.float32)
    kept_parameters = raw_parameters[..., keep_parameter_indices]
    n_agents_grid = np.broadcast_to(
        n_agents_float[:, None, None, None],
        (*grid_shape, 1),
    )
    theta = np.concatenate([n_agents_grid, kept_parameters], axis=-1).astype(
        np.float32,
        copy=False,
    )

    parameter_low = np.asarray(metadata["parameter_low"], dtype=np.float32)[
        keep_parameter_indices
    ]
    parameter_high = np.asarray(metadata["parameter_high"], dtype=np.float32)[
        keep_parameter_indices
    ]
    prior_low = np.concatenate(
        [np.asarray([np.min(n_agents_float)], dtype=np.float32), parameter_low],
    )
    prior_high = np.concatenate(
        [np.asarray([np.max(n_agents_float)], dtype=np.float32), parameter_high],
    )
    return theta, prior_low, prior_high


def _dense_output_grid(
    *,
    grid_shape: tuple[int, int, int],
    result_coordinates: _ResultCoordinates,
    output: PolicyOutput,
) -> NDArray[np.float32]:
    return _dense_result_grid(
        grid_shape=grid_shape,
        result_coordinates=result_coordinates,
        values=output.timeseries,
    )


def _dense_agent_payoff_data_vars(
    *,
    grid_shape: tuple[int, int, int],
    dense_dims: tuple[str, str, str],
    result_coordinates: _ResultCoordinates,
    outputs: Mapping[str, NDArray[np.generic]],
) -> dict[str, Any]:
    data_vars: dict[str, Any] = {}
    for name, values in outputs.items():
        array = np.asarray(values, dtype=np.float32)
        dense = _dense_result_grid(
            grid_shape=grid_shape,
            result_coordinates=result_coordinates,
            values=array,
        )
        if array.ndim == 3:
            data_vars[name] = ((*dense_dims, "step", "agent_slot"), dense)
        elif array.ndim == 4:
            data_vars[name] = (
                (*dense_dims, "episode", "step", "agent_slot"),
                dense,
            )
        else:
            raise ValueError(
                f"agent_payoff outputs must have rank 3 or 4, got {array.ndim}"
            )
    return data_vars


def _dense_policy_welfare_data_vars(
    *,
    grid_shape: tuple[int, int, int],
    dense_dims: tuple[str, str, str],
    result_coordinates: _ResultCoordinates,
    outputs: Mapping[str, NDArray[np.generic]],
) -> dict[str, Any]:
    data_vars: dict[str, Any] = {}
    for name, values in outputs.items():
        array = np.asarray(values, dtype=np.float32)
        dense = _dense_result_grid(
            grid_shape=grid_shape,
            result_coordinates=result_coordinates,
            values=array,
        )
        if array.ndim == 3:
            data_vars[name] = ((*dense_dims, "step", "policy_class"), dense)
        elif array.ndim == 4:
            data_vars[name] = (
                (*dense_dims, "episode", "step", "policy_class"),
                dense,
            )
        else:
            raise ValueError(
                f"policy_welfare outputs must have rank 3 or 4, got {array.ndim}"
            )
    return data_vars


def _dense_result_grid(
    *,
    grid_shape: tuple[int, int, int],
    result_coordinates: _ResultCoordinates,
    values: NDArray[np.generic],
) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    dense = np.full(
        (*grid_shape, *array.shape[1:]),
        np.nan,
        dtype=np.float32,
    )
    dense[
        result_coordinates.n_agents_index,
        result_coordinates.policy_proportion_index,
        result_coordinates.parameter_row,
    ] = array
    return dense


def _normalize_time_axis(
    dataset: xr.Dataset,
    *,
    policy_config: Any,
) -> xr.Dataset:
    """Return a copy of ``dataset`` resampled onto normalized simulation time."""
    if "step" not in dataset.coords:
        raise ValueError("cannot normalize time without a step coordinate")
    if not isinstance(policy_config, Mapping):
        raise ValueError("cannot normalize time without policy configuration metadata")

    raw_step_labels = np.asarray(dataset.coords["step"].values, dtype=np.float64)
    n_agents_values = np.asarray(dataset.coords["n_agents"].values, dtype=np.float64)
    n_actors = _configured_n_actors(policy_config)
    grid = common_normalized_time_grid(
        raw_step_labels,
        n_agents_values=n_agents_values,
        n_actors=n_actors,
    )

    data_vars: dict[str, Any] = {}
    for raw_name, array in dataset.data_vars.items():
        name = str(raw_name)
        if "step" in array.dims:
            if "n_agents" not in array.dims:
                raise ValueError(
                    f"cannot normalize time for {name!r} without an n_agents dimension"
                )
            data_vars[name] = (
                array.dims,
                normalize_time_tensor(
                    array.values,
                    raw_step_labels=raw_step_labels,
                    n_agents=n_agents_values,
                    n_actors=n_actors,
                    n_agents_axis=array.dims.index("n_agents"),
                    step_axis=array.dims.index("step"),
                    target_times=grid.step_labels,
                ).values,
            )
        else:
            data_vars[name] = (array.dims, np.asarray(array.values))

    coords: dict[str, Any] = {}
    for raw_name, coord in dataset.coords.items():
        name = str(raw_name)
        if name != "step":
            coords[name] = (coord.dims, np.asarray(coord.values))
    coords["step"] = ("step", grid.step_labels)

    attrs = dict(dataset.attrs)
    attrs.update(grid.attrs())
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


__all__ = [
    "PolicyOutput",
    "load_policy_results",
]
