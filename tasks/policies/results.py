"""Policy result loading as labeled xarray datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from ..execution.zarr_store import load_zarr_results
from .outputs import PolicyOutput, build_policy_outputs


@dataclass(frozen=True)
class _ResultCoordinates:
    n_agents_index: NDArray[np.int64]
    policy_proportion_index: NDArray[np.int64]
    parameter_row: NDArray[np.int64]


def load_policy_results(zarr_path: str | Path) -> xr.Dataset:
    raw = load_zarr_results(zarr_path)
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
    parameter_rows = np.arange(grid_shape[2], dtype=np.int64)
    output = build_policy_outputs(_raw_outputs(raw, output_schema))

    proportion_name = _policy_proportion_name(policy_config, parameter_names)
    keep_parameter_names, keep_parameter_indices = _theta_parameter_columns(
        parameter_names,
        policy_config=policy_config,
        proportion_name=proportion_name,
    )
    proportion_values = _proportion_values(
        raw_parameters,
        parameter_names,
        proportion_name=proportion_name,
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
    dense_dims = ("n_agents", proportion_name, "parameter_row")
    coords: dict[str, Any] = {
        "n_agents": ("n_agents", n_agents_values),
        proportion_name: (proportion_name, proportion_values),
        "parameter_row": ("parameter_row", parameter_rows),
        "input_index": (dense_dims, input_index_grid),
        "theta_parameter": (
            "theta_parameter",
            ["n_agents", *keep_parameter_names],
        ),
        "observable": ("observable", list(output.observable_names)),
        "step": ("step", output.step_labels),
    }
    data_vars: dict[str, Any] = {
        "theta": ((*dense_dims, "theta_parameter"), theta),
        "prior_low": (("theta_parameter",), prior_low),
        "prior_high": (("theta_parameter",), prior_high),
        "x": (
            (*dense_dims, "step", "observable"),
            x,
        ),
    }

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "result_store_path": str(result_store_path),
            "config_path": str(attrs["config_path"]),
            "has_failures": attrs["has_failures"],
        },
    )


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
    return {
        str(field["name"]): np.asarray(raw[str(field["name"])].values)
        for field in output_schema
    }


def _proportion_values(
    raw_parameters: NDArray[np.float32],
    parameter_names: list[str],
    *,
    proportion_name: str,
) -> NDArray[np.float32]:
    proportion_column = parameter_names.index(proportion_name)
    return raw_parameters[0, :, 0, proportion_column].astype(
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
    x = np.full(
        (*grid_shape, *output.timeseries.shape[1:]),
        np.nan,
        dtype=np.float32,
    )
    x[
        result_coordinates.n_agents_index,
        result_coordinates.policy_proportion_index,
        result_coordinates.parameter_row,
    ] = output.timeseries
    return x


def _policy_proportion_name(
    policy_config: Any,
    parameter_names: list[str],
) -> str:
    if isinstance(policy_config, dict):
        population = policy_config.get("population")
        if isinstance(population, dict):
            proportions = population.get("policy_proportions")
            if isinstance(proportions, dict) and "policy" in proportions:
                return f"{proportions['policy']}_proportion"

    matches = [name for name in parameter_names if name.endswith("_proportion")]
    if len(matches) != 1:
        raise ValueError(
            "could not infer policy proportion axis from result metadata"
        )
    return matches[0]


def _theta_parameter_columns(
    parameter_names: list[str],
    *,
    policy_config: Any,
    proportion_name: str,
) -> tuple[list[str], list[int]]:
    policy_ids: list[str] = []
    if isinstance(policy_config, dict):
        population = policy_config.get("population")
        if isinstance(population, dict):
            policies = population.get("policies")
            if isinstance(policies, list):
                for policy in policies:
                    if isinstance(policy, dict) and "id" in policy:
                        policy_ids.append(str(policy["id"]))

    bookkeeping_names = {proportion_name}
    bookkeeping_names.update(f"{policy_id}_count" for policy_id in policy_ids)
    keep = [
        (index, name)
        for index, name in enumerate(parameter_names)
        if name not in bookkeeping_names
    ]
    return [name for _, name in keep], [index for index, _ in keep]


__all__ = [
    "PolicyOutput",
    "load_policy_results",
]
