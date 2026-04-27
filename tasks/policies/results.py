"""Policy result loading as labeled xarray datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from ..dispatch.result_store import load_result_zarr

if TYPE_CHECKING:
    from .base import PolicyRun


@dataclass(frozen=True)
class PolicyOutput:
    timeseries: NDArray[np.float32]
    observable_names: tuple[str, ...]
    step_labels: NDArray[np.int64]


def load_policy_results(zarr_path: str | Path) -> xr.Dataset:
    raw = load_result_zarr(zarr_path)
    result_store_path = Path(raw.attrs["path"])
    metadata = raw.attrs["metadata"]
    attrs = raw.attrs

    task_name, task_class = _resolve_policy_task(attrs["task_name"])
    output_schema = list(attrs["output_schema"])
    input_index = np.asarray(raw.coords["input_index"].values, dtype=np.int64)
    input_coordinates = np.asarray(metadata["input_coordinates"], dtype=np.int64)
    row_coordinates = input_coordinates[input_index]
    n_agents_index = row_coordinates[:, 0]
    parameter_row = row_coordinates[:, 1]
    n_agents = np.asarray(metadata["n_agents"], dtype=np.float32)[n_agents_index]
    parameters = np.asarray(metadata["parameters"], dtype=np.float32)[
        n_agents_index,
        parameter_row,
    ]
    theta = np.concatenate([n_agents[:, None], parameters], axis=1)
    parameter_low = np.asarray(metadata["parameter_low"], dtype=np.float32)
    parameter_high = np.asarray(metadata["parameter_high"], dtype=np.float32)
    all_n_agents = np.asarray(metadata["n_agents"], dtype=np.float32)
    prior_low = np.concatenate(
        [np.asarray([np.min(all_n_agents)], dtype=np.float32), parameter_low],
    )
    prior_high = np.concatenate(
        [np.asarray([np.max(all_n_agents)], dtype=np.float32), parameter_high],
    )
    raw_outputs = {
        str(field["name"]): np.asarray(raw[str(field["name"])].values)
        for field in output_schema
    }
    output = task_class.build_output_group(raw_outputs)

    coords: dict[str, Any] = {
        "sample": np.arange(input_index.shape[0], dtype=np.int64),
        "input_index": ("sample", input_index),
        "parameter_row": ("sample", parameter_row),
        "theta_parameter": (
            "theta_parameter",
            ["n_agents", *list(attrs["parameter_names"])],
        ),
        "observable": ("observable", list(output.observable_names)),
    }
    coords["step"] = ("step", output.step_labels)
    data_vars: dict[str, Any] = {
        "theta": (("sample", "theta_parameter"), theta),
        "prior_low": (("theta_parameter",), prior_low),
        "prior_high": (("theta_parameter",), prior_high),
        "x": (
            ("sample", "step", "observable"),
            output.timeseries,
        ),
        "duration_ms": (
            ("sample",),
            np.asarray(raw["duration_ms"].values, dtype=np.float64),
        ),
    }

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "task_name": task_name,
            "result_store_path": str(result_store_path),
            "config_path": str(attrs["config_path"]),
            "average_completion_duration_ms": attrs["average_completion_duration_ms"],
            "has_failures": attrs["has_failures"],
        },
    )


def _resolve_policy_task(raw_task_name: object) -> tuple[str, type[PolicyRun]]:
    from tasks.catalog import TASKS

    task_name = str(raw_task_name)
    task_class = TASKS.get(task_name)
    if task_class is None:
        raise ValueError(f"Unsupported policy task: {task_name}")
    return task_name, task_class


__all__ = [
    "PolicyOutput",
    "load_policy_results",
]
