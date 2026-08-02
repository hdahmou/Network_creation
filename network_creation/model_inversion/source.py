"""Open zarr-backed simulation result stores for model inversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import zarr

from .config import InversionConfig, load_inversion_config


@dataclass(frozen=True)
class InversionSource:
    config: InversionConfig
    config_path: Path
    zarr_path: Path
    result_rows: NDArray[np.int64]
    observed_output_arrays: dict[str, Any]
    metadata_arrays: dict[str, NDArray[Any]]
    attrs: dict[str, Any]
    input_index: NDArray[np.int64]
    population_indices: NDArray[np.int64]


def load_inversion_source(config_path: str | Path) -> InversionSource:
    resolved_config_path = Path(config_path).expanduser().resolve()
    config = load_inversion_config(resolved_config_path)
    zarr_path = config.data.zarr_path.expanduser().resolve()
    if not zarr_path.exists():
        raise FileNotFoundError(f"Result store not found: {zarr_path}")

    root = cast(Any, zarr.open_group(zarr_path, mode="r"))
    attrs = dict(root.attrs)
    if attrs.get("sweep_layout") == "sampled_composition":
        raise ValueError(
            "model inversion does not support sampled-composition result stores"
        )
    results_group = root["results"]
    outputs_group = results_group["outputs"]
    metadata_group = root["metadata"]

    raw_input_index = np.asarray(results_group["input_index"], dtype=np.int64)
    order = np.argsort(raw_input_index, kind="stable")
    sorted_input_index = np.asarray(raw_input_index[order], dtype=np.int64)
    output_schema = list(attrs["output_schema"])
    output_arrays = {
        str(field["name"]): outputs_group[str(field["name"])]
        for field in output_schema
        if str(field["name"]) in outputs_group
    }
    metadata_arrays = {
        str(name): np.asarray(metadata_group[str(name)])
        for name in metadata_group.keys()
    }
    n_agents = np.asarray(metadata_arrays["n_agents"], dtype=np.int64)
    raw_step_horizons = np.asarray(
        metadata_arrays["raw_step_horizons"],
        dtype=np.int64,
    )
    if n_agents.ndim != 1 or n_agents.shape[0] == 0:
        raise ValueError("metadata/n_agents must be a non-empty vector")
    if raw_step_horizons.shape != n_agents.shape:
        raise ValueError(
            "metadata/raw_step_horizons must align exactly with metadata/n_agents"
        )

    input_coordinates = np.asarray(
        metadata_arrays["input_coordinates"],
        dtype=np.int64,
    )
    if input_coordinates.ndim != 2 or input_coordinates.shape[1] < 1:
        raise ValueError("metadata/input_coordinates must contain population indices")
    if sorted_input_index.size and (
        np.min(sorted_input_index) < 0
        or np.max(sorted_input_index) >= input_coordinates.shape[0]
    ):
        raise ValueError("results/input_index is outside metadata/input_coordinates")
    population_indices = np.asarray(
        input_coordinates[sorted_input_index, 0],
        dtype=np.int64,
    )
    if population_indices.size and (
        np.min(population_indices) < 0
        or np.max(population_indices) >= n_agents.shape[0]
    ):
        raise ValueError("result population index is outside metadata/n_agents")

    return InversionSource(
        config=config,
        config_path=resolved_config_path,
        zarr_path=zarr_path,
        result_rows=np.asarray(order, dtype=np.int64),
        observed_output_arrays=output_arrays,
        metadata_arrays=metadata_arrays,
        attrs=attrs,
        input_index=sorted_input_index,
        population_indices=population_indices,
    )
