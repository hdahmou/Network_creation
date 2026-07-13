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


def load_inversion_source(config_path: str | Path) -> InversionSource:
    resolved_config_path = Path(config_path).expanduser().resolve()
    config = load_inversion_config(resolved_config_path)
    zarr_path = config.data.zarr_path.expanduser().resolve()
    if not zarr_path.exists():
        raise FileNotFoundError(f"Result store not found: {zarr_path}")

    root = cast(Any, zarr.open_group(zarr_path, mode="r"))
    attrs = dict(root.attrs)
    results_group = root["results"]
    outputs_group = results_group["outputs"]
    metadata_group = root["metadata"]

    raw_input_index = np.asarray(results_group["input_index"], dtype=np.int64)
    order = np.argsort(raw_input_index)
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

    return InversionSource(
        config=config,
        config_path=resolved_config_path,
        zarr_path=zarr_path,
        result_rows=np.asarray(order, dtype=np.int64),
        observed_output_arrays=output_arrays,
        metadata_arrays=metadata_arrays,
        attrs=attrs,
        input_index=sorted_input_index,
    )
