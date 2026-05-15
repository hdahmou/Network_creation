"""Prepare canonical policy result datasets for model inversion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
import xarray as xr

ObservationRepresentation = Literal["final", "timeseries"]

DEFAULT_DISCRETE_THETA_CANDIDATES = ("n_agents", "visibility_distance")
TIME_METADATA_ATTRS = (
    "normalize_time",
    "time_axis",
    "raw_step_labels",
    "common_normalized_time_max",
    "normalized_time_resolution",
)


def filter_inversion_dataset(
    ds: xr.Dataset,
    *,
    observers: Sequence[str],
    theta_parameters: Sequence[str] | None = None,
    input_parameters: Sequence[str] | None = None,
    drop_constant_theta: bool = True,
    x_representation: ObservationRepresentation = "final",
) -> xr.Dataset:
    available_theta_names = tuple(str(v) for v in ds.coords["theta_parameter"].values)
    theta_names = _resolve_names(
        theta_parameters,
        available_theta_names,
        kind="theta parameter",
    )
    observable_names = _resolve_names(
        observers,
        tuple(str(v) for v in ds.coords["observable"].values),
        kind="observable",
    )
    input_parameter_names = (
        ()
        if input_parameters is None
        else _resolve_names(
            input_parameters,
            available_theta_names,
            kind="input parameter",
        )
    )
    overlapping_names = tuple(
        name for name in input_parameter_names if name in set(theta_names)
    )
    if overlapping_names:
        raise ValueError(
            "input parameters cannot also be inferred theta parameters: "
            f"{list(overlapping_names)}"
        )

    theta = np.asarray(
        _stack_for_inversion(
            ds["theta"].sel(theta_parameter=list(theta_names)),
            feature_dims=("theta_parameter",),
        ).values,
        dtype=np.float32,
    )
    prior_low = np.asarray(
        ds["prior_low"].sel(theta_parameter=list(theta_names)).values,
        dtype=np.float32,
    )
    prior_high = np.asarray(
        ds["prior_high"].sel(theta_parameter=list(theta_names)).values,
        dtype=np.float32,
    )
    x_array = ds["x"].sel(observable=list(observable_names))
    raw_step_labels = np.asarray(ds.coords["step"].values)
    step_labels: NDArray[np.generic] | None = raw_step_labels
    final_step_label: int | float | None = None
    if x_representation == "final":
        x_array = x_array.isel(step=-1, drop=True)
        final_step_label = _numeric_scalar(raw_step_labels[-1])
        step_labels = None
    elif x_representation != "timeseries":
        raise ValueError(f"unsupported x representation: {x_representation}")
    feature_dims = ("observable",) if step_labels is None else ("step", "observable")
    x_stacked = _stack_for_inversion(x_array, feature_dims=feature_dims)
    x = np.asarray(x_stacked.values, dtype=np.float32)
    if input_parameter_names:
        input_parameter_values = np.asarray(
            _stack_for_inversion(
                ds["theta"].sel(theta_parameter=list(input_parameter_names)),
                feature_dims=("theta_parameter",),
            ).values,
            dtype=np.float32,
        )
        if input_parameter_values.shape[0] != x.shape[0]:
            raise ValueError("input parameter samples do not align with observations")
        if step_labels is None:
            x = np.concatenate((x, input_parameter_values), axis=1)
        else:
            repeated_inputs = np.repeat(
                input_parameter_values[:, None, :],
                x.shape[1],
                axis=1,
            )
            x = np.concatenate((x, repeated_inputs), axis=2)
    source_sample = np.arange(theta.shape[0], dtype=np.int64)

    kept_mask = np.isfinite(theta).all(axis=1)
    kept_mask &= np.isfinite(x).reshape(theta.shape[0], -1).all(axis=1)
    theta = theta[kept_mask]
    x = x[kept_mask]
    constant_theta: dict[str, float] = {}

    if drop_constant_theta:
        theta, theta_names, prior_low, prior_high, constant_theta = _drop_constant_theta(
            theta=theta,
            theta_names=theta_names,
            prior_low=prior_low,
            prior_high=prior_high,
        )

    coords: dict[str, Any] = {
        "sample": ("sample", source_sample[kept_mask]),
        "theta_parameter": ("theta_parameter", list(theta_names)),
        "observable": ("observable", list(observable_names + input_parameter_names)),
        "source_sample": ("source_sample", source_sample),
    }
    for coord_name in _sample_coordinate_names(x_stacked):
        if coord_name in {"sample", "source_sample"}:
            continue
        coord_values = np.asarray(x_stacked.coords[coord_name].values)
        if coord_values.shape == source_sample.shape:
            coords[coord_name] = ("sample", coord_values[kept_mask])

    if step_labels is None:
        x_dims = ("sample", "observable")
    else:
        x_dims = ("sample", "step", "observable")
        coords["step"] = ("step", step_labels)

    attrs = {
        "result_store_path": str(ds.attrs["result_store_path"]),
        "config_path": str(ds.attrs["config_path"]),
        "constant_theta": constant_theta,
        "has_failures": bool(ds.attrs["has_failures"]),
        "x_representation": x_representation,
        "observer_names": list(observable_names),
        "input_parameter_names": list(input_parameter_names),
        **_source_time_attrs(ds.attrs),
    }
    if final_step_label is not None:
        attrs["final_step_label"] = final_step_label

    return xr.Dataset(
        data_vars={
            "theta": (("sample", "theta_parameter"), theta),
            "x": (x_dims, x),
            "prior_low": ("theta_parameter", prior_low),
            "prior_high": ("theta_parameter", prior_high),
            "kept_mask": ("source_sample", kept_mask),
        },
        coords=coords,
        attrs=attrs,
    )


def resolve_discrete_theta_parameters(
    theta_names: Sequence[str],
    requested: Sequence[str] | None = None,
    *,
    default_candidates: Sequence[str] = DEFAULT_DISCRETE_THETA_CANDIDATES,
) -> tuple[str, ...]:
    """Resolve theta names that should be modelled categorically by MNPE.

    ``requested=None`` auto-detects known categorical theta parameters when
    present. Passing an empty sequence explicitly disables mixed-parameter
    inference.
    """

    available = tuple(str(value) for value in theta_names)
    names_to_resolve = tuple(default_candidates) if requested is None else tuple(requested)

    resolved: list[str] = []
    for raw_name in names_to_resolve:
        name = str(raw_name).strip()
        if name in available:
            if name not in resolved:
                resolved.append(name)
            continue

        matches = [
            available_name
            for available_name in available
            if available_name.startswith(f"{name}.")
        ]
        if len(matches) == 1:
            match = matches[0]
            if match not in resolved:
                resolved.append(match)
            continue
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous discrete theta parameter {name!r}; "
                f"choose one of {matches}"
            )

        if requested is None:
            continue
        raise ValueError(
            f"unknown discrete theta parameter {name!r}; "
            f"available values are {list(available)}"
        )

    return tuple(resolved)


def move_discrete_theta_to_end(
    dataset: xr.Dataset,
    *,
    discrete_theta_parameters: Sequence[str] | None,
) -> xr.Dataset:
    """Return a dataset ordered as ``continuous theta..., discrete theta...``.

    SBI's MNPE builder expects categorical/discrete theta columns to be the last
    columns.  This helper validates integer-valued discrete columns, reorders the
    theta coordinate, and stores mixed-parameter metadata in the returned dataset.
    """

    theta_names = tuple(str(value) for value in dataset.coords["theta_parameter"].values)
    discrete_names = resolve_discrete_theta_parameters(
        theta_names,
        requested=discrete_theta_parameters,
    )
    continuous_names = tuple(name for name in theta_names if name not in discrete_names)
    ordered_names = continuous_names + discrete_names

    reordered = dataset.sel(theta_parameter=list(ordered_names))
    attrs = dict(reordered.attrs)
    attrs["theta_original_order"] = list(theta_names)
    attrs["continuous_theta_parameters"] = list(continuous_names)
    attrs["discrete_theta_parameters"] = list(discrete_names)

    category_values: list[list[int]] = []
    category_counts: list[int] = []
    if discrete_names:
        category_values = _validate_and_collect_discrete_values(
            reordered,
            discrete_theta_parameters=discrete_names,
        )
        category_counts = [len(values) for values in category_values]

    attrs["discrete_theta_values"] = category_values
    attrs["num_categories_per_discrete_theta"] = category_counts
    return reordered.assign_attrs(attrs)


def inversion_dataset_metadata(dataset: xr.Dataset) -> dict[str, Any]:
    metadata = {
        "result_store_path": str(dataset.attrs["result_store_path"]),
        "config_path": str(dataset.attrs["config_path"]),
        "theta_names": [
            str(value) for value in dataset.coords["theta_parameter"].values
        ],
        "x_names": [str(value) for value in dataset.coords["observable"].values],
        "observer_names": [
            str(value) for value in dataset.attrs.get("observer_names", [])
        ],
        "input_parameter_names": [
            str(value) for value in dataset.attrs.get("input_parameter_names", [])
        ],
        "constant_theta": dict(dataset.attrs["constant_theta"]),
        "prior_low": np.asarray(dataset["prior_low"].values, dtype=np.float32).tolist(),
        "prior_high": np.asarray(dataset["prior_high"].values, dtype=np.float32).tolist(),
        "x_event_shape": list(dataset["x"].shape[1:]),
        "x_representation": dataset.attrs["x_representation"],
        "n_simulations": int(dataset.sizes["sample"]),
    }
    for key in (*TIME_METADATA_ATTRS, "final_step_label"):
        if key in dataset.attrs:
            metadata[key] = _json_value(dataset.attrs[key])
    if "step" in dataset.coords:
        metadata["step_labels"] = _numeric_list(dataset.coords["step"].values)

    if "discrete_theta_parameters" in dataset.attrs:
        metadata["continuous_theta_names"] = [
            str(value) for value in dataset.attrs.get("continuous_theta_parameters", [])
        ]
        metadata["discrete_theta_names"] = [
            str(value) for value in dataset.attrs.get("discrete_theta_parameters", [])
        ]
        metadata["theta_original_order"] = [
            str(value) for value in dataset.attrs.get("theta_original_order", [])
        ]
        metadata["discrete_theta_values"] = [
            [int(item) for item in values]
            for values in dataset.attrs.get("discrete_theta_values", [])
        ]
        metadata["num_categories_per_discrete_theta"] = [
            int(value)
            for value in dataset.attrs.get("num_categories_per_discrete_theta", [])
        ]

    return metadata


def _source_time_attrs(attrs: dict[Any, Any]) -> dict[str, Any]:
    return {
        key: _json_value(attrs[key])
        for key in TIME_METADATA_ATTRS
        if key in attrs
    }


def _json_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, (np.bool_, bool)):
            return bool(scalar)
        if isinstance(scalar, (np.integer, int)):
            return int(scalar)
        if isinstance(scalar, (np.floating, float)):
            return float(scalar)
        return scalar
    return _numeric_list(array)


def _numeric_list(values: Any) -> list[int | float]:
    array = np.asarray(values)
    return [_numeric_scalar(value) for value in array.reshape(-1)]


def _numeric_scalar(value: Any) -> int | float:
    scalar = np.asarray(value).item()
    if isinstance(scalar, (np.integer, int)):
        return int(scalar)
    return float(scalar)


def _stack_for_inversion(
    array: xr.DataArray,
    *,
    feature_dims: Sequence[str],
) -> xr.DataArray:
    sample_dims = tuple(dim for dim in array.dims if dim not in feature_dims)
    if not sample_dims:
        raise ValueError("inversion arrays must have at least one sample dimension")

    ordered = array.transpose(*sample_dims, *feature_dims)
    if sample_dims == ("sample",):
        return ordered
    return ordered.stack(sample=sample_dims).transpose("sample", *feature_dims)


def _sample_coordinate_names(array: xr.DataArray) -> tuple[str, ...]:
    return tuple(
        name
        for name, coord in array.coords.items()
        if coord.dims == ("sample",)
    )


def _drop_constant_theta(
    *,
    theta: NDArray[np.float32],
    theta_names: Sequence[str],
    prior_low: NDArray[np.float32],
    prior_high: NDArray[np.float32],
) -> tuple[
    NDArray[np.float32],
    tuple[str, ...],
    NDArray[np.float32],
    NDArray[np.float32],
    dict[str, float],
]:
    observed_low = np.nanmin(theta, axis=0)
    observed_high = np.nanmax(theta, axis=0)
    is_constant = np.isclose(observed_low, observed_high, rtol=0.0, atol=1e-12)
    keep = np.where(~is_constant)[0]
    constants = {
        name: float(observed_low[index])
        for index, name in enumerate(theta_names)
        if is_constant[index]
    }
    return (
        np.asarray(theta[:, keep], dtype=np.float32),
        tuple(theta_names[index] for index in keep),
        np.asarray(prior_low[keep], dtype=np.float32),
        np.asarray(prior_high[keep], dtype=np.float32),
        constants,
    )


def _resolve_names(
    requested: Sequence[str] | None,
    available: Sequence[str],
    *,
    kind: str,
) -> tuple[str, ...]:
    if requested is None:
        return tuple(available)

    resolved: list[str] = []
    for raw_name in requested:
        name = str(raw_name).strip()
        if name in available:
            resolved.append(name)
            continue

        matches = [
            candidate for candidate in available if candidate.startswith(f"{name}.")
        ]
        if len(matches) == 1:
            resolved.append(matches[0])
            continue
        if len(matches) > 1:
            raise ValueError(f"ambiguous {kind} {name!r}; choose one of {matches}")
        raise ValueError(
            f"unknown {kind} {name!r}; available values are {list(available)}"
        )

    return tuple(resolved)


def _validate_and_collect_discrete_values(
    dataset: xr.Dataset,
    *,
    discrete_theta_parameters: Sequence[str],
) -> list[list[int]]:
    values = np.asarray(
        dataset["theta"].sel(theta_parameter=list(discrete_theta_parameters)).values,
        dtype=np.float32,
    )
    if values.ndim == 1:
        values = values.reshape((-1, 1))

    if not np.isfinite(values).all():
        raise ValueError("discrete theta parameters must be finite")

    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=1e-6):
        bad_column = int(np.where(~np.isclose(values, rounded, rtol=0.0, atol=1e-6))[1][0])
        bad_name = str(discrete_theta_parameters[bad_column])
        raise ValueError(
            f"discrete theta parameter {bad_name!r} contains non-integer values"
        )

    categories: list[list[int]] = []
    for index in range(len(discrete_theta_parameters)):
        unique_values = np.unique(rounded[:, index].astype(np.int64))
        categories.append([int(value) for value in unique_values.tolist()])
    return categories
