"""Prepare canonical policy result datasets for model inversion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
import xarray as xr

ObservationRepresentation = Literal["final", "timeseries"]


def filter_inversion_dataset(
    ds: xr.Dataset,
    *,
    observers: Sequence[str],
    theta_parameters: Sequence[str] | None = None,
    drop_constant_theta: bool = True,
    x_representation: ObservationRepresentation = "final",
) -> xr.Dataset:
    theta_names = _resolve_names(
        theta_parameters,
        tuple(str(v) for v in ds.coords["theta_parameter"].values),
        kind="theta parameter",
    )
    observable_names = _resolve_names(
        observers,
        tuple(str(v) for v in ds.coords["observable"].values),
        kind="observable",
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
    step_labels: tuple[int, ...] | None = tuple(int(v) for v in ds.coords["step"].values)
    if x_representation == "final":
        x_array = x_array.isel(step=-1, drop=True)
        step_labels = None
    elif x_representation != "timeseries":
        raise ValueError(f"unsupported x representation: {x_representation}")
    feature_dims = ("observable",) if step_labels is None else ("step", "observable")
    x_stacked = _stack_for_inversion(x_array, feature_dims=feature_dims)
    x = np.asarray(x_stacked.values, dtype=np.float32)
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
        "observable": ("observable", list(observable_names)),
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
        coords["step"] = ("step", list(step_labels))

    return xr.Dataset(
        data_vars={
            "theta": (("sample", "theta_parameter"), theta),
            "x": (x_dims, x),
            "prior_low": ("theta_parameter", prior_low),
            "prior_high": ("theta_parameter", prior_high),
            "kept_mask": ("source_sample", kept_mask),
        },
        coords=coords,
        attrs={
            "result_store_path": str(ds.attrs["result_store_path"]),
            "config_path": str(ds.attrs["config_path"]),
            "constant_theta": constant_theta,
            "has_failures": bool(ds.attrs["has_failures"]),
            "x_representation": x_representation,
        },
    )


def inversion_dataset_metadata(dataset: xr.Dataset) -> dict[str, Any]:
    metadata = {
        "result_store_path": str(dataset.attrs["result_store_path"]),
        "config_path": str(dataset.attrs["config_path"]),
        "theta_names": [
            str(value) for value in dataset.coords["theta_parameter"].values
        ],
        "x_names": [str(value) for value in dataset.coords["observable"].values],
        "constant_theta": dict(dataset.attrs["constant_theta"]),
        "prior_low": np.asarray(dataset["prior_low"].values, dtype=np.float32).tolist(),
        "prior_high": np.asarray(dataset["prior_high"].values, dtype=np.float32).tolist(),
        "x_event_shape": list(dataset["x"].shape[1:]),
        "x_representation": dataset.attrs["x_representation"],
        "n_simulations": int(dataset.sizes["sample"]),
    }
    if "step" in dataset.coords:
        metadata["step_labels"] = [
            int(value) for value in dataset.coords["step"].values
        ]

    return metadata


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
