"""Observer-specific reference values for policy result plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm

from network_creation.data.choi_2020_references import (
    DEFAULT_REFERENCE_CACHE_PATH,
    DEFAULT_REFERENCE_PARAMETERS_PATH,
    ReferenceRequest,
    ChoiReferenceParameters,
    load_reference_cache,
    load_reference_parameters,
    lookup_reference_value,
    lookup_sigmoid_fit_reference_values,
    dimension_name_from_observable,
    observer_name_from_observable,
)
from network_creation.stats.topology_welfare import normalized_welfare_per_topology


BOUNDED_OBSERVER_NAMES = {
    "centered_mean_welfare",
    "central_point_of_dominance",
    "density",
    "global_closeness_centrality",
}
WELFARE_OBSERVER_NAME = "centered_mean_welfare"


def get_observer_reference_style(
    observable_name: str,
    values: np.ndarray,
    *,
    n_agents: int,
    theta_labels: Sequence[str],
    theta_values: np.ndarray,
    cache_path: str | Path | None = None,
    parameters_path: str | Path | None = None,
    extra_references: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], Normalize | None]:
    observer_name = observer_name_from_observable(observable_name)
    theta = _unique_theta_parameters(
        theta_labels,
        theta_values,
        ("benefit_to_cost_ratio", "distance_discount"),
    )
    references = (
        _welfare_references(n_agents, theta)
        if observer_name == WELFARE_OBSERVER_NAME and theta is not None
        else {}
    )

    request = _reference_request(
        observer_name,
        n_agents,
        theta,
        observable_name,
        parameters_path=parameters_path,
    )
    if request is not None:
        value = lookup_reference_value(load_reference_cache(_cache_path(cache_path)), request)
        if value is not None:
            references["Experimental"] = value

    references.update(
        {
            str(label): float(value)
            for label, value in (extra_references or {}).items()
            if np.isfinite(value)
        }
    )
    return references, _reference_norm(observer_name, values, references)


def get_sigmoid_fit_reference_values(
    observable_name: str,
    *,
    n_agents: int,
    cache_path: str | Path | None = None,
    parameters_path: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    observer_name = observer_name_from_observable(observable_name)
    request = _reference_request(
        observer_name,
        int(n_agents),
        None,
        observable_name,
        parameters_path=parameters_path,
    )
    if request is None:
        return {}

    references = lookup_sigmoid_fit_reference_values(
        load_reference_cache(_cache_path(cache_path)),
        request,
    )
    return {
        key: {"Experimental fit": value}
        for key, value in references.items()
        if np.isfinite(value)
    }


def positive_reference_norm(
    values: np.ndarray,
    references: Mapping[str, float],
) -> Normalize:
    finite = _finite_values(values, references)
    if finite.size == 0:
        return Normalize(vmin=0.0)
    vmax = max(0.0, float(np.max(finite)))
    return Normalize(vmin=0.0, vmax=vmax)


def annotate_colorbar_reference_values(
    colorbar_axis: Axes,
    label_axis: Axes,
    references: Mapping[str, float],
    *,
    value_format: str = "{:.2f}",
    fontsize: float | str = "small",
) -> None:
    colorbar_ylim = colorbar_axis.get_ylim()
    label_axis.clear()
    label_axis.set_ylim(colorbar_ylim)
    label_axis.set_xlim(0.0, 1.0)
    label_axis.axis("off")

    items = sorted(
        (
            (str(label), float(value))
            for label, value in references.items()
            if np.isfinite(value)
        ),
        key=lambda item: item[1],
    )
    if not items:
        return

    for label, value in items:
        value_text = colorbar_axis.text(
            -0.35,
            value,
            value_format.format(value),
            transform=colorbar_axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=fontsize,
            clip_on=False,
        )
        label_text = colorbar_axis.text(
            1.75,
            value,
            label,
            transform=colorbar_axis.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=fontsize,
            clip_on=False,
        )
        value_text.set_in_layout(False)
        label_text.set_in_layout(False)
    colorbar_axis.set_ylim(colorbar_ylim)


def _reference_request(
    observer_name: str,
    n_agents: int,
    theta: Mapping[str, float] | None,
    observable_name: str = "",
    *,
    parameters_path: str | Path | None = None,
) -> ReferenceRequest | None:
    if observer_name != WELFARE_OBSERVER_NAME:
        dimension_name = dimension_name_from_observable(observable_name)
        return ReferenceRequest(observer_name, int(n_agents), dimension_name=dimension_name)
    choi_parameters = _choi_parameters_for_group_size(n_agents, parameters_path)
    if choi_parameters is not None:
        return ReferenceRequest(
            observer_name,
            int(n_agents),
            choi_parameters.benefit_to_cost_ratio,
            choi_parameters.distance_discount,
        )
    if theta is None:
        return None
    return ReferenceRequest(
        observer_name,
        int(n_agents),
        theta["benefit_to_cost_ratio"],
        theta["distance_discount"],
    )


def _reference_norm(
    observer_name: str,
    values: np.ndarray,
    references: Mapping[str, float],
) -> Normalize | None:
    if observer_name == WELFARE_OBSERVER_NAME and {"Empty", "Random", "Star"} <= set(
        references
    ):
        empty, random, star = (references[name] for name in ("Empty", "Random", "Star"))
        if empty < random < star:
            return TwoSlopeNorm(vmin=empty, vcenter=random, vmax=star)
    if observer_name in BOUNDED_OBSERVER_NAMES:
        return Normalize(vmin=0.0, vmax=1.0)

    finite = _finite_values(values, references)
    if finite.size == 0:
        return None
    return Normalize(vmin=float(np.min(finite)), vmax=float(np.max(finite)))


def _welfare_references(
    n_agents: int,
    theta: Mapping[str, float],
) -> dict[str, float]:
    def welfare(topology_class: str) -> float:
        return normalized_welfare_per_topology(
            topology_class=topology_class,  # type: ignore[arg-type]
            n=int(n_agents),
            benefit_to_cost_ratio=theta["benefit_to_cost_ratio"],
            distance_discount=theta["distance_discount"],
        )

    reference_max = welfare("max")
    if not np.isfinite(reference_max) or reference_max == 0.0:
        return {}
    return {
        "Empty": float(welfare("empty") / reference_max),
        "Random": float(welfare("random") / reference_max),
        "Star": float(welfare("star") / reference_max),
    }


def _cache_path(path: str | Path | None) -> Path:
    return _project_path(DEFAULT_REFERENCE_CACHE_PATH if path is None else path)


def _parameters_path(path: str | Path | None) -> Path:
    return _project_path(DEFAULT_REFERENCE_PARAMETERS_PATH if path is None else path)


def _project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "network_creation").exists():
            return candidate / path
    return current / path


def _finite_values(
    values: np.ndarray,
    references: Mapping[str, float],
) -> np.ndarray:
    finite_values = np.asarray(values, dtype=float)
    return np.concatenate(
        [
            finite_values[np.isfinite(finite_values)],
            np.asarray([v for v in references.values() if np.isfinite(v)], dtype=float),
        ]
    )


def _choi_parameters_for_group_size(
    n_agents: int,
    path: str | Path | None,
) -> ChoiReferenceParameters | None:
    try:
        parameters = load_reference_parameters(_parameters_path(path))
    except FileNotFoundError:
        return None
    return {int(parameter.group_size): parameter for parameter in parameters}.get(
        int(n_agents)
    )


def _unique_theta_parameters(
    theta_labels: Sequence[str],
    theta_values: np.ndarray,
    names: Sequence[str],
) -> dict[str, float] | None:
    labels = [str(label) for label in theta_labels]
    if any(name not in labels for name in names):
        return None

    values = np.asarray(theta_values, dtype=float)
    values = values[None, :] if values.ndim == 1 else values.reshape(-1, values.shape[-1])
    parameters = {}
    for name in names:
        column = values[:, labels.index(name)]
        unique = np.unique(np.round(column[np.isfinite(column)], 12))
        if unique.size != 1:
            return None
        parameters[name] = float(unique[0])
    return parameters
