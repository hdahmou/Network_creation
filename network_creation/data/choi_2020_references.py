"""Choi 2020 reference statistics for result plots."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from inspect import Parameter, signature
import json
from pathlib import Path
import pickle
from typing import Any, Callable, cast

import numpy as np

from network_creation.core.observability import ObservationSampling
from network_creation.observers.grimoire import OBSERVER_GRIMOIRE
from network_creation.observers.statistics.episode import EpisodeStatisticObserver
from network_creation.stats.sigmoid_fit import fit_zero_start_sigmoids

from .choi_2020 import (
    list_choi_2020_configs,
    load_choi_2020_normalized_data,
    run_choi_2020_replay_batch,
)


DEFAULT_REFERENCE_CACHE_PATH = Path("artifacts/data/choi_2020_reference_stats.pkl")
DEFAULT_REFERENCE_PARAMETERS_PATH = Path("data/choi_2020/parameters.json")
WELFARE_OBSERVER_NAME = "centered_mean_welfare"
SIGMOID_FIT_OBSERVER_NAMES = frozenset(
    {
        "central_point_of_dominance",
        WELFARE_OBSERVER_NAME,
    }
)
SIGMOID_FIT_PARAMETER_NAMES = ("A", "k", "t0")
BASE_OBSERVER_PARAMETERS = {
    "self",
    "n_agents",
    "observation_sampling",
    "aggregations",
}


@dataclass(frozen=True)
class ChoiReferenceParameters:
    group_size: int
    benefit_to_cost_ratio: float
    distance_discount: float


@dataclass(frozen=True)
class ReferenceRequest:
    observer_name: str
    n_agents: int
    benefit_to_cost_ratio: float | None = None
    distance_discount: float | None = None
    dimension_name: str | None = None


def observer_name_from_observable(observable_name: str) -> str:
    return str(observable_name).split(".", maxsplit=1)[0]


def dimension_name_from_observable(observable_name: str) -> str | None:
    parts = str(observable_name).split(".")
    return parts[2] if len(parts) > 2 else None


def load_reference_parameters(path: str | Path) -> tuple[ChoiReferenceParameters, ...]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(
        ChoiReferenceParameters(
            group_size=int(row["group_size"]),
            benefit_to_cost_ratio=float(row["benefit_to_cost_ratio"]),
            distance_discount=float(row["distance_discount"]),
        )
        for row in rows
    )


def available_reference_observer_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, observer_class in OBSERVER_GRIMOIRE.items()
            if _is_cacheable_reference_observer(name, observer_class)
        )
    )


def reference_requests(
    parameters: Sequence[ChoiReferenceParameters],
) -> tuple[ReferenceRequest, ...]:
    requests: list[ReferenceRequest] = []
    parameters_by_size = {
        int(parameter.group_size): parameter for parameter in parameters
    }
    for n_agents, parameter in sorted(parameters_by_size.items()):
        for observer_name in available_reference_observer_names():
            dimension_names = getattr(
                OBSERVER_GRIMOIRE[observer_name],
                "dimension_names",
                None,
            )
            if dimension_names is None:
                requests.append(
                    _reference_request_for_observer(observer_name, n_agents, parameter)
                )
            else:
                requests.extend(
                    _reference_request_for_observer(
                        observer_name,
                        n_agents,
                        parameter,
                        dimension_name=str(dimension_name),
                    )
                    for dimension_name in dimension_names
                )
    return tuple(requests)


def build_choi_observers(
    *,
    n_agents: int,
    observer_names: Sequence[str],
    observation_sampling: ObservationSampling,
    aggregations: Sequence[str] | Mapping[str, Sequence[str]],
    benefit_to_cost_ratio: float | None = None,
    distance_discount: float | None = None,
    settings_by_observer: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, EpisodeStatisticObserver]:
    settings_by_observer = settings_by_observer or {}
    return {
        name: _build_observer(
            name,
            n_agents=n_agents,
            observation_sampling=observation_sampling,
            aggregations=_aggregations(name, aggregations),
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
            settings=settings_by_observer.get(name, {}),
        )
        for name in observer_names
    }


def build_reference_cache(
    *,
    data_root: str | Path,
    requests: Iterable[ReferenceRequest],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    requests = tuple(sorted(set(requests), key=reference_key))
    reference_observer_names = set(available_reference_observer_names())
    root = _choi_raw_root(data_root)
    if progress is not None:
        progress(f"reading Choi replay configs from {root}")

    configs_by_size = defaultdict(list)
    for config in list_choi_2020_configs(root):
        configs_by_size[config.group_size].append(config)
    if progress is not None:
        progress(f"found {sum(len(configs) for configs in configs_by_size.values())} configs")

    values: dict[tuple[str, int, str, str, str], float] = {}
    sigmoid_values: dict[tuple[str, int, str, str, str, str], float] = {}
    missing_group_sizes = []
    for n_agents in sorted({request.n_agents for request in requests}):
        configs = configs_by_size.get(n_agents, [])
        if not configs:
            missing_group_sizes.append(n_agents)
            if progress is not None:
                progress(f"skipping n_agents={n_agents}: no Choi configs")
            continue

        group_requests = [
            request
            for request in requests
            if request.n_agents == n_agents
            and request.observer_name in reference_observer_names
        ]
        if not group_requests:
            continue
        if progress is not None:
            progress(
                f"replaying n_agents={n_agents}: "
                f"{len(configs)} configs, {len(group_requests)} reference values"
            )
        request_groups = defaultdict(list)
        for request in group_requests:
            request_groups[_observer_run_key(request)].append(request)
        observer_keys = tuple(request_groups)
        terminal_observers = [
            _build_observer(
                key[0],
                n_agents=key[1],
                observation_sampling=ObservationSampling.terminal(),
                aggregations=("mean",),
                benefit_to_cost_ratio=key[2],
                distance_discount=key[3],
            )
            for key in observer_keys
        ]
        sigmoid_observer_keys = tuple(
            key for key in observer_keys if key[0] in SIGMOID_FIT_OBSERVER_NAMES
        )
        sigmoid_observers = [
            _build_observer(
                key[0],
                n_agents=key[1],
                observation_sampling=ObservationSampling.periodic(1),
                aggregations=("mean",),
                benefit_to_cost_ratio=key[2],
                distance_discount=key[3],
            )
            for key in sigmoid_observer_keys
        ]
        max_steps = max(
            load_choi_2020_normalized_data(config).max_step for config in configs
        )
        run_choi_2020_replay_batch(
            configs=tuple(sorted(configs)),
            observers=tuple(terminal_observers + sigmoid_observers),
            max_steps=max_steps,
        )
        for requests_for_observer, observer in zip(
            request_groups.values(),
            terminal_observers,
            strict=True,
        ):
            _store_terminal_reference_values(
                values,
                requests=requests_for_observer,
                observer=observer,
            )
        for observer_key, observer in zip(
            sigmoid_observer_keys,
            sigmoid_observers,
            strict=True,
        ):
            _store_sigmoid_reference_values(
                sigmoid_values,
                requests=request_groups[observer_key],
                observer=observer,
            )
        if progress is not None:
            progress(f"finished n_agents={n_agents}")

    return {
        "values": values,
        "sigmoid_values": sigmoid_values,
        "missing_group_sizes": missing_group_sizes,
    }


def save_reference_cache(
    cache: Mapping[str, Any],
    path: str | Path = DEFAULT_REFERENCE_CACHE_PATH,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(dict(cache), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_reference_cache(
    path: str | Path = DEFAULT_REFERENCE_CACHE_PATH,
) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        cache = pickle.load(handle)
    if not isinstance(cache, dict):
        raise ValueError(f"invalid Choi reference cache at {path}")
    return cache


def lookup_reference_value(
    cache: Mapping[str, Any] | None,
    request: ReferenceRequest,
) -> float | None:
    if cache is None:
        return None
    value = cast(
        Mapping[tuple[str, int, str, str, str], float],
        cache.get("values", {}),
    ).get(reference_key(request))
    return None if value is None else float(value)


def sigmoid_reference_key(
    request: ReferenceRequest,
    parameter_name: str,
) -> tuple[str, int, str, str, str, str]:
    return (*reference_key(request), str(parameter_name))


def lookup_sigmoid_fit_reference_values(
    cache: Mapping[str, Any] | None,
    request: ReferenceRequest,
) -> dict[str, float]:
    """Return cached sigmoid references in their fitted time units."""

    if cache is None or request.observer_name not in SIGMOID_FIT_OBSERVER_NAMES:
        return {}
    raw_values = cast(
        Mapping[tuple[str, int, str, str, str, str], float],
        cache.get("sigmoid_values", {}),
    )
    references: dict[str, float] = {}
    for parameter_name in SIGMOID_FIT_PARAMETER_NAMES:
        value = raw_values.get(sigmoid_reference_key(request, parameter_name))
        if value is not None and np.isfinite(value):
            references[parameter_name] = float(value)
    return references


def reference_key(request: ReferenceRequest) -> tuple[str, int, str, str, str]:
    return (
        request.observer_name,
        int(request.n_agents),
        "" if request.dimension_name is None else request.dimension_name,
        _float_key(request.benefit_to_cost_ratio),
        _float_key(request.distance_discount),
    )


def _reference_request_for_observer(
    observer_name: str,
    n_agents: int,
    parameter: ChoiReferenceParameters,
    *,
    dimension_name: str | None = None,
) -> ReferenceRequest:
    if observer_name != WELFARE_OBSERVER_NAME:
        return ReferenceRequest(
            observer_name,
            int(n_agents),
            dimension_name=dimension_name,
        )
    return ReferenceRequest(
        observer_name,
        int(n_agents),
        benefit_to_cost_ratio=parameter.benefit_to_cost_ratio,
        distance_discount=parameter.distance_discount,
        dimension_name=dimension_name,
    )


def _observer_run_key(
    request: ReferenceRequest,
) -> tuple[str, int, float | None, float | None]:
    return (
        request.observer_name,
        int(request.n_agents),
        request.benefit_to_cost_ratio,
        request.distance_discount,
    )


def _store_terminal_reference_values(
    values: dict[tuple[str, int, str, str, str], float],
    *,
    requests: Sequence[ReferenceRequest],
    observer: EpisodeStatisticObserver,
) -> None:
    result = observer.get()
    mean = np.asarray(result.outputs["mean"], dtype=np.float32)
    if not mean.size:
        return
    if result.dimension_names is None:
        values[reference_key(requests[0])] = float(mean.ravel()[-1])
        return

    terminal_values = mean.reshape(-1, len(result.dimension_names))[-1]
    for request in requests:
        if request.dimension_name not in result.dimension_names:
            continue
        dimension_index = result.dimension_names.index(request.dimension_name)
        values[reference_key(request)] = float(terminal_values[dimension_index])


def _store_sigmoid_reference_values(
    values: dict[tuple[str, int, str, str, str, str], float],
    *,
    requests: Sequence[ReferenceRequest],
    observer: EpisodeStatisticObserver,
) -> None:
    result = observer.get()
    if result.steps is None:
        return

    mean = np.asarray(result.outputs["mean"], dtype=np.float32)
    if not mean.size:
        return

    t = np.asarray(result.steps, dtype=np.float32)
    fit_requests: list[ReferenceRequest] = []
    fit_series: list[np.ndarray] = []
    for request in requests:
        series = _select_observable_series(
            mean,
            dimension_names=result.dimension_names,
            dimension_name=request.dimension_name,
        )
        if series is None or series.shape != t.shape:
            continue
        fit_requests.append(request)
        fit_series.append(series)

    if not fit_series:
        return

    params, _ = fit_zero_start_sigmoids(np.stack(fit_series), t=t)
    params = np.asarray(params, dtype=np.float32)
    for request, row in zip(fit_requests, params, strict=True):
        for parameter_name, value in zip(SIGMOID_FIT_PARAMETER_NAMES, row, strict=True):
            if np.isfinite(value):
                values[sigmoid_reference_key(request, parameter_name)] = float(value)


def _build_observer(
    observer_name: str,
    *,
    n_agents: int,
    observation_sampling: ObservationSampling,
    aggregations: Sequence[str],
    benefit_to_cost_ratio: float | None = None,
    distance_discount: float | None = None,
    settings: Mapping[str, Any] | None = None,
) -> EpisodeStatisticObserver:
    observer_class = OBSERVER_GRIMOIRE[observer_name]
    if not issubclass(observer_class, EpisodeStatisticObserver):
        raise ValueError(f"observer {observer_name!r} is not a statistic observer")

    kwargs: dict[str, Any] = {
        "n_agents": int(n_agents),
        "observation_sampling": observation_sampling,
        "aggregations": tuple(aggregations),
        **dict(settings or {}),
    }
    if observer_name == WELFARE_OBSERVER_NAME:
        if benefit_to_cost_ratio is None or distance_discount is None:
            raise ValueError("welfare references require Choi payoff parameters")
        kwargs["benefit_to_cost_ratio"] = float(cast(float, benefit_to_cost_ratio))
        kwargs["distance_discount"] = float(cast(float, distance_discount))
    return cast(type[EpisodeStatisticObserver], observer_class)(**kwargs)


def _is_cacheable_reference_observer(
    observer_name: str,
    observer_class: type[Any],
) -> bool:
    if not issubclass(observer_class, EpisodeStatisticObserver):
        return False
    if observer_name == WELFARE_OBSERVER_NAME:
        return True
    for parameter in signature(observer_class.__init__).parameters.values():
        if parameter.name in BASE_OBSERVER_PARAMETERS:
            continue
        if parameter.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue
        if parameter.default is Parameter.empty:
            return False
    return True


def _aggregations(
    observer_name: str,
    aggregations: Sequence[str] | Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    if isinstance(aggregations, Mapping):
        return tuple(aggregations.get(observer_name, aggregations.get("*", ())))
    return tuple(aggregations)


def _float_key(value: float | None) -> str:
    return "" if value is None else f"{float(value):.12g}"


def _select_observable_series(
    values: np.ndarray,
    *,
    dimension_names: tuple[str, ...] | None,
    dimension_name: str | None,
) -> np.ndarray | None:
    if dimension_names is None:
        if dimension_name is not None:
            return None
        return np.asarray(values, dtype=np.float32).reshape(-1)

    if dimension_name not in dimension_names:
        return None
    dimension_index = dimension_names.index(dimension_name)
    return np.asarray(values, dtype=np.float32).reshape(-1, len(dimension_names))[
        :, dimension_index
    ]


def _choi_raw_root(data_root: str | Path) -> Path:
    root = _resolve_path(data_root)
    return root if list(root.glob("Linking_*_*/data.csv")) else root / "raw"


def _resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "network_creation").exists():
            return (candidate / path).resolve()
    return path.resolve()
