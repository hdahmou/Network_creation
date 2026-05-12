"""Observer output specs for policy runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from network_creation.core.observability import ObservationSampling
from tasks.execution.schema import OutputArraySpec

AGENT_PAYOFF_OBSERVER_NAME = "agent_payoff"

if TYPE_CHECKING:
    from network_creation.observers.statistics.episode import EpisodeStatisticObserver


@dataclass(frozen=True)
class ObserverOutputSpec:
    name: str
    output: str
    observer_class: type[EpisodeStatisticObserver]
    kwargs: dict[str, Any]
    field: OutputArraySpec


def _load_observation_sampling(raw: Mapping[str, object]) -> ObservationSampling:
    kind = str(raw["kind"])
    if kind == "periodic":
        return ObservationSampling.periodic(int(cast(Any, raw["every"])))
    if kind == "terminal":
        return ObservationSampling.terminal()
    raise ValueError(f"unsupported observation sampling kind: {kind}")


def build_observer_specs(
    raw_observers: Mapping[str, Any],
    *,
    max_steps: int,
    n_episodes: int,
    n_agent_slots: int | None = None,
) -> tuple[ObserverOutputSpec, ...]:
    from network_creation.observers.grimoire import OBSERVER_GRIMOIRE
    from network_creation.observers.statistics.episode import EpisodeStatisticObserver

    defaults = cast(Mapping[str, Any], raw_observers.get("defaults", {}))
    default_settings = {
        "observation_sampling": defaults["sampling"],
        "aggregations": list(cast(Sequence[str], defaults["aggregations"])),
    }
    outputs = cast(Sequence[Mapping[str, Any]], raw_observers["outputs"])

    observer_specs: list[ObserverOutputSpec] = []
    for observer in outputs:
        name = str(observer["name"])
        output = str(observer["output"])
        observer_class = cast(type[EpisodeStatisticObserver], OBSERVER_GRIMOIRE[name])
        kwargs = dict(default_settings)
        if "sampling" in observer:
            kwargs["observation_sampling"] = observer["sampling"]
        if "aggregations" in observer:
            kwargs["aggregations"] = list(cast(Sequence[str], observer["aggregations"]))
        kwargs.update(cast(Mapping[str, Any], observer.get("settings", {})))
        if name == AGENT_PAYOFF_OBSERVER_NAME and "n_agent_slots" not in kwargs:
            if n_agent_slots is None:
                raise ValueError("agent_payoff observer requires n_agent_slots")
            kwargs["n_agent_slots"] = int(n_agent_slots)
        observation_sampling = _load_observation_sampling(
            cast(Mapping[str, object], kwargs["observation_sampling"])
        )
        kwargs["observation_sampling"] = observation_sampling
        if name == AGENT_PAYOFF_OBSERVER_NAME:
            base_shape = cast(Any, observer_class).fixed_output_shape_for_slots(
                max_steps=max_steps,
                observation_sampling=observation_sampling,
                n_agent_slots=int(cast(Any, kwargs["n_agent_slots"])),
            )
        else:
            base_shape = observer_class.fixed_output_shape(
                max_steps=max_steps,
                observation_sampling=observation_sampling,
            )
        field_shape = (n_episodes, *base_shape) if output == "runs" else base_shape

        observer_specs.append(
            ObserverOutputSpec(
                name=name,
                output=output,
                observer_class=observer_class,
                kwargs=kwargs,
                field=OutputArraySpec(
                    name=f"{name}.{output}",
                    shape=field_shape,
                    dtype=observer_class.fixed_output_dtype(),
                ),
            )
        )

    return tuple(observer_specs)


def build_observer_instance(
    spec: ObserverOutputSpec,
    configuration: Any,
    *,
    column_index_by_name: Mapping[str, int],
) -> Any:
    from network_creation.observers.statistics.episode import (
        EpisodeStatisticObserver,
    )

    kwargs = dict(spec.kwargs)
    if issubclass(spec.observer_class, EpisodeStatisticObserver):
        kwargs["n_agents"] = int(configuration[column_index_by_name["n_agents"]])
    if spec.name in {AGENT_PAYOFF_OBSERVER_NAME, "centered_mean_welfare"}:
        for parameter_name in (
            "benefit_to_cost_ratio",
            "distance_discount",
        ):
            if parameter_name in kwargs:
                continue
            kwargs[parameter_name] = float(
                configuration[column_index_by_name[parameter_name]]
            )

    return spec.observer_class(**kwargs)
