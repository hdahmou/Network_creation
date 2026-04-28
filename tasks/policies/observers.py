"""Observer output specs for policy runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from network_creation.core.observability import ObservationSampling
from tasks.execution.schema import OutputArraySpec

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
        observation_sampling = _load_observation_sampling(
            cast(Mapping[str, object], kwargs["observation_sampling"])
        )
        kwargs["observation_sampling"] = observation_sampling
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
