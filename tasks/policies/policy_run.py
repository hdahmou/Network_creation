"""Prepare and run policy simulations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from network_creation.core.agent import Agent
from ..execution.schema import OutputSpec, RunOutput
from .registry import POLICY_REGISTRY
from .observers import ObserverOutputSpec, build_observer_specs
from .sweep import (
    ParameterSamples,
    PolicyEntry,
    SweepPlan,
    build_sweep,
    collect_parameter_specs,
    infer_parameter_bounds,
    parse_policy_entries,
    sample_parameters,
)

if TYPE_CHECKING:
    from network_creation.core.runtime import SimulationSettings


@dataclass(frozen=True)
class PreparedPolicyRun:
    inputs: Sequence[int]
    output_schema: OutputSpec
    run: Callable[[int], RunOutput]
    result_attrs: Mapping[str, Any]
    result_arrays: Mapping[str, Any]
    runtime: PolicyRunContext

    @property
    def parameter_samples(self) -> ParameterSamples:
        return self.runtime.parameter_samples

    @property
    def sweep(self) -> SweepPlan:
        return self.runtime.sweep

    def build_agent(
        self,
        node_id: int,
        configuration: NDArray[np.float32],
        simulation_settings: SimulationSettings,
    ) -> Agent:
        return self.runtime.build_agent(node_id, configuration, simulation_settings)


@dataclass(frozen=True)
class PolicyRunContext:
    policy_entries: tuple[PolicyEntry, ...]
    parameter_samples: ParameterSamples
    sweep: SweepPlan
    observer_specs: tuple[ObserverOutputSpec, ...]

    def validate_sample_idx(self, sample_idx: int) -> None:
        if not 0 <= sample_idx < self.sweep.total_inputs:
            raise IndexError(f"sample_idx {sample_idx} is out of range")

    def build_agent(
        self,
        node_id: int,
        configuration: NDArray[np.float32],
        simulation_settings: SimulationSettings,
    ) -> Agent:
        policy_entry = self._policy_entry_for_node(node_id, configuration)
        policy_kind, policy_kwargs = self._policy_kwargs(
            policy_entry=policy_entry,
            configuration=configuration,
        )
        try:
            policy_factory = POLICY_REGISTRY[policy_kind]
        except KeyError as exc:
            raise ValueError(f"Unknown policy {policy_kind!r}") from exc
        return Agent(
            id=node_id,
            visibility_distance=int(policy_kwargs["visibility_distance"]),
            policy=policy_factory(
                seed=node_id,
                n_nodes=simulation_settings.n_agents,
                **policy_kwargs,
            ),
            seed=node_id,
        )

    def _policy_entry_for_node(
        self,
        node_id: int,
        configuration: NDArray[np.float32],
    ) -> PolicyEntry:
        if len(self.policy_entries) == 1:
            return self.policy_entries[0]

        first_count = int(
            configuration[
                self.sweep.column_index_by_name[
                    f"{self.policy_entries[0][0]}_count"
                ]
            ]
        )
        if node_id < first_count:
            return self.policy_entries[0]
        return self.policy_entries[1]

    def _policy_kwargs(
        self,
        *,
        policy_entry: PolicyEntry,
        configuration: NDArray[np.float32],
    ) -> tuple[str, dict[str, float]]:
        parameters: dict[str, float] = {}
        policy_id, policy_name = policy_entry
        policy_prefix = f"{policy_id}."
        for name in self.sweep.configuration_names:
            if "." in name and not name.startswith(policy_prefix):
                continue
            parameter_name = name.removeprefix(policy_prefix)
            parameters[parameter_name] = float(
                configuration[self.sweep.column_index_by_name[name]]
            )
        return policy_name, parameters

    def _build_observer(
        self,
        spec: ObserverOutputSpec,
        configuration: NDArray[np.float32],
    ) -> Any:
        from network_creation.observers.statistics.episode import (
            EpisodeStatisticObserver,
        )

        kwargs = dict(spec.kwargs)
        if issubclass(spec.observer_class, EpisodeStatisticObserver):
            kwargs["n_agents"] = int(
                configuration[self.sweep.column_index_by_name["n_agents"]]
            )
        if spec.name in {"centered_mean_welfare"}:
            for parameter_name in (
                "benefit_to_cost_ratio",
                "distance_discount",
            ):
                if parameter_name in kwargs:
                    continue
                column_index = self.sweep.column_index_by_name[
                    parameter_name
                ]
                kwargs[parameter_name] = float(configuration[column_index])

        return spec.observer_class(**kwargs)

    def run(self, sample_idx: int) -> dict[str, np.ndarray[Any, Any]]:
        from network_creation.core.runtime import create_simulation_runtime

        self.validate_sample_idx(sample_idx)
        configuration = self.sweep.configuration_at(sample_idx)
        simulation_settings = self.sweep.resolve(configuration)
        observers = tuple(
            self._build_observer(spec, configuration) for spec in self.observer_specs
        )
        _, _, simulator = create_simulation_runtime(
            simulation_settings=simulation_settings,
            build_agent=lambda node_id: self.build_agent(
                node_id,
                configuration,
                simulation_settings,
            ),
            observers=observers,
            seed=simulation_settings.seed,
        )
        simulator.run(
            n_episodes=simulation_settings.n_episodes,
            reset_observers=True,
        )

        return {
            spec.field.name: np.asarray(
                observer.get().outputs[spec.output],
                dtype=spec.observer_class.fixed_output_dtype(),
            )
            for spec, observer in zip(
                self.observer_specs,
                observers,
                strict=True,
            )
        }


def prepare_policy_run(config: Mapping[str, Any]) -> PreparedPolicyRun:
    policy_config = cast(Mapping[str, Any], config)
    policy_entries = parse_policy_entries(policy_config)
    input_spec = collect_parameter_specs(policy_config)
    parameter_specs = cast(Mapping[str, Mapping[str, object]], input_spec["parameters"])
    parameter_samples = sample_parameters(input_spec)
    sweep = build_sweep(
        policy_config,
        parameter_samples,
        parameter_specs=parameter_specs,
        policy_ids=tuple(policy_id for policy_id, _ in policy_entries),
    )
    observer_specs = build_observer_specs(
        raw_observers=policy_config["observers"],
        max_steps=sweep.max_steps,
        n_episodes=sweep.n_episodes,
    )
    runtime = PolicyRunContext(
        policy_entries=policy_entries,
        parameter_samples=parameter_samples,
        sweep=sweep,
        observer_specs=observer_specs,
    )
    result_attrs, result_arrays = _result_metadata(
        config=config,
        sweep=sweep,
        parameter_specs=parameter_specs,
    )

    return PreparedPolicyRun(
        inputs=tuple(range(sweep.total_inputs)),
        output_schema=OutputSpec(fields=tuple(spec.field for spec in observer_specs)),
        run=runtime.run,
        result_attrs=result_attrs,
        result_arrays=result_arrays,
        runtime=runtime,
    )


def _result_metadata(
    *,
    config: Mapping[str, Any],
    sweep: SweepPlan,
    parameter_specs: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    parameter_names = sweep.parameter_names_for_results
    parameters = sweep.parameter_values_for_results()
    parameter_low, parameter_high = infer_parameter_bounds(
        parameters,
        parameter_names=parameter_names,
        parameter_specs=parameter_specs,
    )
    input_coordinates = np.asarray(
        [
            sweep.coordinates(input_index)
            for input_index in range(sweep.total_inputs)
        ],
        dtype=np.int64,
    )

    return (
        {
            "parameter_names": list(parameter_names),
            "policy_config": {
                key: config[key]
                for key in ("sampling", "simulator", "game", "population", "observers")
                if key in config
            },
        },
        {
            "n_agents": np.asarray(
                sweep.n_agents_values,
                dtype=np.int64,
            ),
            "parameters": parameters,
            "parameter_low": parameter_low,
            "parameter_high": parameter_high,
            "input_coordinates": input_coordinates,
        },
    )
