"""Run one prepared policy simulation input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..execution.schema import OutputArraySpec
from .agent_parameters import SampledAgentParameterPlan
from .agent_attributes import AgentAttributePlan
from .agents import PolicyAgentBuilder
from .observers import (
    POLICY_WELFARE_OBSERVER_NAME,
    ObserverOutputSpec,
    build_observer_instance,
)
from .outputs import SAMPLED_AGENT_PARAMETERS_FIELD
from .sweep import SweepPlan

if TYPE_CHECKING:
    from network_creation.core.observability import SimulationObserver
    from network_creation.core.runtime import SimulationSettings
    from network_creation.core.simulator import Simulator


def _pad_observer_time_axis(
    values: np.ndarray[Any, Any],
    field: OutputArraySpec,
    *,
    aggregation: str,
    is_terminal: bool,
) -> np.ndarray[Any, Any]:
    expected_shape = field.shape
    actual_shape = tuple(values.shape)
    if is_terminal:
        if actual_shape != expected_shape:
            raise ValueError(
                f"terminal field {field.name!r} exported shape {actual_shape} "
                f"but expected {expected_shape}"
            )
        return values

    time_axis = 1 if aggregation == "runs" else 0
    if aggregation not in {"mean", "std", "runs"}:
        raise ValueError(f"unsupported observer aggregation {aggregation!r}")
    if values.ndim != len(expected_shape) or any(
        actual != expected
        for axis, (actual, expected) in enumerate(
            zip(actual_shape, expected_shape, strict=True)
        )
        if axis != time_axis
    ):
        raise ValueError(
            f"field {field.name!r} exported shape {actual_shape} with non-time "
            f"dimensions incompatible with {expected_shape}"
        )
    if actual_shape[time_axis] > expected_shape[time_axis]:
        raise ValueError(
            f"field {field.name!r} exported shape {actual_shape} longer than "
            f"expected {expected_shape}"
        )
    if actual_shape == expected_shape:
        return values

    padded = np.full(expected_shape, np.nan, dtype=field.resolved_dtype)
    valid_prefix = [slice(None)] * values.ndim
    valid_prefix[time_axis] = slice(0, actual_shape[time_axis])
    padded[tuple(valid_prefix)] = values
    return padded


@dataclass(frozen=True)
class PreparedPolicySimulation:
    """One resolved policy input ready to run with chosen observers."""

    simulator: Simulator
    settings: SimulationSettings
    configuration: NDArray[np.float32]
    sampled_agent_parameters: NDArray[np.float32]


@dataclass(frozen=True)
class PolicySimulationRuntime:
    sweep: SweepPlan
    observer_specs: tuple[ObserverOutputSpec, ...]
    agent_builder: PolicyAgentBuilder
    sampled_agent_parameter_plan: SampledAgentParameterPlan
    agent_attribute_plan: AgentAttributePlan
    max_n_agents: int

    def validate_sample_idx(self, sample_idx: int) -> None:
        if not 0 <= sample_idx < self.sweep.total_inputs:
            raise IndexError(f"sample_idx {sample_idx} is out of range")

    def prepare_simulation(
        self,
        sample_idx: int,
        *,
        observers: tuple[SimulationObserver, ...] | None = None,
    ) -> PreparedPolicySimulation:
        """Resolve one sweep input and construct its simulation runtime."""

        from network_creation.core.runtime import create_simulation_runtime

        self.validate_sample_idx(sample_idx)
        configuration = self.sweep.configuration_at(sample_idx)
        simulation_settings = self.sweep.resolve(configuration)
        _, _, parameter_row_index = self.sweep.coordinates(sample_idx)
        sampled_agent_parameters = self.sampled_agent_parameter_plan.sample(
            parameter_row_index,
            configuration,
            simulation_settings,
            self.max_n_agents,
            self.agent_builder.policy_entry_for_node,
        )
        resolved_agent_attributes = self.agent_attribute_plan.resolve(
            configuration,
            simulation_settings,
            parameter_row_index=parameter_row_index,
            max_n_agents=self.max_n_agents,
            policy_for_node=self.agent_builder.policy_entry_for_node,
        )
        if resolved_agent_attributes.sampled_values.shape[1]:
            sampled_agent_parameters = np.concatenate(
                (
                    sampled_agent_parameters,
                    resolved_agent_attributes.sampled_values,
                ),
                axis=1,
            )
        if observers is None:
            policy_assignment = (
                self.agent_builder.policy_assignment_for_configuration(
                    configuration,
                    n_agents=simulation_settings.n_agents,
                )
                if any(
                    spec.name == POLICY_WELFARE_OBSERVER_NAME
                    for spec in self.observer_specs
                )
                else None
            )
            resolved_observers = tuple(
                build_observer_instance(
                    spec,
                    configuration,
                    column_index_by_name=self.sweep.column_index_by_name,
                    policy_assignment=policy_assignment,
                )
                for spec in self.observer_specs
            )
        else:
            resolved_observers = observers
        _, _, simulator = create_simulation_runtime(
            simulation_settings=simulation_settings,
            build_agent=lambda node_id: self.agent_builder.build_agent(
                node_id,
                configuration,
                simulation_settings,
                sampled_agent_parameters=sampled_agent_parameters,
                activation_weight=float(
                    resolved_agent_attributes.activation_weights[node_id]
                ),
            ),
            observers=resolved_observers,
            seed=simulation_settings.seed,
        )

        return PreparedPolicySimulation(
            simulator=simulator,
            settings=simulation_settings,
            configuration=configuration.copy(),
            sampled_agent_parameters=sampled_agent_parameters.copy(),
        )

    def run(self, sample_idx: int) -> dict[str, np.ndarray[Any, Any]]:
        prepared = self.prepare_simulation(sample_idx)
        prepared.simulator.run(
            n_episodes=prepared.settings.n_episodes,
            reset_observers=True,
        )

        output = {}
        for spec, observer in zip(
            self.observer_specs,
            prepared.simulator.observers,
            strict=True,
        ):
            result = observer.get()
            for field in spec.fields:
                aggregation = field.name.split(".", maxsplit=1)[1]
                values = np.asarray(
                    result.outputs[aggregation],
                    dtype=field.resolved_dtype,
                )
                output[field.name] = _pad_observer_time_axis(
                    values,
                    field,
                    aggregation=aggregation,
                    is_terminal=(
                        spec.kwargs["observation_sampling"].kind == "terminal"
                    ),
                )
        if prepared.sampled_agent_parameters.shape[1]:
            output[SAMPLED_AGENT_PARAMETERS_FIELD] = (
                prepared.sampled_agent_parameters
            )
        return output
