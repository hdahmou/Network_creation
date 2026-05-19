"""Run one prepared policy simulation input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .agent_parameters import SampledAgentParameterPlan
from .agents import PolicyAgentBuilder
from .observers import (
    POLICY_WELFARE_OBSERVER_NAME,
    ObserverOutputSpec,
    build_observer_instance,
)
from .outputs import SAMPLED_AGENT_PARAMETERS_FIELD
from .sweep import SweepPlan


@dataclass(frozen=True)
class PolicySimulationRuntime:
    sweep: SweepPlan
    observer_specs: tuple[ObserverOutputSpec, ...]
    agent_builder: PolicyAgentBuilder
    sampled_agent_parameter_plan: SampledAgentParameterPlan
    max_n_agents: int

    def validate_sample_idx(self, sample_idx: int) -> None:
        if not 0 <= sample_idx < self.sweep.total_inputs:
            raise IndexError(f"sample_idx {sample_idx} is out of range")

    def run(self, sample_idx: int) -> dict[str, np.ndarray[Any, Any]]:
        from network_creation.core.runtime import create_simulation_runtime

        self.validate_sample_idx(sample_idx)
        configuration = self.sweep.configuration_at(sample_idx)
        simulation_settings = self.sweep.resolve(configuration)
        sampled_agent_parameters = self.sampled_agent_parameter_plan.sample(
            sample_idx,
            configuration,
            simulation_settings,
            self.max_n_agents,
            self.agent_builder.policy_entry_for_node,
        )
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
        observers = tuple(
            build_observer_instance(
                spec,
                configuration,
                column_index_by_name=self.sweep.column_index_by_name,
                policy_assignment=policy_assignment,
            )
            for spec in self.observer_specs
        )
        _, _, simulator = create_simulation_runtime(
            simulation_settings=simulation_settings,
            build_agent=lambda node_id: self.agent_builder.build_agent(
                node_id,
                configuration,
                simulation_settings,
                sampled_agent_parameters=sampled_agent_parameters,
            ),
            observers=observers,
            seed=simulation_settings.seed,
        )
        simulator.run(
            n_episodes=simulation_settings.n_episodes,
            reset_observers=True,
        )

        output = {
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
        if self.sampled_agent_parameter_plan.names:
            output[SAMPLED_AGENT_PARAMETERS_FIELD] = sampled_agent_parameters
        return output
