"""Per-agent sampled policy parameters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .heterogeneity import (
    Heterogeneity,
    parse_heterogeneity,
    sample_heterogeneous_value,
)
from .parameter_space import ParameterSpec
from .sweep import PolicyEntry

if TYPE_CHECKING:
    from network_creation.core.runtime import SimulationSettings


PolicyForNode = Callable[[int, NDArray[np.float32]], PolicyEntry]


@dataclass(frozen=True)
class SampledParameterPlanEntry:
    parameter_name: str
    policy_kwarg: str
    center_column: int
    sampled_column: int
    heterogeneity: Heterogeneity


@dataclass(frozen=True)
class SampledAgentParameterPlan:
    names: tuple[str, ...]
    parameters_by_policy_id: Mapping[str, tuple[SampledParameterPlanEntry, ...]]

    def sample(
        self,
        parameter_row_index: int,
        configuration: NDArray[np.float32],
        simulation_settings: SimulationSettings,
        max_n_agents: int,
        policy_for_node: PolicyForNode,
    ) -> NDArray[np.float32]:
        sampled: NDArray[np.float32] = np.full(
            (max_n_agents, len(self.names)),
            np.nan,
            dtype=np.float32,
        )
        if not self.names:
            return sampled

        for node_id in range(simulation_settings.n_agents):
            policy_id, _ = policy_for_node(node_id, configuration)
            for parameter in self.parameters_by_policy_id[policy_id]:
                sampled[node_id, parameter.sampled_column] = sample_heterogeneous_value(
                    float(configuration[parameter.center_column]),
                    parameter.heterogeneity,
                    seed=simulation_settings.seed,
                    n_agents=simulation_settings.n_agents,
                    parameter_row_index=parameter_row_index,
                    node_id=node_id,
                    parameter_name=parameter.parameter_name,
                )
        return sampled


def build_sampled_agent_parameter_plan(
    parameter_specs: Mapping[str, ParameterSpec],
    *,
    configuration_column_by_name: Mapping[str, int],
    policy_entries: Sequence[PolicyEntry],
) -> SampledAgentParameterPlan:
    parameters_by_policy_id: dict[str, list[SampledParameterPlanEntry]] = {
        policy_id: [] for policy_id, _ in policy_entries
    }
    names: list[str] = []
    for name, spec in parameter_specs.items():
        heterogeneity = parse_heterogeneity(name, spec)
        if heterogeneity is None:
            continue
        policy_id, parameter_name = _policy_parameter_name(name)
        sampled_parameter = SampledParameterPlanEntry(
            parameter_name=name,
            policy_kwarg=parameter_name,
            center_column=configuration_column_by_name[name],
            sampled_column=len(names),
            heterogeneity=heterogeneity,
        )

        if policy_id is None:
            for sampled_parameters in parameters_by_policy_id.values():
                sampled_parameters.append(sampled_parameter)
        elif policy_id in parameters_by_policy_id:
            parameters_by_policy_id[policy_id].append(sampled_parameter)
        else:
            continue
        names.append(name)

    return SampledAgentParameterPlan(
        names=tuple(names),
        parameters_by_policy_id={
            policy_id: tuple(parameters)
            for policy_id, parameters in parameters_by_policy_id.items()
        },
    )


def _policy_parameter_name(name: str) -> tuple[str | None, str]:
    if "." not in name:
        return None, name
    policy_id, parameter_name = name.split(".", maxsplit=1)
    return policy_id, parameter_name
