"""Per-agent sampled policy parameters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .parameter_space import ParameterSpec, validate_continuous_parameter_spec
from .sweep import PolicyEntry

if TYPE_CHECKING:
    from network_creation.core.runtime import SimulationSettings


PolicyForNode = Callable[[int, NDArray[np.float32]], PolicyEntry]


@dataclass(frozen=True)
class SampledParameterPlanEntry:
    policy_kwarg: str
    mean_column: int
    sampled_column: int
    minimum: float
    maximum: float
    std: float


@dataclass(frozen=True)
class SampledAgentParameterPlan:
    names: tuple[str, ...]
    parameters_by_policy_id: Mapping[str, tuple[SampledParameterPlanEntry, ...]]

    def sample(
        self,
        sample_idx: int,
        configuration: NDArray[np.float32],
        simulation_settings: SimulationSettings,
        max_n_agents: int,
        policy_for_node: PolicyForNode,
    ) -> NDArray[np.float32]:
        sampled = np.full(
            (max_n_agents, len(self.names)),
            np.nan,
            dtype=np.float32,
        )
        if not self.names:
            return sampled

        rng = np.random.default_rng(
            np.random.SeedSequence(
                (
                    int(simulation_settings.seed),
                    int(sample_idx),
                    int(simulation_settings.n_agents),
                )
            )
        )
        for node_id in range(simulation_settings.n_agents):
            policy_id, _ = policy_for_node(node_id, configuration)
            for parameter in self.parameters_by_policy_id[policy_id]:
                sampled[node_id, parameter.sampled_column] = np.clip(
                    rng.normal(
                        loc=configuration[parameter.mean_column],
                        scale=parameter.std,
                    ),
                    parameter.minimum,
                    parameter.maximum,
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
        if str(spec["kind"]) != "continuous":
            continue
        minimum, maximum, std = validate_continuous_parameter_spec(name, spec)
        if std == 0.0:
            continue
        policy_id, parameter_name = _policy_parameter_name(name)
        sampled_parameter = SampledParameterPlanEntry(
            policy_kwarg=parameter_name,
            mean_column=configuration_column_by_name[name],
            sampled_column=len(names),
            minimum=minimum,
            maximum=maximum,
            std=std,
        )
        names.append(name)

        if policy_id is None:
            for sampled_parameters in parameters_by_policy_id.values():
                sampled_parameters.append(sampled_parameter)
        elif policy_id in parameters_by_policy_id:
            parameters_by_policy_id[policy_id].append(sampled_parameter)

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
