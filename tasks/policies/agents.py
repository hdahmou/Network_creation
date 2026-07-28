"""Agent construction for policy runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from network_creation.core.agent import Agent

from .agent_parameters import SampledParameterPlanEntry
from .registry import POLICY_REGISTRY
from .sweep import PolicyEntry

if TYPE_CHECKING:
    from network_creation.core.runtime import SimulationSettings


@dataclass(frozen=True)
class PolicyAssignment:
    policy_ids: tuple[str, ...]
    policy_slot_by_agent: NDArray[np.int32]


@dataclass(frozen=True)
class PolicyAgentBuilder:
    policy_entries: tuple[PolicyEntry, ...]
    configuration_names: tuple[str, ...]
    column_index_by_name: Mapping[str, int]
    sampled_parameters_by_policy_id: Mapping[
        str,
        tuple[SampledParameterPlanEntry, ...],
    ]

    def policy_entry_for_node(
        self,
        node_id: int,
        configuration: NDArray[np.float32],
    ) -> PolicyEntry:
        if len(self.policy_entries) == 1:
            return self.policy_entries[0]

        first_policy_id, _ = self.policy_entries[0]
        first_count = int(
            configuration[self.column_index_by_name[f"{first_policy_id}_count"]]
        )
        if node_id < first_count:
            return self.policy_entries[0]
        return self.policy_entries[1]

    def policy_assignment_for_configuration(
        self,
        configuration: NDArray[np.float32],
        *,
        n_agents: int,
    ) -> PolicyAssignment:
        policy_ids = tuple(policy_id for policy_id, _ in self.policy_entries)
        policy_slot_by_id = {
            policy_id: slot for slot, policy_id in enumerate(policy_ids)
        }
        policy_slot_by_agent: NDArray[np.int32] = np.empty(
            int(n_agents),
            dtype=np.int32,
        )
        for node_id in range(int(n_agents)):
            policy_id, _ = self.policy_entry_for_node(node_id, configuration)
            policy_slot_by_agent[node_id] = policy_slot_by_id[policy_id]
        return PolicyAssignment(
            policy_ids=policy_ids,
            policy_slot_by_agent=policy_slot_by_agent,
        )

    def build_agent(
        self,
        node_id: int,
        configuration: NDArray[np.float32],
        simulation_settings: SimulationSettings,
        *,
        sampled_agent_parameters: NDArray[np.float32],
        activation_weight: float,
    ) -> Agent:
        policy_entry = self.policy_entry_for_node(node_id, configuration)
        policy_kind, policy_kwargs = self._policy_kwargs(
            policy_entry=policy_entry,
            configuration=configuration,
            node_id=node_id,
            sampled_agent_parameters=sampled_agent_parameters,
        )
        try:
            policy_factory = POLICY_REGISTRY[policy_kind]
        except KeyError as exc:
            raise ValueError(f"Unknown policy {policy_kind!r}") from exc
        visibility_distance = float(policy_kwargs["visibility_distance"])
        if not visibility_distance.is_integer():
            raise ValueError("resolved visibility_distance must be an integer")
        return Agent(
            id=node_id,
            visibility_distance=int(visibility_distance),
            policy=policy_factory(
                seed=node_id,
                n_nodes=simulation_settings.n_agents,
                **policy_kwargs,
            ),
            seed=node_id,
            activation_weight=activation_weight,
        )

    def _policy_kwargs(
        self,
        *,
        policy_entry: PolicyEntry,
        configuration: NDArray[np.float32],
        node_id: int,
        sampled_agent_parameters: NDArray[np.float32],
    ) -> tuple[str, dict[str, Any]]:
        parameters: dict[str, Any] = {}
        policy_id, policy_name = policy_entry
        policy_prefix = f"{policy_id}."
        for name in self.configuration_names:
            if "." in name and not name.startswith(policy_prefix):
                continue
            parameter_name = name.removeprefix(policy_prefix)
            if parameter_name.startswith("agent."):
                continue
            parameters[parameter_name] = configuration[self.column_index_by_name[name]]

        for parameter in self.sampled_parameters_by_policy_id[policy_id]:
            parameters[parameter.policy_kwarg] = sampled_agent_parameters[
                node_id,
                parameter.sampled_column,
            ]
        return policy_name, parameters
