"""Configuration and resolution of universal per-agent attributes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from network_creation.core.runtime import SimulationSettings

from .parameter_space import ParameterSpec, validate_continuous_parameter_spec


type PolicyEntry = tuple[str, str]
type PolicyForNode = Callable[[int, NDArray[np.float32]], PolicyEntry]
type AttributeTransform = Literal["identity", "scaled_other_agents"]

ACTIVATION_WEIGHT = "activation_weight"


@dataclass(frozen=True)
class AgentAttributeDefinition:
    """One configured source parameter and its runtime transformation."""

    policy_id: str | None
    attribute_name: str
    parameter_name: str
    output_name: str
    transform: AttributeTransform
    parameter_spec: ParameterSpec


@dataclass(frozen=True)
class AgentAttributePlanEntry:
    definition: AgentAttributeDefinition
    source_column: int
    sampled_column: int | None
    minimum: float | None
    maximum: float | None
    std: float


@dataclass(frozen=True)
class ResolvedAgentAttributes:
    activation_weights: NDArray[np.float32]
    sampled_values: NDArray[np.float32]


@dataclass(frozen=True)
class AgentAttributePlan:
    """Resolve configured agent attributes for one simulation input."""

    entry_by_policy_id: Mapping[str, AgentAttributePlanEntry]
    sampled_names: tuple[str, ...]

    def resolve(
        self,
        configuration: NDArray[np.float32],
        simulation_settings: SimulationSettings,
        *,
        parameter_row_index: int,
        max_n_agents: int,
        policy_for_node: PolicyForNode,
    ) -> ResolvedAgentAttributes:
        activation_weights = np.empty(
            simulation_settings.n_agents,
            dtype=np.float32,
        )
        sampled_values = np.full(
            (max_n_agents, len(self.sampled_names)),
            np.nan,
            dtype=np.float32,
        )

        for node_id in range(simulation_settings.n_agents):
            policy_id, _ = policy_for_node(node_id, configuration)
            entry = self.entry_by_policy_id[policy_id]

            source_value = float(configuration[entry.source_column])
            if entry.sampled_column is not None:
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        (
                            int(simulation_settings.seed),
                            int(simulation_settings.n_agents),
                            int(parameter_row_index),
                            int(node_id),
                            int(entry.source_column),
                        )
                    )
                )
                assert entry.minimum is not None
                assert entry.maximum is not None
                source_value = float(
                    np.clip(
                        rng.normal(loc=source_value, scale=entry.std),
                        entry.minimum,
                        entry.maximum,
                    )
                )

            activation_weight = _resolve_activation_weight(
                source_value,
                transform=entry.definition.transform,
                n_agents=simulation_settings.n_agents,
            )
            activation_weights[node_id] = activation_weight
            if entry.sampled_column is not None:
                sampled_values[node_id, entry.sampled_column] = activation_weight

        return ResolvedAgentAttributes(
            activation_weights=activation_weights,
            sampled_values=sampled_values,
        )


def collect_agent_attribute_definitions(
    population: Mapping[str, object],
    policy_configs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, ParameterSpec], tuple[AgentAttributeDefinition, ...]]:
    """Flatten agent-attribute sources into ordinary parameter specifications."""

    definitions: list[AgentAttributeDefinition] = []

    if "agent_parameters" not in population:
        raise ValueError(
            "population.agent_parameters must define 'activation_weight'"
        )
    global_attributes = _agent_attribute_mapping(
        population["agent_parameters"],
        location="population.agent_parameters",
    )
    if ACTIVATION_WEIGHT not in global_attributes:
        raise ValueError(
            "population.agent_parameters must define 'activation_weight'"
        )
    for attribute_name, raw_spec in global_attributes.items():
        definition = _parse_definition(
            policy_id=None,
            attribute_name=attribute_name,
            raw_spec=raw_spec,
        )
        definitions.append(definition)

    for policy_config in policy_configs:
        policy_id = str(policy_config["id"])
        attributes = _agent_attribute_mapping(
            policy_config.get("agent_parameters", {}),
            location=f"population.policies[{policy_id}].agent_parameters",
        )
        for attribute_name, raw_spec in attributes.items():
            definition = _parse_definition(
                policy_id=policy_id,
                attribute_name=attribute_name,
                raw_spec=raw_spec,
            )
            definitions.append(definition)

    policy_ids = {str(policy_config["id"]) for policy_config in policy_configs}
    override_keys = {
        (definition.policy_id, definition.attribute_name)
        for definition in definitions
        if definition.policy_id is not None
    }
    effective_definitions = tuple(
        definition
        for definition in definitions
        if definition.policy_id is not None
        or any(
            (policy_id, definition.attribute_name) not in override_keys
            for policy_id in policy_ids
        )
    )
    source_specs = {
        definition.parameter_name: definition.parameter_spec
        for definition in effective_definitions
    }
    return source_specs, tuple(definitions)


def build_agent_attribute_plan(
    definitions: Sequence[AgentAttributeDefinition],
    *,
    configuration_column_by_name: Mapping[str, int],
    policy_entries: Sequence[PolicyEntry],
    n_agents_values: Sequence[int],
    n_actors: int,
) -> AgentAttributePlan:
    global_definition = next(
        definition for definition in definitions if definition.policy_id is None
    )
    override_by_policy_id = {
        definition.policy_id: definition
        for definition in definitions
        if definition.policy_id is not None
    }
    effective_by_policy_id = {
        policy_id: override_by_policy_id.get(policy_id, global_definition)
        for policy_id, _ in policy_entries
    }

    has_multiple_effective_actors = any(
        min(n_actors, n_agents) > 1 for n_agents in n_agents_values
    )
    if has_multiple_effective_actors and not _attributes_are_uniform(
        effective_by_policy_id
    ):
        raise ValueError("unequal activation weights require n_actors == 1")

    if any(
        definition.transform == "scaled_other_agents"
        for definition in effective_by_policy_id.values()
    ) and any(n_agents < 2 for n_agents in n_agents_values):
        raise ValueError("scaled_other_agents requires n_agents >= 2")

    sampled_names: list[str] = []
    sampled_column_by_parameter_name: dict[str, int] = {}
    entry_by_policy_id: dict[str, AgentAttributePlanEntry] = {}

    for policy_id, definition in effective_by_policy_id.items():
        minimum: float | None = None
        maximum: float | None = None
        std = 0.0
        if str(definition.parameter_spec["kind"]) == "continuous":
            minimum, maximum, std = validate_continuous_parameter_spec(
                definition.parameter_name,
                definition.parameter_spec,
            )

        sampled_column: int | None = None
        if std > 0.0:
            sampled_column = sampled_column_by_parameter_name.get(
                definition.parameter_name
            )
            if sampled_column is None:
                sampled_column = len(sampled_names)
                sampled_column_by_parameter_name[definition.parameter_name] = (
                    sampled_column
                )
                sampled_names.append(definition.output_name)

        entry_by_policy_id[policy_id] = AgentAttributePlanEntry(
            definition=definition,
            source_column=configuration_column_by_name[definition.parameter_name],
            sampled_column=sampled_column,
            minimum=minimum,
            maximum=maximum,
            std=std,
        )

    return AgentAttributePlan(
        entry_by_policy_id=entry_by_policy_id,
        sampled_names=tuple(sampled_names),
    )


def _agent_attribute_mapping(raw: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{location} must be a mapping")
    unsupported = set(raw) - {ACTIVATION_WEIGHT}
    if unsupported:
        name = sorted(map(str, unsupported))[0]
        raise ValueError(f"unsupported agent parameter {name!r}")
    return cast(Mapping[str, object], raw)


def _parse_definition(
    *,
    policy_id: str | None,
    attribute_name: str,
    raw_spec: object,
) -> AgentAttributeDefinition:
    if attribute_name != ACTIVATION_WEIGHT:
        raise ValueError(f"unsupported agent parameter {attribute_name!r}")
    if not isinstance(raw_spec, Mapping) or "kind" not in raw_spec:
        raise ValueError(f"agent parameter {attribute_name!r} must be a parameter spec")

    kind = str(raw_spec["kind"])
    prefix = "agent" if policy_id is None else f"{policy_id}.agent"
    if kind == "scaled_other_agents":
        share_spec = raw_spec.get("share")
        if not isinstance(share_spec, Mapping) or "kind" not in share_spec:
            raise ValueError("scaled_other_agents requires a share parameter spec")
        parameter_name = f"{prefix}.activation_share"
        parameter_spec = cast(ParameterSpec, share_spec)
        _validate_numeric_spec(
            parameter_name,
            parameter_spec,
            lower=0.0,
            upper=1.0,
        )
        return AgentAttributeDefinition(
            policy_id=policy_id,
            attribute_name=attribute_name,
            parameter_name=parameter_name,
            output_name=f"{prefix}.activation_weight",
            transform="scaled_other_agents",
            parameter_spec=parameter_spec,
        )

    parameter_name = f"{prefix}.activation_weight"
    parameter_spec = cast(ParameterSpec, raw_spec)
    _validate_numeric_spec(parameter_name, parameter_spec, lower=0.0, upper=None)
    return AgentAttributeDefinition(
        policy_id=policy_id,
        attribute_name=attribute_name,
        parameter_name=parameter_name,
        output_name=parameter_name,
        transform="identity",
        parameter_spec=parameter_spec,
    )


def _validate_numeric_spec(
    name: str,
    spec: ParameterSpec,
    *,
    lower: float,
    upper: float | None,
) -> None:
    kind = str(spec["kind"])
    values: Sequence[object]
    if kind == "fixed":
        values = (spec.get("value"),)
    elif kind == "sampled_choice":
        raw_values = spec.get("values")
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
            raise ValueError(f"{name}.values must be a sequence")
        values = tuple(raw_values)
        if not values:
            raise ValueError(f"{name}.values must not be empty")
    elif kind == "continuous":
        minimum, maximum, _ = validate_continuous_parameter_spec(name, spec)
        with np.errstate(over="ignore", under="ignore"):
            stored_minimum = np.float32(minimum)
            stored_maximum = np.float32(maximum)
        if not stored_minimum < stored_maximum:
            raise ValueError(f"{name} bounds must remain distinct as float32 values")
        values = (minimum, maximum)
    else:
        raise ValueError(
            f"agent parameter {name!r} does not support parameter kind {kind!r}"
        )

    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} values must be finite real numbers")
        numeric = float(value)
        with np.errstate(over="ignore", under="ignore"):
            stored = float(np.float32(numeric))
        if not np.isfinite(numeric) or not np.isfinite(stored):
            raise ValueError(f"{name} values must be finite real numbers")
        if stored <= lower or (upper is not None and stored >= upper):
            interval = f"({lower:g}, {upper:g})" if upper is not None else f"> {lower:g}"
            raise ValueError(f"{name} values must be {interval}")


def _resolve_activation_weight(
    source_value: float,
    *,
    transform: AttributeTransform,
    n_agents: int,
) -> np.float32:
    if transform == "identity":
        value = source_value
    else:
        value = (source_value / (1.0 - source_value)) * (n_agents - 1)
    with np.errstate(over="ignore", under="ignore"):
        stored = np.float32(value)
    if not np.isfinite(value) or not np.isfinite(stored) or stored <= 0.0:
        raise ValueError("resolved activation_weight must be finite and positive")
    return stored


def _attributes_are_uniform(
    effective_by_policy_id: Mapping[str, AgentAttributeDefinition],
) -> bool:
    definitions = tuple(effective_by_policy_id.values())
    if not definitions:
        return True
    first = definitions[0]
    if str(first.parameter_spec["kind"]) == "continuous":
        _, _, std = validate_continuous_parameter_spec(
            first.parameter_name,
            first.parameter_spec,
        )
        if std > 0.0:
            return False
    return all(
        definition.transform == first.transform
        and definition.parameter_name == first.parameter_name
        for definition in definitions[1:]
    )


__all__ = [
    "AgentAttributeDefinition",
    "AgentAttributePlan",
    "ResolvedAgentAttributes",
    "build_agent_attribute_plan",
    "collect_agent_attribute_definitions",
]
