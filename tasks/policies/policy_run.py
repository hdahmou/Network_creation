"""Prepare policy simulation runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ..execution.schema import PreparedRun
from .agent_parameters import build_sampled_agent_parameter_plan
from .agent_attributes import (
    AgentAttributeDefinition,
    build_agent_attribute_plan,
)
from .agents import PolicyAgentBuilder
from .observers import build_observer_specs
from .outputs import build_policy_output_spec, build_result_metadata
from .parameter_space import ParameterSpec
from .simulation import PolicySimulationRuntime
from .sweep import (
    build_sweep,
    collect_parameter_specs,
    parse_policy_entries,
    sample_parameters,
)


def prepare_policy_run(config: Mapping[str, Any]) -> PreparedRun[int]:
    policy_config = cast(Mapping[str, Any], config)
    policy_entries = parse_policy_entries(policy_config)
    policy_ids = tuple(policy_id for policy_id, _ in policy_entries)
    input_spec = collect_parameter_specs(policy_config)
    parameter_specs = cast(Mapping[str, ParameterSpec], input_spec["parameters"])
    policy_parameter_specs = cast(
        Mapping[str, ParameterSpec], input_spec["policy_parameters"]
    )
    agent_attribute_definitions = cast(
        tuple[AgentAttributeDefinition, ...],
        input_spec["agent_attribute_definitions"],
    )
    parameter_samples = sample_parameters(input_spec)
    sweep = build_sweep(
        policy_config,
        parameter_samples,
        parameter_specs=parameter_specs,
        policy_ids=policy_ids,
    )
    max_n_agents = max(sweep.n_agents_values)
    observer_specs = build_observer_specs(
        raw_observers=policy_config["observers"],
        max_steps=sweep.storage_max_steps,
        n_episodes=sweep.n_episodes,
        n_agent_slots=max_n_agents,
        policy_ids=policy_ids,
    )
    sampled_parameter_plan = build_sampled_agent_parameter_plan(
        policy_parameter_specs,
        configuration_column_by_name=sweep.column_index_by_name,
        policy_entries=policy_entries,
    )
    agent_attribute_plan = build_agent_attribute_plan(
        agent_attribute_definitions,
        configuration_column_by_name=sweep.column_index_by_name,
        policy_entries=policy_entries,
        n_agents_values=sweep.n_agents_values,
        n_actors=sweep.n_actors,
    )
    sampled_agent_parameter_names = (
        *sampled_parameter_plan.names,
        *agent_attribute_plan.sampled_names,
    )
    agent_builder = PolicyAgentBuilder(
        policy_entries=policy_entries,
        configuration_names=sweep.configuration_names,
        column_index_by_name=sweep.column_index_by_name,
        sampled_parameters_by_policy_id=sampled_parameter_plan.parameters_by_policy_id,
    )
    runtime = PolicySimulationRuntime(
        sweep=sweep,
        observer_specs=observer_specs,
        agent_builder=agent_builder,
        sampled_agent_parameter_plan=sampled_parameter_plan,
        agent_attribute_plan=agent_attribute_plan,
        max_n_agents=max_n_agents,
    )
    result_attrs, result_arrays = build_result_metadata(
        config=config,
        sweep=sweep,
        parameter_specs=parameter_specs,
        sampled_agent_parameter_names=sampled_agent_parameter_names,
    )
    output_schema = build_policy_output_spec(
        tuple(field for spec in observer_specs for field in spec.fields),
        sampled_agent_parameter_names=sampled_agent_parameter_names,
        max_n_agents=max_n_agents,
    )

    return PreparedRun(
        inputs=tuple(range(sweep.total_inputs)),
        output_schema=output_schema,
        run=runtime.run,
        result_attrs=result_attrs,
        result_arrays=result_arrays,
    )
