"""Build policy parameter samples and simulation sweeps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from network_creation.core.runtime import SimulationSettings
from .parameter_space import generate_parameter_space

type ParameterSpec = Mapping[str, object]
type ParameterConfig = Mapping[str, ParameterSpec]
type PolicyEntry = tuple[str, str]
type PolicyProportionSweep = tuple[str, tuple[float, ...]]


_POPULATION_STRUCTURAL_KEYS = frozenset(
    {"n_agents", "policies", "policy_proportions"}
)


@dataclass(frozen=True)
class ParameterSamples:
    parameter_names: tuple[str, ...]
    parameter_values: NDArray[np.float32]
    column_index_by_name: Mapping[str, int]


def _input_coordinates(
    input_index: int,
    *,
    n_policy_proportions: int,
    n_parameter_rows: int,
) -> tuple[int, int, int]:
    parameter_row_index = input_index % n_parameter_rows
    outer_index = input_index // n_parameter_rows
    policy_proportion_index = outer_index % n_policy_proportions
    n_agents_index = outer_index // n_policy_proportions
    return n_agents_index, policy_proportion_index, parameter_row_index


@dataclass(frozen=True)
class SweepPlan:
    parameter_samples: ParameterSamples
    configuration_names: tuple[str, ...]
    configuration_values: NDArray[np.float32]
    column_index_by_name: Mapping[str, int]
    n_agents_values: tuple[int, ...]
    policy_proportions: PolicyProportionSweep
    n_actors: int
    seed: int
    max_steps: int
    n_episodes: int
    stagnation_patience: int | None
    initial_graph_kind: str = "empty"
    initial_graph_p: float | None = None

    @property
    def n_parameter_rows(self) -> int:
        return int(self.parameter_samples.parameter_values.shape[0])

    @property
    def n_policy_proportions(self) -> int:
        _, values = self.policy_proportions
        return len(values)

    @property
    def total_inputs(self) -> int:
        return (
            len(self.n_agents_values)
            * self.n_policy_proportions
            * self.n_parameter_rows
        )

    def coordinates(self, input_index: int) -> tuple[int, int, int]:
        return _input_coordinates(
            input_index,
            n_policy_proportions=self.n_policy_proportions,
            n_parameter_rows=self.n_parameter_rows,
        )

    def configuration_at(self, input_index: int) -> NDArray[np.float32]:
        return self.configuration_values[self.coordinates(input_index)]

    def parameter_values_for_results(self) -> NDArray[np.float32]:
        return np.asarray(self.configuration_values[..., 1:], dtype=np.float32)

    @property
    def parameter_names_for_results(self) -> tuple[str, ...]:
        return self.configuration_names[1:]

    def resolve(self, configuration: NDArray[np.float32]) -> SimulationSettings:
        n_agents = int(configuration[self.column_index_by_name["n_agents"]])
        return SimulationSettings(
            n_agents=n_agents,
            n_actors=min(self.n_actors, n_agents),
            seed=self.seed,
            max_steps=self.max_steps,
            n_episodes=self.n_episodes,
            stagnation_patience=self.stagnation_patience,
            initial_graph_kind=self.initial_graph_kind,
            initial_graph_p=self.initial_graph_p,
        )


def parse_policy_entries(config: Mapping[str, Any]) -> tuple[PolicyEntry, ...]:
    population = cast(Mapping[str, Any], config["population"])
    raw_policies = population.get("policies")
    if not isinstance(raw_policies, list) or not raw_policies:
        raise ValueError("population.policies must contain at least one policy")
    if len(raw_policies) > 2:
        raise ValueError("population.policies supports at most two policies")

    policy_entries: list[PolicyEntry] = []
    for raw_policy in raw_policies:
        entry = cast(Mapping[str, Any], raw_policy)
        policy_id = str(entry["id"])
        policy_entries.append((policy_id, str(entry["kind"])))

    return tuple(policy_entries)


def collect_parameter_specs(config: Mapping[str, Any]) -> dict[str, object]:
    sampling = cast(Mapping[str, Any], config["sampling"])
    parameters: dict[str, ParameterSpec] = {}
    population = cast(Mapping[str, object], config["population"])
    policy_entries = cast(Sequence[Mapping[str, Any]], population["policies"])
    should_namespace_policy_parameters = len(policy_entries) > 1

    for name, spec in cast(Mapping[str, object], config.get("game", {})).items():
        if not isinstance(spec, Mapping) or "kind" not in spec:
            raise ValueError(f"game.{name} must be a parameter spec")
        parameters[name] = cast(ParameterSpec, spec)

    for name, spec in population.items():
        if name in _POPULATION_STRUCTURAL_KEYS:
            continue
        if not isinstance(spec, Mapping) or "kind" not in spec:
            raise ValueError(f"population.{name} must be a parameter spec")
        parameters[name] = cast(ParameterSpec, spec)

    for policy_entry in policy_entries:
        policy_id = str(policy_entry["id"])
        raw_parameters = cast(
            Mapping[str, object],
            policy_entry.get("parameters", {}),
        )
        for name, spec in raw_parameters.items():
            if not isinstance(spec, Mapping) or "kind" not in spec:
                raise ValueError(
                    f"population.policies[{policy_id}].parameters.{name} "
                    "must be a parameter spec"
                )
            column_name = (
                f"{policy_id}.{name}" if should_namespace_policy_parameters else name
            )
            parameters[column_name] = cast(ParameterSpec, spec)

    return {
        "log2_budget": sampling["log2_budget"],
        "seed": sampling["seed"],
        "parameters": parameters,
    }


def _resolve_scaled_inverse_agents(
    configuration_values: NDArray[np.float32],
    *,
    configuration_names: Sequence[str],
    n_agents_values: Sequence[int],
    parameter_specs: ParameterConfig,
) -> None:
    n_agents_array = np.asarray(n_agents_values, dtype=np.float64)

    for parameter_name, spec in parameter_specs.items():
        if str(spec["kind"]) != "scaled_inverse_agents":
            continue

        numerator = float(cast(Any, spec["numerator"]))
        factor = float(cast(Any, spec["factor"]))
        resolved_values = numerator / (factor * n_agents_array)
        column_index = configuration_names.index(parameter_name)
        resolved = np.asarray(resolved_values, dtype=np.float32)
        if configuration_values.ndim == 3:
            configuration_values[:, :, column_index] = resolved[:, None]
        elif configuration_values.ndim == 4:
            configuration_values[:, :, :, column_index] = resolved[:, None, None]
        else:
            raise ValueError("configuration_values must be 3D or 4D")


def _load_policy_proportions(
    raw: Mapping[str, Any],
    *,
    policy_ids: Sequence[str],
) -> PolicyProportionSweep:
    raw_proportions = cast(Mapping[str, Any], raw["policy_proportions"])
    policy_id = str(raw_proportions["policy"])
    if policy_id not in policy_ids:
        raise ValueError(
            "population.policy_proportions.policy must match one policy id"
        )

    values = cast(Sequence[Any], raw_proportions["values"])
    proportions = tuple(float(value) for value in values)
    for index, proportion in enumerate(proportions):
        if (
            isinstance(values[index], bool)
            or not isinstance(values[index], Real)
            or not 0.0 <= proportion <= 1.0
        ):
            raise ValueError(
                "population.policy_proportions.values"
                f"[{index}] must be in [0, 1]"
            )

    return policy_id, proportions


def _load_initial_graph(raw: Mapping[str, Any]) -> tuple[str, float | None]:
    raw_initial_graph = raw.get("initial_graph")
    initial_graph = {} if raw_initial_graph is None else raw_initial_graph
    if not isinstance(initial_graph, Mapping):
        raise ValueError("simulator.initial_graph must be an object")

    kind = str(initial_graph.get("kind", "empty"))
    if kind == "empty":
        return kind, None

    if kind != "erdos_renyi":
        return kind, None

    if "p" not in initial_graph:
        raise ValueError("simulator.initial_graph.p is required for erdos_renyi")

    raw_p = initial_graph["p"]
    if isinstance(raw_p, bool) or not isinstance(raw_p, Real):
        raise ValueError("simulator.initial_graph.p must be a number in [0, 1]")

    p = float(raw_p)
    if not 0.0 <= p <= 1.0:
        raise ValueError("simulator.initial_graph.p must be a number in [0, 1]")

    return kind, p


def _load_n_agents_values(raw_n_agents: Any) -> tuple[int, ...]:
    if isinstance(raw_n_agents, Mapping):
        if str(raw_n_agents["kind"]) != "sampled_discrete_range":
            raise ValueError(
                "population.n_agents object must use kind "
                "'sampled_discrete_range'"
            )
        values = np.arange(
            int(cast(Any, raw_n_agents["min"])),
            int(cast(Any, raw_n_agents["max"])) + 1,
            int(cast(Any, raw_n_agents.get("step", 1))),
            dtype=np.int64,
        ).tolist()
    else:
        values = raw_n_agents if isinstance(raw_n_agents, list) else [raw_n_agents]
    if not values:
        raise ValueError("population.n_agents must contain at least one value")

    n_agents_values: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
            field_name = (
                f"population.n_agents[{index}]"
                if isinstance(raw_n_agents, list)
                else "population.n_agents"
            )
            raise ValueError(f"{field_name} must be a positive integer")
        n_agents_values.append(int(value))
    return tuple(n_agents_values)


def _build_configuration_grid(
    *,
    n_agents_values: Sequence[int],
    target_id: str,
    proportions: Sequence[float],
    policy_ids: Sequence[str],
    parameter_samples: ParameterSamples,
) -> tuple[tuple[str, ...], NDArray[np.float32]]:
    count_names = tuple(f"{policy_id}_count" for policy_id in policy_ids)
    configuration_names = (
        "n_agents",
        f"{target_id}_proportion",
        *count_names,
        *parameter_samples.parameter_names,
    )
    n_parameter_rows = int(parameter_samples.parameter_values.shape[0])
    configuration_values: NDArray[np.float32] = np.empty(
        (
            len(n_agents_values),
            len(proportions),
            n_parameter_rows,
            len(configuration_names),
        ),
        dtype=np.float32,
    )
    n_agents_array = np.asarray(n_agents_values, dtype=np.float32)
    proportions_array = np.asarray(proportions, dtype=np.float32)
    target_counts = np.rint(
        proportions_array[None, :] * n_agents_array[:, None]
    ).astype(np.float32)
    configuration_values[:, :, :, 0] = n_agents_array[:, None, None]
    configuration_values[:, :, :, 1] = proportions_array[None, :, None]

    count_start = 2
    for offset, policy_id in enumerate(policy_ids):
        if len(policy_ids) == 1 or policy_id == target_id:
            counts = target_counts
        else:
            counts = n_agents_array[:, None] - target_counts
        configuration_values[:, :, :, count_start + offset] = counts[:, :, None]

    parameter_start = count_start + len(count_names)
    configuration_values[:, :, :, parameter_start:] = parameter_samples.parameter_values[
        None,
        None,
        :,
        :,
    ]
    return configuration_names, configuration_values


def build_sweep(
    config: Mapping[str, Any],
    parameter_samples: ParameterSamples,
    *,
    parameter_specs: ParameterConfig,
    policy_ids: Sequence[str] = (),
) -> SweepPlan:
    population = cast(Mapping[str, Any], config["population"])
    simulator = cast(Mapping[str, Any], config["simulator"])
    initial_graph_kind, initial_graph_p = _load_initial_graph(simulator)
    stagnation_patience = simulator.get("stagnation_patience")
    n_agents_values = _load_n_agents_values(population["n_agents"])

    policy_proportions = _load_policy_proportions(
        population,
        policy_ids=policy_ids,
    )
    target_id, proportions = policy_proportions
    configuration_names, configuration_values = _build_configuration_grid(
        n_agents_values=n_agents_values,
        target_id=target_id,
        proportions=proportions,
        policy_ids=policy_ids,
        parameter_samples=parameter_samples,
    )
    _resolve_scaled_inverse_agents(
        configuration_values,
        configuration_names=configuration_names,
        n_agents_values=n_agents_values,
        parameter_specs=parameter_specs,
    )

    return SweepPlan(
        parameter_samples=parameter_samples,
        configuration_names=configuration_names,
        configuration_values=configuration_values,
        column_index_by_name={
            name: index for index, name in enumerate(configuration_names)
        },
        n_agents_values=n_agents_values,
        policy_proportions=policy_proportions,
        n_actors=int(simulator["actor_scheduler"]["n_actors"]),
        seed=int(simulator["seed"]),
        max_steps=int(simulator["max_steps"]),
        n_episodes=int(simulator["n_episodes"]),
        stagnation_patience=(
            None if stagnation_patience is None else int(stagnation_patience)
        ),
        initial_graph_kind=initial_graph_kind,
        initial_graph_p=initial_graph_p,
    )


def infer_parameter_bounds(
    parameter_values: NDArray[np.float32],
    *,
    parameter_names: Sequence[str],
    parameter_specs: ParameterConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    flat_parameters = parameter_values.reshape(-1, parameter_values.shape[-1])
    low = np.nanmin(flat_parameters, axis=0).astype(np.float32)
    high = np.nanmax(flat_parameters, axis=0).astype(np.float32)

    for index, name in enumerate(parameter_names):
        spec = parameter_specs.get(name)
        if spec is None:
            continue
        kind = str(spec["kind"])
        if kind == "continuous":
            low[index] = np.float32(cast(Any, spec["min"]))
            high[index] = np.float32(cast(Any, spec["max"]))
        elif kind == "fixed":
            low[index] = np.float32(cast(Any, spec["value"]))
            high[index] = np.float32(cast(Any, spec["value"]))
        elif kind == "sampled_choice":
            values = np.asarray(spec["values"], dtype=np.float32)
            low[index] = np.min(values)
            high[index] = np.max(values)
        elif kind == "sampled_discrete_range":
            low[index] = np.float32(cast(Any, spec["min"]))
            high[index] = np.float32(cast(Any, spec["max"]))

    return low, high


def sample_parameters(input_spec: Mapping[str, object]) -> ParameterSamples:
    parameters = cast(ParameterConfig, input_spec["parameters"])
    sampler_seed = int(cast(Any, input_spec["seed"]))
    log2_budget = int(cast(Any, input_spec["log2_budget"]))
    if log2_budget < 0:
        raise ValueError("log2_budget must be non-negative")

    sample_budget = 2**log2_budget
    parameter_names, parameter_values = generate_parameter_space(
        parameters,
        sample_budget=sample_budget,
        sampler_seed=sampler_seed,
    )
    column_index_by_name = {name: index for index, name in enumerate(parameter_names)}

    return ParameterSamples(
        parameter_names=parameter_names,
        parameter_values=parameter_values,
        column_index_by_name=column_index_by_name,
    )
