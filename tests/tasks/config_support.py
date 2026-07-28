from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def fixed(value: float | int) -> dict[str, object]:
    return {"kind": "fixed", "value": value}


def continuous(
    minimum: float,
    maximum: float,
    *,
    heterogeneity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    spec: dict[str, object] = {
        "kind": "continuous",
        "min": minimum,
        "max": maximum,
    }
    if heterogeneity is not None:
        spec["heterogeneity"] = dict(heterogeneity)
    return spec


def normal_heterogeneity(std: float) -> dict[str, object]:
    return {"kind": "normal", "std": std}


def discrete_offset_heterogeneity(
    offsets: Sequence[int],
) -> dict[str, object]:
    return {"kind": "discrete_offsets", "offsets": list(offsets)}


def sampled_choice(values: Sequence[float | int]) -> dict[str, object]:
    return {"kind": "sampled_choice", "values": list(values)}


def sampled_discrete_range(
    minimum: int,
    maximum: int,
    *,
    step: int = 1,
    heterogeneity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    spec: dict[str, object] = {
        "kind": "sampled_discrete_range",
        "min": minimum,
        "max": maximum,
        "step": step,
    }
    if heterogeneity is not None:
        spec["heterogeneity"] = dict(heterogeneity)
    return spec


def scaled_inverse_agents(numerator: float, factor: float) -> dict[str, object]:
    return {
        "kind": "scaled_inverse_agents",
        "numerator": numerator,
        "factor": factor,
    }


def task_config(
    *,
    n_agents: int | Sequence[int] = 4,
    n_actors: int = 4,
    log2_budget: int = 0,
    seed: int = 123,
    simulator_seed: int = 11,
    max_steps: int = 3,
    n_episodes: int = 2,
    stagnation_patience: int | None = None,
    game: Mapping[str, object] | None = None,
    visibility_distance: Mapping[str, object] | None = None,
    agent_parameters: Mapping[str, object] | None = None,
    policies: Sequence[Mapping[str, Any]] | None = None,
    policy_proportions: Mapping[str, object] | None = None,
    policy_counts: Mapping[str, object] | None = None,
    observers: Mapping[str, object] | None = None,
) -> dict[str, object]:
    policy_configs = list(
        policies or [{"id": "random", "kind": "random", "parameters": {}}]
    )
    population: dict[str, object] = {
        "n_agents": list(n_agents) if not isinstance(n_agents, int) else n_agents,
        "visibility_distance": dict(visibility_distance or fixed(4)),
        "agent_parameters": dict(
            agent_parameters
            if agent_parameters is not None
            else {"activation_weight": fixed(1.0)}
        ),
        "policies": policy_configs,
    }
    if policy_counts is None:
        population["policy_proportions"] = dict(
            policy_proportions
            or {"policy": str(policy_configs[0]["id"]), "values": [1.0]}
        )
    else:
        population["policy_counts"] = dict(policy_counts)

    return {
        "sampling": {"log2_budget": log2_budget, "seed": seed},
        "simulator": {
            "seed": simulator_seed,
            "max_steps": max_steps,
            "n_episodes": n_episodes,
            "stagnation_patience": stagnation_patience,
            "actor_scheduler": {"n_actors": n_actors},
            "initial_graph": {"kind": "empty"},
        },
        "game": dict(
            game
            or {
                "benefit_to_cost_ratio": fixed(0.5),
                "distance_discount": fixed(0.5),
            }
        ),
        "population": population,
        "observers": dict(observers or default_observers()),
    }


def execution_config(
    *,
    scheduler_address: str = "inproc://tests",
    output_dir: str = "artifacts/simulation_runs",
    write_batch_size: int = 2,
    **task_kwargs: Any,
) -> dict[str, object]:
    config = task_config(**task_kwargs)
    return {
        "execution": {
            "scheduler_address": scheduler_address,
            "output_dir": output_dir,
            "write_batch_size": write_batch_size,
        },
        **config,
    }


def policy(
    policy_id: str,
    kind: str,
    parameters: Mapping[str, object] | None = None,
    *,
    agent_parameters: Mapping[str, object] | None = None,
) -> dict[str, object]:
    output: dict[str, object] = {
        "id": policy_id,
        "kind": kind,
        "parameters": dict(parameters or {}),
    }
    if agent_parameters is not None:
        output["agent_parameters"] = dict(agent_parameters)
    return output


def scaled_other_agents(share: Mapping[str, object]) -> dict[str, object]:
    return {"kind": "scaled_other_agents", "share": dict(share)}


def default_observers() -> dict[str, object]:
    return {
        "defaults": {
            "sampling": {"kind": "periodic", "every": 1},
            "aggregations": ["mean"],
        },
        "outputs": [
            {"name": "density"},
            {"name": "centered_mean_welfare"},
        ],
    }
