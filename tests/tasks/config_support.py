from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def fixed(value: float | int) -> dict[str, object]:
    return {"kind": "fixed", "value": value}


def continuous(minimum: float, maximum: float) -> dict[str, object]:
    return {"kind": "continuous", "min": minimum, "max": maximum}


def sampled_choice(values: Sequence[float | int]) -> dict[str, object]:
    return {"kind": "sampled_choice", "values": list(values)}


def sampled_discrete_range(
    minimum: int,
    maximum: int,
    *,
    step: int = 1,
) -> dict[str, object]:
    return {
        "kind": "sampled_discrete_range",
        "min": minimum,
        "max": maximum,
        "step": step,
    }


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
    policies: Sequence[Mapping[str, Any]] | None = None,
    policy_proportions: Mapping[str, object] | None = None,
    observers: Mapping[str, object] | None = None,
) -> dict[str, object]:
    policy_configs = list(
        policies or [{"id": "random", "kind": "random", "parameters": {}}]
    )
    population: dict[str, object] = {
        "n_agents": list(n_agents) if not isinstance(n_agents, int) else n_agents,
        "visibility_distance": dict(visibility_distance or fixed(4)),
        "policies": policy_configs,
        "policy_proportions": dict(
            policy_proportions
            or {"policy": str(policy_configs[0]["id"]), "values": [1.0]}
        ),
    }

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
) -> dict[str, object]:
    return {
        "id": policy_id,
        "kind": kind,
        "parameters": dict(parameters or {}),
    }


def default_observers() -> dict[str, object]:
    return {
        "defaults": {
            "sampling": {"kind": "periodic", "every": 1},
            "aggregations": ["mean"],
        },
        "outputs": [
            {"name": "density", "output": "mean"},
            {"name": "centered_mean_welfare", "output": "mean"},
        ],
    }
