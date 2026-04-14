"""Reusable helpers for assembling one simulation runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .agent import Agent
from .environment import Environment
from .graph import create_empty_graph
from .model import ActorScheduler
from .observability import SimulationObserver
from .simulator import Simulator


@dataclass(frozen=True)
class SimulationSettings:
    """Fixed settings for one simulation run family."""

    n_agents: int
    n_actors: int
    seed: int
    max_steps: int
    n_episodes: int
    stagnation_patience: int | None
    initial_graph_kind: str = "empty"


def create_environment(
    *,
    simulation_settings: SimulationSettings,
    seed: int,
) -> Environment:
    if simulation_settings.initial_graph_kind != "empty":
        raise ValueError(
            f"unsupported initial graph kind {simulation_settings.initial_graph_kind!r}; "
            "only 'empty' is supported"
        )

    return Environment(
        create_empty_graph(simulation_settings.n_agents),
        seed=seed,
    )


def create_agents(
    *,
    simulation_settings: SimulationSettings,
    build_agent: Callable[[int], Agent],
) -> tuple[Agent, ...]:
    return tuple(build_agent(node_id) for node_id in range(simulation_settings.n_agents))


def create_simulator(
    *,
    simulation_settings: SimulationSettings,
    environment: Environment,
    agents: tuple[Agent, ...],
    observers: tuple[SimulationObserver, ...],
    seed: int,
) -> Simulator:
    actor_scheduler = ActorScheduler(
        seed=seed,
        n_agents=len(agents),
        n_actors=simulation_settings.n_actors,
    )
    return Simulator(
        env=environment,
        agents=agents,
        actor_scheduler=actor_scheduler,
        max_steps=simulation_settings.max_steps,
        rng=np.random.default_rng(seed),
        observers=observers,
        stagnation_patience=simulation_settings.stagnation_patience,
    )


def create_simulation_runtime(
    *,
    simulation_settings: SimulationSettings,
    build_agent: Callable[[int], Agent],
    observers: tuple[SimulationObserver, ...],
    seed: int,
) -> tuple[Environment, tuple[Agent, ...], Simulator]:
    environment = create_environment(
        simulation_settings=simulation_settings,
        seed=seed,
    )
    agents = create_agents(
        simulation_settings=simulation_settings,
        build_agent=build_agent,
    )
    simulator = create_simulator(
        simulation_settings=simulation_settings,
        environment=environment,
        agents=agents,
        observers=observers,
        seed=seed,
    )
    return environment, agents, simulator
