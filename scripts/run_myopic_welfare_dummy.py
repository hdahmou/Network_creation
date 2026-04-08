#!/usr/bin/env python3
"""Silent fixed-workload script for profiling the myopic policy."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from network_creation.core.agent import Agent
from network_creation.core.observability import ObservationSampling
from network_creation.core.runtime import (
    SimulationSettings,
    create_simulation_runtime,
)
from network_creation.observers.social_welfare_observer import (
    NormalizedWelfareObserver,
)
from network_creation.policies.myopic_payoff import MyopicPayoffPolicy


N_AGENTS = 50
MAX_STEPS = 360
N_EPISODES = 32
PARAMETER_SETS = (
    {
        "benefit_to_cost_ratio": 0.5,
        "distance_discount": 0.9,
        "bridge_probability": 0.4,
    },
    {
        "benefit_to_cost_ratio": 0.5,
        "distance_discount": 0.7,
        "bridge_probability": 0.2,
    },
)


def run_case(case_index: int, policy_kwargs: dict[str, float]) -> None:
    def build_agent(node_id: int) -> Agent:
        policy_seed = 1_000 * (case_index + 1) + node_id
        return Agent(
            id=node_id,
            visibility_fraction=1.0,
            policy=MyopicPayoffPolicy(
                seed=policy_seed,
                benefit_to_cost_ratio=policy_kwargs["benefit_to_cost_ratio"],
                distance_discount=policy_kwargs["distance_discount"],
                bridge_probability=policy_kwargs["bridge_probability"],
            ),
            seed=policy_seed,
        )

    settings = SimulationSettings(
        n_agents=N_AGENTS,
        actor_fraction=1.0,
        seed=100 + case_index,
        max_steps=MAX_STEPS,
        n_episodes=N_EPISODES,
        stagnation_patience=None,
    )
    welfare_observer = NormalizedWelfareObserver(
        observation_sampling=ObservationSampling.periodic(1),
        benefit_to_cost_ratio=policy_kwargs["benefit_to_cost_ratio"],
        distance_discount=policy_kwargs["distance_discount"],
        aggregations=(),
    )
    _, _, simulator = create_simulation_runtime(
        simulation_settings=settings,
        build_agent=build_agent,
        observers=(welfare_observer,),
        seed=10_000 + case_index,
    )
    simulator.run(n_episodes=settings.n_episodes, reset_observers=True)
    _ = welfare_observer.values


def main() -> None:
    for case_index, policy_kwargs in enumerate(PARAMETER_SETS):
        run_case(case_index=case_index, policy_kwargs=policy_kwargs)


if __name__ == "__main__":
    main()
