#!/usr/bin/env python3
"""Silent fixed-workload script for profiling the best-response policy."""

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
from network_creation.policies.best_response import BestResponsePolicy


N_AGENTS = 50
MAX_STEPS = 360
N_EPISODES = 32
PARAMETER_SETS = (
    {
        "benefit_to_cost_ratio": 0.5,
        "distance_discount": 0.9,
        "bridge_weight": 0.0,
        "pass_bias": 0.5,
        "exploration": 0.0,
        "opportunity_sensitivity": 0.5,
    },
    {
        "benefit_to_cost_ratio": 0.5,
        "distance_discount": 0.7,
        "bridge_weight": 0.5,
        "pass_bias": 0.525,
        "exploration": 0.0,
        "opportunity_sensitivity": 0.5,
    },
)


def run_case(case_index: int, policy_kwargs: dict[str, float]) -> None:
    def build_agent(node_id: int) -> Agent:
        policy_seed = 1_000 * (case_index + 1) + node_id
        return Agent(
            id=node_id,
            visibility_distance=N_AGENTS,
            policy=BestResponsePolicy(
                n_nodes=N_AGENTS,
                seed=policy_seed,
                benefit_to_cost_ratio=policy_kwargs["benefit_to_cost_ratio"],
                distance_discount=policy_kwargs["distance_discount"],
                bridge_weight=policy_kwargs["bridge_weight"],
                pass_bias=policy_kwargs["pass_bias"],
                exploration=policy_kwargs["exploration"],
                opportunity_sensitivity=policy_kwargs["opportunity_sensitivity"],
            ),
            seed=policy_seed,
        )

    settings = SimulationSettings(
        n_agents=N_AGENTS,
        n_actors=2,
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
    print("Running best-response welfare dummy script...")
    print(f"Number of agents: {N_AGENTS}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Number of episodes: {N_EPISODES}")
    for case_index, policy_kwargs in enumerate(PARAMETER_SETS):
        run_case(case_index=case_index, policy_kwargs=policy_kwargs)


if __name__ == "__main__":
    main()
