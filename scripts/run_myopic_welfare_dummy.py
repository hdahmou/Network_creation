#!/usr/bin/env python3
"""Silent fixed-workload script for profiling the myopic policy."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from network_creation.core.agent import Agent
from network_creation.core.model import SchedulingMode
from network_creation.core.runtime import (
    SimulationSettings,
    create_simulation_runtime,
)
from network_creation.observers.social_welfare_observer import SocialWelfareObserver
from network_creation.policies.myopic_payoff import MyopicPayoffPolicy


N_AGENTS = 50
MAX_STEPS = 360
N_EPISODES = 32
PARAMETER_SETS = (
    {
        "link_cost": float(N_AGENTS * 2),
        "decay": 0.9,
        "per_link_benefit": 10.0,
        "bridge_probability": 0.4,
    },
    {
        "link_cost": float(N_AGENTS * 2),
        "decay": 0.7,
        "per_link_benefit": 10.0,
        "bridge_probability": 0.2,
    },
)


def run_case(case_index: int, policy_kwargs: dict[str, float]) -> None:
    def build_agent(node_id: int) -> Agent:
        policy_seed = 1_000 * (case_index + 1) + node_id
        return Agent(
            id=node_id,
            radius=N_AGENTS + 1,
            policy=MyopicPayoffPolicy(
                seed=policy_seed,
                link_cost=policy_kwargs["link_cost"],
                decay=policy_kwargs["decay"],
                per_link_benefit=policy_kwargs["per_link_benefit"],
                bridge_probability=policy_kwargs["bridge_probability"],
            ),
            seed=policy_seed,
        )

    settings = SimulationSettings(
        n_agents=N_AGENTS,
        mode=SchedulingMode.SIMULTANEOUS,
        seed=100 + case_index,
        max_steps=MAX_STEPS,
        n_episodes=N_EPISODES,
        stagnation_patience=None,
    )
    welfare_observer = SocialWelfareObserver(
        observation_frequency=1,
        link_cost=policy_kwargs["link_cost"],
        decay=policy_kwargs["decay"],
        per_link_benefit=policy_kwargs["per_link_benefit"],
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
