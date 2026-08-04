"""Tests for the in-memory final-hub ranking observer."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.graph import create_empty_graph
from network_creation.core.model import Action, ActionType, ActorScheduler
from network_creation.core.simulator import Simulator
from network_creation.observers.hub_ranking_observer import HubRankingObserver
from tests.core.simulation_support import ScriptedPolicy


class HubRankingObserverTest(unittest.TestCase):
    def test_records_distinct_orders_and_unique_hub(self) -> None:
        observer = HubRankingObserver(n_agents=3)
        simulator = Simulator(
            env=Environment(create_empty_graph(3), seed=3),
            agents=(
                Agent(
                    id=0,
                    visibility_distance=3,
                    policy=ScriptedPolicy(
                        [[Action(ActionType.BRIDGE, source=0, target=1)]]
                    ),
                    seed=5,
                    activation_weight=1.0,
                ),
                Agent(
                    id=1,
                    visibility_distance=3,
                    policy=ScriptedPolicy([[Action(ActionType.PASS, source=1)]]),
                    seed=7,
                    activation_weight=1.0,
                ),
                Agent(
                    id=2,
                    visibility_distance=3,
                    policy=ScriptedPolicy(
                        [[Action(ActionType.BRIDGE, source=2, target=1)]]
                    ),
                    seed=11,
                    activation_weight=1.0,
                ),
            ),
            actor_scheduler=ActorScheduler(
                seed=13,
                n_agents=3,
                n_actors=3,
                activation_weights=(1.0, 1.0, 1.0),
            ),
            max_steps=1,
            rng=np.random.default_rng(17),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=True)
        result = observer.get()

        np.testing.assert_array_equal(result.actor_order, [[0, 1, 2]])
        np.testing.assert_array_equal(result.recipient_order, [[1, -1, -1]])
        np.testing.assert_array_equal(result.final_hub_id, [1])

    def test_suppresses_repeats_and_marks_tied_hub(self) -> None:
        observer = HubRankingObserver(n_agents=2)
        simulator = Simulator(
            env=Environment(create_empty_graph(2), seed=19),
            agents=(
                Agent(
                    id=0,
                    visibility_distance=2,
                    policy=ScriptedPolicy(
                        [
                            [Action(ActionType.PASS, source=0)],
                            [Action(ActionType.BRIDGE, source=0, target=1)],
                        ]
                    ),
                    seed=23,
                    activation_weight=1.0,
                ),
                Agent(
                    id=1,
                    visibility_distance=2,
                    policy=ScriptedPolicy(
                        [
                            [Action(ActionType.PASS, source=1)],
                            [Action(ActionType.BRIDGE, source=1, target=0)],
                        ]
                    ),
                    seed=29,
                    activation_weight=1.0,
                ),
            ),
            actor_scheduler=ActorScheduler(
                seed=31,
                n_agents=2,
                n_actors=2,
                activation_weights=(1.0, 1.0),
            ),
            max_steps=2,
            rng=np.random.default_rng(37),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=True)
        result = observer.get()

        np.testing.assert_array_equal(result.actor_order, [[0, 1]])
        np.testing.assert_array_equal(result.recipient_order, [[1, 0]])
        np.testing.assert_array_equal(result.final_hub_id, [-1])


if __name__ == "__main__":
    unittest.main()
