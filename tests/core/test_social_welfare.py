"""Tests for normalized welfare statistics and observer integration."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.graph import create_empty_graph
from network_creation.core.model import (
    Action,
    ActionSpace,
    ActionType,
    Observation,
    PolicyDecision,
    SchedulingMode,
)
from network_creation.core.observability import ObservationSampling
from network_creation.core.simulator import Simulator
from network_creation.observers.social_welfare_observer import (
    NormalizedWelfareObserver,
    _compute_normalized_welfare,
)
from network_creation.policies.base import Policy
from tests.graph_factories import graph_from_adjacency_matrix


class ConstantPolicy(Policy):
    def __init__(self, action: Action) -> None:
        self.action = action

    def reset(self, seed: int) -> None:
        del seed

    def get_decision(
        self,
        observation: Observation,
        action_space: ActionSpace,
    ) -> PolicyDecision:
        del observation
        del action_space
        return PolicyDecision(actions=(self.action,), value=None)


class SocialWelfareTest(unittest.TestCase):
    def test_normalized_welfare_excludes_self_terms_and_zeroes_disconnected_pairs(
        self,
    ) -> None:
        graph = create_empty_graph(2)

        welfare = _compute_normalized_welfare(
            graph,
            benefit_to_cost_ratio=0.3,
            distance_discount=0.5,
        )

        self.assertEqual(welfare, 0.0)

    def test_normalized_welfare_counts_bilateral_links_in_node_means(self) -> None:
        graph = graph_from_adjacency_matrix(
            np.array(
                [
                    [0, 1],
                    [1, 0],
                ],
                dtype=np.int8,
            )
        )

        welfare = _compute_normalized_welfare(
            graph,
            benefit_to_cost_ratio=0.3,
            distance_discount=0.5,
        )

        self.assertEqual(welfare, -0.85)

    def test_normalized_welfare_observer_rejects_decay_outside_zero_to_one_interval(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError, "0 <= distance_discount < 1"
        ):
            NormalizedWelfareObserver(
                observation_sampling=ObservationSampling.periodic(1),
                benefit_to_cost_ratio=0.3,
                distance_discount=1.0,
                aggregations=(),
            )

    def test_normalized_welfare_observer_rejects_invalid_ratio(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "benefit_to_cost_ratio must satisfy 0 <= "
            "benefit_to_cost_ratio <= 1",
        ):
            NormalizedWelfareObserver(
                observation_sampling=ObservationSampling.periodic(1),
                benefit_to_cost_ratio=1.1,
                distance_discount=0.5,
                aggregations=(),
            )

    def test_normalized_welfare_observer_tracks_step_series(self) -> None:
        observer = NormalizedWelfareObserver(
            observation_sampling=ObservationSampling.periodic(1),
            benefit_to_cost_ratio=0.3,
            distance_discount=0.5,
            aggregations=(),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(2), seed=3),
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1,
                    policy=ConstantPolicy(
                        Action(ActionType.BRIDGE, source=0, target=1)
                    ),
                    seed=5,
                ),
                Agent(
                    id=1,
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=7,
                ),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=2,
            rng=np.random.default_rng(11),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(observer.steps, np.array([0, 1, 2]))
        np.testing.assert_allclose(
            observer.values,
            np.array([0.0, -0.35, -0.35], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
