"""Tests for social welfare statistics and observer integration."""

from __future__ import annotations

import unittest

import igraph as ig
import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.model import (
    Action,
    ActionSpace,
    ActionType,
    Observation,
    SchedulingMode,
)
from network_creation.core.simulator import Simulator
from network_creation.observers.social_welfare_observer import (
    SocialWelfareObserver,
    _compute_social_welfare,
)
from network_creation.policies.base import Policy


class ConstantPolicy(Policy):
    def __init__(self, action: Action) -> None:
        self.action = action

    def reset(self, seed: int) -> None:
        del seed

    def get_actions(
        self,
        observation: Observation,
        action_space: ActionSpace,
    ) -> list[Action]:
        del observation
        del action_space
        return [self.action]


class SocialWelfareTest(unittest.TestCase):
    def test_social_welfare_includes_self_terms_and_zero_for_disconnected_pairs(
        self,
    ) -> None:
        graph = ig.Graph(n=2)
        ownership_matrix = np.zeros((2, 2), dtype=np.int8)

        welfare = _compute_social_welfare(
            graph,
            ownership_matrix,
            link_cost=3.0,
            decay=0.5,
            per_link_benefit=10.0,
        )

        self.assertEqual(welfare, 20.0)

    def test_social_welfare_counts_bilateral_ownership_twice(self) -> None:
        graph = ig.Graph(edges=[(0, 1)], directed=False)
        graph.vs["name"] = [0, 1]
        ownership_matrix = np.array(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=np.int8,
        )

        welfare = _compute_social_welfare(
            graph,
            ownership_matrix,
            link_cost=3.0,
            decay=0.5,
            per_link_benefit=10.0,
        )

        self.assertEqual(welfare, 24.0)

    def test_social_welfare_observer_rejects_decay_outside_zero_to_one_interval(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "0 <= decay < 1"):
            SocialWelfareObserver(
                observation_frequency=1,
                link_cost=3.0,
                decay=1.0,
                per_link_benefit=10.0,
                aggregations=(),
            )

    def test_social_welfare_observer_tracks_step_series(self) -> None:
        observer = SocialWelfareObserver(
            observation_frequency=1,
            link_cost=3.0,
            decay=0.5,
            per_link_benefit=10.0,
            aggregations=(),
        )
        simulator = Simulator(
            env=Environment(np.zeros((2, 2), dtype=np.int8), seed=3),
            agents=(
                Agent(
                    id=0,
                    radius=0,
                    policy=ConstantPolicy(
                        Action(ActionType.BRIDGE, source=0, target=1)
                    ),
                    seed=5,
                ),
                Agent(
                    id=1,
                    radius=0,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=7,
                ),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=2,
            rng=np.random.default_rng(11),
            observers=(observer,),
        )

        simulator.run(n_episodes=1)

        np.testing.assert_array_equal(observer.steps, np.array([0, 1, 2]))
        np.testing.assert_allclose(
            observer.social_welfares,
            np.array([20.0, 27.0, 27.0], dtype=np.float32),
        )

    def test_social_welfare_observer_can_min_max_normalize_welfare(self) -> None:
        observer = SocialWelfareObserver(
            observation_frequency=1,
            link_cost=3.0,
            decay=0.5,
            per_link_benefit=10.0,
            max_welfare=30.0,
            baseline_welfare=20.0,
            aggregations=(),
        )
        simulator = Simulator(
            env=Environment(np.zeros((2, 2), dtype=np.int8), seed=3),
            agents=(
                Agent(
                    id=0,
                    radius=0,
                    policy=ConstantPolicy(
                        Action(ActionType.BRIDGE, source=0, target=1)
                    ),
                    seed=5,
                ),
                Agent(
                    id=1,
                    radius=0,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=7,
                ),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=2,
            rng=np.random.default_rng(11),
            observers=(observer,),
        )

        simulator.run(n_episodes=1)

        np.testing.assert_allclose(
            observer.social_welfares,
            np.array([0.0, 0.7, 0.7], dtype=np.float32),
        )

    def test_social_welfare_observer_rejects_partial_normalization_bounds(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "must be provided together",
        ):
            SocialWelfareObserver(
                observation_frequency=1,
                link_cost=3.0,
                decay=0.5,
                per_link_benefit=10.0,
                max_welfare=10.0,
                aggregations=(),
            )

    def test_social_welfare_observer_rejects_invalid_normalization_bounds(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "max_welfare must be greater than baseline_welfare",
        ):
            SocialWelfareObserver(
                observation_frequency=1,
                link_cost=3.0,
                decay=0.5,
                per_link_benefit=10.0,
                max_welfare=10.0,
                baseline_welfare=10.0,
                aggregations=(),
            )


if __name__ == "__main__":
    unittest.main()
