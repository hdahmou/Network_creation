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
    ActorScheduler,
)
from network_creation.core.observability import ObservationSampling
from network_creation.core.simulator import Simulator
from network_creation.observers.social_welfare_observer import (
    NormalizedWelfareObserver,
    _compute_normalized_welfare,
)
from network_creation.policies.base import Policy
from network_creation.stats.er_payoff import expected_aggregate_payoff
from tests.graph_factories import graph_from_adjacency_matrix


def _local_action(observation: Observation, action: Action) -> Action:
    return Action(
        action.type,
        observation.local_from_global(action.source),
        None
        if action.target is None
        else observation.local_from_global(action.target),
    )


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
        del action_space
        return PolicyDecision(
            actions=(_local_action(observation, self.action),),
            value=None,
        )


class SocialWelfareTest(unittest.TestCase):
    def test_reference_values_return_closed_form_star_and_empty_baselines(self) -> None:
        n = 4
        distance_discount = 0.5
        benefit_to_cost_ratio = 0.3
        reference_values = NormalizedWelfareObserver.reference_values(
            n=n,
            distance_discount=distance_discount,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
        )
        expected_star = (
            benefit_to_cost_ratio
            * (
                n
                + 2 * (n - 1) * distance_discount
                + (n - 1) * (n - 2) * distance_discount**2
            )
            + (n - 1) ** 2
        ) / (n * (2 * n - 1))
        expected_empty = (benefit_to_cost_ratio + n - 1) / (2 * n - 1)

        self.assertEqual(set(reference_values), {"star", "empty", "random"})
        self.assertAlmostEqual(reference_values["star"], expected_star)
        self.assertAlmostEqual(reference_values["empty"], expected_empty)

    def test_reference_values_random_matches_er_expected_payoff_argmax(self) -> None:
        expected_random_reference = expected_aggregate_payoff(
            4,
            0.3,
            p=None,
            delta=0.5,
        )
        assert isinstance(expected_random_reference, tuple)

        reference_values = NormalizedWelfareObserver.reference_values(
            n=4,
            distance_discount=0.5,
            benefit_to_cost_ratio=0.3,
        )

        self.assertAlmostEqual(
            reference_values["random"], expected_random_reference[0]
        )

    def test_reference_values_reject_invalid_n(self) -> None:
        with self.assertRaisesRegex(ValueError, "n must satisfy n >= 1"):
            NormalizedWelfareObserver.reference_values(
                n=0,
                distance_discount=0.5,
                benefit_to_cost_ratio=0.3,
            )

    def test_reference_values_reject_invalid_ratio(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "benefit_to_cost_ratio must satisfy 0 <= "
            "benefit_to_cost_ratio <= 1",
        ):
            NormalizedWelfareObserver.reference_values(
                n=4,
                distance_discount=0.5,
                benefit_to_cost_ratio=1.1,
            )

    def test_reference_values_reject_invalid_discount(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "distance_discount must satisfy 0 <= distance_discount < 1"
        ):
            NormalizedWelfareObserver.reference_values(
                n=4,
                distance_discount=1.0,
                benefit_to_cost_ratio=0.3,
            )

    def test_normalized_welfare_keeps_empty_graph_in_bounded_positive_range(
        self,
    ) -> None:
        graph = create_empty_graph(2)

        welfare = _compute_normalized_welfare(
            graph,
            benefit_to_cost_ratio=0.3,
            distance_discount=0.5,
        )

        self.assertAlmostEqual(welfare, 13.0 / 30.0)

    def test_normalized_welfare_counts_bilateral_links_on_zero_to_one_scale(self) -> None:
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

        self.assertAlmostEqual(welfare, 0.15)

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
                    visibility_distance=10,
                    policy=ConstantPolicy(
                        Action(ActionType.BRIDGE, source=0, target=1)
                    ),
                    seed=5,
                ),
                Agent(
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=7,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, n_actors=1, seed=21),
            max_steps=2,
            rng=np.random.default_rng(11),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(observer.steps, np.array([0, 1, 2]))
        np.testing.assert_allclose(
            observer.values,
            np.array([13.0 / 30.0, 19.0 / 60.0, 19.0 / 60.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
