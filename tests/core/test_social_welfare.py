"""Tests for topology welfare helpers and observer integration."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.graph import create_empty_graph
from network_creation.core.model import (
    Action,
    ActionType,
    ActorScheduler,
)
from network_creation.core.observability import ObservationSampling
from network_creation.core.simulator import Simulator
from network_creation.observers.social_welfare_observer import (
    CenteredMeanWelfareObserver,
)
from network_creation.stats.topology_welfare import (
    get_ER_normalized_welfare,
    normalized_welfare,
    normalized_welfare_per_topology,
    total_welfare,
)
from tests.core.simulation_support import ConstantPolicy
from tests.graph_factories import graph_from_adjacency_matrix


class SocialWelfareTest(unittest.TestCase):
    def test_total_welfare_is_zero_on_empty_graph(self) -> None:
        graph = create_empty_graph(3)

        welfare = total_welfare(
            graph,
            benefit_to_cost_ratio=1.0,
            distance_discount=0.5,
        )

        self.assertAlmostEqual(welfare, 0.0)

    def test_total_welfare_counts_one_directed_link_in_raw_units(self) -> None:
        graph = graph_from_adjacency_matrix(
            np.array(
                [
                    [0, 1],
                    [0, 0],
                ],
                dtype=np.int8,
            )
        )

        welfare = total_welfare(
            graph,
            benefit_to_cost_ratio=2.0,
            distance_discount=0.5,
        )

        self.assertAlmostEqual(welfare, 1.0)

    def test_normalized_welfare_divides_total_by_number_of_agents(self) -> None:
        graph = graph_from_adjacency_matrix(
            np.array(
                [
                    [0, 1],
                    [0, 0],
                ],
                dtype=np.int8,
            )
        )

        welfare = normalized_welfare(
            graph,
            benefit_to_cost_ratio=2.0,
            distance_discount=0.5,
        )

        self.assertAlmostEqual(welfare, 0.5)

    def test_reference_topologies_match_closed_forms_and_er_argmax(self) -> None:
        n = 4
        distance_discount = 0.5
        benefit_to_cost_ratio = 1.0

        star_reference = normalized_welfare_per_topology(
            topology_class="star",
            n=n,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )
        empty_reference = normalized_welfare_per_topology(
            topology_class="empty",
            n=n,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )
        random_reference = normalized_welfare_per_topology(
            topology_class="random",
            n=n,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )
        max_reference = normalized_welfare_per_topology(
            topology_class="max",
            n=n,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )

        expected_star = ((n - 1) / n) * (
            2 * benefit_to_cost_ratio * distance_discount
            + (n - 2) * benefit_to_cost_ratio * distance_discount**2
            - 1
        )
        expected_random = max(
            get_ER_normalized_welfare(
                n=n,
                benefit_to_cost_ratio=benefit_to_cost_ratio,
                distance_discount=distance_discount,
                p_directed=candidate_p,
            )
            for candidate_p in np.linspace(0.0, 1.0, 1000, dtype=float)
        )

        self.assertAlmostEqual(star_reference, expected_star)
        self.assertAlmostEqual(empty_reference, 0.0)
        self.assertAlmostEqual(random_reference, expected_random)
        self.assertAlmostEqual(max_reference, max(empty_reference, star_reference))

    def test_reference_topologies_reject_invalid_name(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "topology_class must be one of 'star', 'empty', 'random', or 'max'",
        ):
            normalized_welfare_per_topology(
                topology_class="cycle",
                n=4,
                benefit_to_cost_ratio=1.0,
                distance_discount=0.5,
            )

    def test_centered_mean_welfare_observer_tracks_step_series(self) -> None:
        observer = CenteredMeanWelfareObserver(
            n_agents=2,
            observation_sampling=ObservationSampling.periodic(1),
            benefit_to_cost_ratio=2.0,
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
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
