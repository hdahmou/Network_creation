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
from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    ObservationSampling,
    StateView,
    StepEvent,
)
from network_creation.core.simulator import Simulator
from network_creation.observers.agent_payoff_observer import AgentPayoffObserver
from network_creation.observers.policy_welfare_observer import PolicyWelfareObserver
from network_creation.observers.social_welfare_observer import (
    CenteredMeanWelfareObserver,
)
from network_creation.stats.topology_welfare import (
    agent_payoff,
    centered_mean_welfare,
    centered_total_welfare,
    max_er_centered_mean_welfare,
    reference_centered_mean_welfare,
)
from tests.core.simulation_support import ConstantPolicy
from tests.graph_factories import graph_from_adjacency_matrix


def _record_agent_payoff_episode(
    observer: AgentPayoffObserver,
    final_graph,
    *,
    seed: int = 123,
    initial_graph=None,
) -> None:
    if initial_graph is None:
        initial_graph = create_empty_graph(final_graph.vcount())

    observer.on_episode_start(
        EpisodeStartEvent(
            episode_index=observer.n_episodes,
            seed=seed,
            max_steps=1,
            n_agents=final_graph.vcount(),
            initial_state=StateView(initial_graph),
        )
    )
    observer.on_step_end(
        StepEvent(
            step=0,
            decisions=(),
            modified=final_graph.ecount() != initial_graph.ecount(),
            pre_state=StateView(initial_graph),
            post_state=StateView(final_graph),
        )
    )
    observer.on_episode_end(
        EpisodeEndEvent(
            episode_index=observer.n_episodes,
            seed=seed,
            current_step=1,
            last_modified_step=0,
            final_state=StateView(final_graph),
        )
    )


class SocialWelfareTest(unittest.TestCase):
    def test_centered_total_welfare_is_zero_on_empty_graph(self) -> None:
        graph = create_empty_graph(3)

        welfare = centered_total_welfare(
            graph,
            benefit_to_cost_ratio=1.0,
            distance_discount=0.5,
        )

        self.assertAlmostEqual(welfare, 0.0)

    def test_centered_total_welfare_counts_one_directed_link(self) -> None:
        graph = graph_from_adjacency_matrix(
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=np.int8,
            )
        )

        welfare = centered_total_welfare(
            graph,
            benefit_to_cost_ratio=2.0,
            distance_discount=0.5,
        )

        self.assertAlmostEqual(welfare, 1.0)

    def test_agent_payoff_returns_normalized_payoff_per_agent(self) -> None:
        empty_graph = create_empty_graph(3)

        empty_values = agent_payoff(
            empty_graph,
            benefit_to_cost_ratio=2.0,
            distance_discount=0.5,
        )

        np.testing.assert_allclose(
            empty_values,
            np.zeros(3, dtype=np.float32),
        )
        self.assertEqual(empty_values.dtype, np.float32)

        graph = graph_from_adjacency_matrix(
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=np.int8,
            )
        )

        values = agent_payoff(
            graph,
            benefit_to_cost_ratio=2.0,
            distance_discount=0.5,
        )

        np.testing.assert_allclose(
            values,
            np.array([0.0, 0.5, 0.0], dtype=np.float32),
        )

    def test_centered_mean_welfare_divides_total_by_number_of_agents(self) -> None:
        graph = graph_from_adjacency_matrix(
            np.array(
                [
                    [0, 1],
                    [0, 0],
                ],
                dtype=np.int8,
            )
        )

        welfare = centered_mean_welfare(
            graph,
            benefit_to_cost_ratio=2.0,
            distance_discount=0.5,
        )

        self.assertAlmostEqual(welfare, 0.5)

    def test_welfare_references_match_closed_forms_and_er_argmax(self) -> None:
        n = 4
        distance_discount = 0.5
        benefit_to_cost_ratio = 1.0

        star_reference = reference_centered_mean_welfare(
            reference="star",
            n_agents=n,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )
        empty_reference = reference_centered_mean_welfare(
            reference="empty",
            n_agents=n,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )
        random_reference = reference_centered_mean_welfare(
            reference="random",
            n_agents=n,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )
        max_reference = reference_centered_mean_welfare(
            reference="max",
            n_agents=n,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )

        expected_star = ((n - 1) / n) * (
            2 * benefit_to_cost_ratio * distance_discount
            + (n - 2) * benefit_to_cost_ratio * distance_discount**2
            - 1
        )
        expected_random = max_er_centered_mean_welfare(
            n_agents=n,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )

        self.assertAlmostEqual(star_reference, expected_star)
        self.assertAlmostEqual(empty_reference, 0.0)
        self.assertAlmostEqual(random_reference, expected_random)
        self.assertAlmostEqual(max_reference, max(empty_reference, star_reference))

    def test_random_reference_resolves_optimum_below_uniform_grid_spacing(self) -> None:
        n = 4000
        random_reference = reference_centered_mean_welfare(
            reference="random",
            n_agents=n,
            benefit_to_cost_ratio=5.0 / n,
            distance_discount=0.9,
        )

        self.assertGreater(random_reference, 0.7)

    def test_welfare_references_reject_invalid_name(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "reference must be one of 'star', 'empty', 'random', or 'max'",
        ):
            reference_centered_mean_welfare(
                reference="cycle",  # type: ignore[arg-type]
                n_agents=4,
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
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(
                        Action(ActionType.BRIDGE, source=0, target=1)
                    ),
                    seed=5,
                ),
                Agent(
                    activation_weight=1.0,
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=7,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, n_actors=1, activation_weights=(1.0, 1.0), seed=21),
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

    def test_agent_payoff_observer_tracks_agent_series(self) -> None:
        observer = AgentPayoffObserver(
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
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(
                        Action(ActionType.BRIDGE, source=0, target=1)
                    ),
                    seed=5,
                ),
                Agent(
                    activation_weight=1.0,
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=7,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, n_actors=2, activation_weights=(1.0, 1.0), seed=21),
            max_steps=1,
            rng=np.random.default_rng(11),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(observer.steps, np.array([0, 1]))
        self.assertEqual(observer.values.shape, (2, 2))
        np.testing.assert_allclose(
            observer.values,
            np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        self.assertEqual(observer.y_label, "normalized payoff")

    def test_policy_welfare_observer_averages_payoff_by_assignment(self) -> None:
        observer = PolicyWelfareObserver(
            n_agents=3,
            observation_sampling=ObservationSampling.periodic(1),
            benefit_to_cost_ratio=2.0,
            distance_discount=0.5,
            aggregations=(),
            policy_ids=("first", "missing", "second"),
            policy_slot_by_agent=np.asarray([0, 2, 0], dtype=np.int32),
        )
        graph = graph_from_adjacency_matrix(
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=np.int8,
            )
        )

        observer.on_episode_start(
            EpisodeStartEvent(
                episode_index=0,
                seed=17,
                max_steps=0,
                n_agents=3,
                initial_state=StateView(graph),
            )
        )

        self.assertEqual(observer.values.shape, (1, 3))
        np.testing.assert_allclose(
            observer.values[:, [0, 2]],
            np.asarray([[0.0, 0.5]], dtype=np.float32),
        )
        self.assertTrue(np.isnan(observer.values[:, 1]).all())

    def test_agent_payoff_observer_pads_agent_slots(self) -> None:
        observer = AgentPayoffObserver(
            n_agents=2,
            observation_sampling=ObservationSampling.periodic(1),
            benefit_to_cost_ratio=2.0,
            distance_discount=0.5,
            aggregations=(),
            n_agent_slots=4,
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(2), seed=3),
            agents=(
                Agent(
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(
                        Action(ActionType.BRIDGE, source=0, target=1)
                    ),
                    seed=5,
                ),
                Agent(
                    activation_weight=1.0,
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=7,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, n_actors=2, activation_weights=(1.0, 1.0), seed=21),
            max_steps=1,
            rng=np.random.default_rng(11),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        self.assertEqual(observer.values.shape, (2, 4))
        np.testing.assert_allclose(
            observer.values[:, :2],
            np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        self.assertTrue(np.isnan(observer.values[:, 2:]).all())

    def test_agent_payoff_highest_degree_uses_final_graph_all_degree(self) -> None:
        observer = AgentPayoffObserver(
            n_agents=3,
            observation_sampling=ObservationSampling.periodic(1),
            benefit_to_cost_ratio=2.0,
            distance_discount=0.5,
            aggregations=(),
        )
        final_graph = graph_from_adjacency_matrix(
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                ],
                dtype=np.int8,
            )
        )

        _record_agent_payoff_episode(observer, final_graph)

        np.testing.assert_allclose(
            observer.degree_ranked_values("highest_degree"),
            np.array([0.0, 1.0], dtype=np.float32),
        )

    def test_agent_payoff_degree_ranked_roles_ignore_padded_slots(self) -> None:
        seed = 17
        observer = AgentPayoffObserver(
            n_agents=3,
            observation_sampling=ObservationSampling.periodic(1),
            benefit_to_cost_ratio=2.0,
            distance_discount=0.5,
            aggregations=(),
            n_agent_slots=5,
        )
        final_graph = graph_from_adjacency_matrix(
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                ],
                dtype=np.int8,
            )
        )
        rng = np.random.default_rng(seed)
        top_group = np.array([1], dtype=np.int_)
        rng.shuffle(top_group)
        tied_second_group = np.array([0, 2], dtype=np.int_)
        rng.shuffle(tied_second_group)
        expected_second = int(tied_second_group[0])
        expected_rest = int(tied_second_group[1])

        _record_agent_payoff_episode(observer, final_graph, seed=seed)

        np.testing.assert_allclose(
            observer.degree_ranked_values("second_highest_degree"),
            observer.values[:, expected_second],
        )
        np.testing.assert_allclose(
            observer.degree_ranked_values("rest_except_top_two"),
            observer.values[:, expected_rest],
        )

    def test_agent_payoff_tie_breaking_is_reproducible_from_episode_seed(
        self,
    ) -> None:
        seed = 29
        final_graph = graph_from_adjacency_matrix(
            np.array(
                [
                    [0, 1],
                    [0, 0],
                ],
                dtype=np.int8,
            )
        )
        observers = tuple(
            AgentPayoffObserver(
                n_agents=2,
                observation_sampling=ObservationSampling.periodic(1),
                benefit_to_cost_ratio=2.0,
                distance_discount=0.5,
                aggregations=(),
            )
            for _ in range(2)
        )

        for observer in observers:
            _record_agent_payoff_episode(observer, final_graph, seed=seed)

        np.testing.assert_array_equal(
            observers[0].degree_ranked_values("highest_degree"),
            observers[1].degree_ranked_values("highest_degree"),
        )

    def test_agent_payoff_missing_degree_ranked_roles_are_nan(self) -> None:
        observer = AgentPayoffObserver(
            n_agents=1,
            observation_sampling=ObservationSampling.periodic(1),
            benefit_to_cost_ratio=2.0,
            distance_discount=0.5,
            aggregations=(),
        )

        _record_agent_payoff_episode(observer, create_empty_graph(1))

        self.assertTrue(
            np.isnan(observer.degree_ranked_values("highest_degree")).all()
        )
        self.assertTrue(
            np.isnan(observer.degree_ranked_values("second_highest_degree")).all()
        )
        self.assertTrue(
            np.isnan(observer.degree_ranked_values("rest_except_top_two")).all()
        )


if __name__ == "__main__":
    unittest.main()
