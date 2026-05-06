"""Tests for topology-oriented observers."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.graph import create_empty_graph
from network_creation.core.model import Action, ActionType, ActorScheduler
from network_creation.core.observability import ObservationSampling
from network_creation.core.simulator import Simulator
from network_creation.observers.central_point_of_dominance_observer import (
    CentralPointOfDominanceObserver,
)
from network_creation.observers.density_observer import DensityObserver
from network_creation.observers.distance_observer import DistanceObserver
from network_creation.observers.global_closeness_centrality_observer import (
    GlobalClosenessCentralityObserver,
)
from network_creation.observers.hub_persistence_observer import (
    HubPersistenceObserver,
)
from network_creation.observers.indegree_observer import IndegreeObserver
from network_creation.observers.triad_census_observer import (
    TRIAD_TYPE_NAMES,
    TriadCensusObserver,
)
from tests.core.simulation_support import ConstantPolicy, EpisodeCallPolicy
from tests.graph_factories import graph_from_adjacency_matrix


class TopologyObserverTest(unittest.TestCase):
    def test_density_observer_tracks_density_only(self) -> None:
        density_observer = DensityObserver(
            n_agents=2,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(2), seed=13)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=17,
                ),
                Agent(
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=19,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, n_actors=3, seed=21),
            max_steps=2,
            rng=np.random.default_rng(23),
            observers=(density_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(density_observer.steps, np.array([0, 1, 2]))
        np.testing.assert_allclose(
            density_observer.values,
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
        )

    def test_density_reference_values_include_empty_and_star(self) -> None:
        self.assertEqual(
            DensityObserver.reference_values({"n_agents": 5}, {"choi2020": 0.4}),
            {"choi2020": 0.4, "empty": 0.0, "star": 0.4},
        )

    def test_indegree_observer_uses_directed_adjacency_matrix(self) -> None:
        indegree_observer = IndegreeObserver(
            n_agents=0,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(
            graph_from_adjacency_matrix(
                np.array(
                    [
                        [0, 1, 0],
                        [0, 0, 0],
                        [0, 1, 0],
                    ],
                    dtype=np.int8,
                )
            ),
            seed=17,
        )
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=0,
            rng=np.random.default_rng(19),
            observers=(indegree_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(indegree_observer.steps, np.array([0]))
        np.testing.assert_allclose(
            indegree_observer.values,
            np.array([2.0 / 3.0], dtype=np.float32),
        )

    def test_indegree_reference_values_include_empty_and_star(self) -> None:
        references = IndegreeObserver.reference_values(
            {"n_agents": 4},
            {"choi2020": 0.5},
        )

        self.assertEqual(
            references,
            {"choi2020": 0.5, "empty": 0.0, "star": 0.75},
        )

    def test_triad_reference_values_include_empty_and_star_by_dimension(self) -> None:
        self.assertEqual(
            TriadCensusObserver.reference_values(
                {"n_agents": 5, "dimension_name": "003"},
                {"choi2020": 0.6},
            ),
            {"choi2020": 0.6, "empty": 1.0, "star": 0.4},
        )
        self.assertEqual(
            TriadCensusObserver.reference_values(
                {"n_agents": 5, "dimension_name": "021U"}
            ),
            {"empty": 0.0, "star": 0.6},
        )
        self.assertEqual(
            TriadCensusObserver.reference_values(
                {"n_agents": 5, "dimension_name": "021D"}
            ),
            {"empty": 0.0, "star": 0.0},
        )

    def test_triad_census_observer_returns_zero_when_no_triads_exist(self) -> None:
        triad_observer = TriadCensusObserver(
            n_agents=0,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(2), seed=17)
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=0,
            rng=np.random.default_rng(19),
            observers=(triad_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_allclose(
            triad_observer.values,
            np.zeros((1, len(TRIAD_TYPE_NAMES)), dtype=np.float32),
        )

    def test_triad_census_observer_tracks_proportion_series(self) -> None:
        triad_observer = TriadCensusObserver(
            n_agents=1,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(3), seed=17)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=19,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=1,
            rng=np.random.default_rng(23),
            observers=(triad_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        expected = np.zeros((2, len(TRIAD_TYPE_NAMES)), dtype=np.float32)
        expected[0, TRIAD_TYPE_NAMES.index("003")] = 1.0
        expected[1, TRIAD_TYPE_NAMES.index("012")] = 1.0
        np.testing.assert_array_equal(triad_observer.steps, np.array([0, 1]))
        self.assertEqual(triad_observer.dimension_names, TRIAD_TYPE_NAMES)
        np.testing.assert_allclose(triad_observer.values, expected)

    def test_triad_census_observer_ignores_types_outside_configured_subset(self) -> None:
        triad_observer = TriadCensusObserver(
            n_agents=0,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(
            graph_from_adjacency_matrix(
                np.array(
                    [
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 0],
                    ],
                    dtype=np.int8,
                )
            ),
            seed=29,
        )
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=0,
            rng=np.random.default_rng(31),
            observers=(triad_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_allclose(
            triad_observer.values,
            np.zeros((1, len(TRIAD_TYPE_NAMES)), dtype=np.float32),
        )

    def test_triad_census_observer_reports_selected_type_proportions(self) -> None:
        triad_observer = TriadCensusObserver(
            n_agents=0,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(4), seed=31)
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=0,
            rng=np.random.default_rng(37),
            observers=(triad_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        expected = np.zeros((1, len(TRIAD_TYPE_NAMES)), dtype=np.float32)
        expected[0, TRIAD_TYPE_NAMES.index("003")] = 1.0
        np.testing.assert_allclose(triad_observer.values, expected)

    def test_triad_census_observer_aggregates_vector_outputs(self) -> None:
        triad_observer = TriadCensusObserver(
            n_agents=1,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=("mean", "std", "runs"),
        )
        env = Environment(create_empty_graph(3), seed=37)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=41,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=1,
            rng=np.random.default_rng(43),
            observers=(triad_observer,),
        )

        simulator.run(n_episodes=2, reset_observers=False)

        result = triad_observer.get()
        expected = np.zeros((2, len(TRIAD_TYPE_NAMES)), dtype=np.float32)
        expected[0, TRIAD_TYPE_NAMES.index("003")] = 1.0
        expected[1, TRIAD_TYPE_NAMES.index("012")] = 1.0

        self.assertEqual(result.n_episodes, 2)
        self.assertEqual(result.sample_shape, (len(TRIAD_TYPE_NAMES),))
        self.assertEqual(result.dimension_names, triad_observer.triad_type_names)
        np.testing.assert_array_equal(result.steps, np.array([0, 1]))
        np.testing.assert_allclose(result.outputs["mean"], expected)
        np.testing.assert_allclose(
            result.outputs["std"],
            np.zeros((2, len(TRIAD_TYPE_NAMES)), dtype=np.float32),
        )
        runs = result.outputs["runs"]
        self.assertEqual(len(runs), 2)
        np.testing.assert_allclose(runs[0], expected)

    def test_distance_observer_uses_giant_component_and_full_graph_scale(self) -> None:
        distance_observer = DistanceObserver(
            n_agents=0,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(
            graph_from_adjacency_matrix(
                np.array(
                    [
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    dtype=np.int8,
                )
            ),
            seed=23,
        )
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=0,
            rng=np.random.default_rng(29),
            observers=(distance_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(distance_observer.steps, np.array([0]))
        np.testing.assert_allclose(
            distance_observer.values,
            np.array([1.0], dtype=np.float32),
        )

    def test_distance_reference_values_include_empty_and_star(self) -> None:
        self.assertEqual(
            DistanceObserver.reference_values({"n_agents": 5}, {"choi2020": 1.5}),
            {"choi2020": 1.5, "empty": 0.0, "star": 1.6},
        )

    def test_global_closeness_centrality_observer_uses_full_graph_harmonic_scores(
        self,
    ) -> None:
        closeness_observer = GlobalClosenessCentralityObserver(
            n_agents=0,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(
            graph_from_adjacency_matrix(
                np.array(
                    [
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    dtype=np.int8,
                )
            ),
            seed=31,
        )
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=0,
            rng=np.random.default_rng(37),
            observers=(closeness_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(closeness_observer.steps, np.array([0]))
        np.testing.assert_allclose(
            closeness_observer.values,
            np.array([2.0 / 3.0], dtype=np.float32),
        )

    def test_global_closeness_reference_values_include_empty_and_star(self) -> None:
        self.assertEqual(
            GlobalClosenessCentralityObserver.reference_values(
                {"n_agents": 5},
                {"choi2020": 0.7},
            ),
            {"choi2020": 0.7, "empty": 0.0, "star": 1.0},
        )

    def test_central_point_of_dominance_observer_matches_normalized_betweenness(
        self,
    ) -> None:
        dominance_observer = CentralPointOfDominanceObserver(
            n_agents=0,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(
            graph_from_adjacency_matrix(
                np.array(
                    [
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                    ],
                    dtype=np.int8,
                )
            ),
            seed=39,
        )
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=0,
            rng=np.random.default_rng(41),
            observers=(dominance_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(dominance_observer.steps, np.array([0]))
        np.testing.assert_allclose(
            dominance_observer.values,
            np.array([4.0 / 9.0], dtype=np.float32),
        )

    def test_central_point_reference_values_include_empty_and_star(self) -> None:
        self.assertEqual(
            CentralPointOfDominanceObserver.reference_values(
                {"n_agents": 5},
                {"choi2020": 0.8},
            ),
            {"choi2020": 0.8, "empty": 0.0, "star": 1.0},
        )

    def test_distance_observer_aggregates_mean_and_runs(self) -> None:
        distance_observer = DistanceObserver(
            n_agents=0,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=("mean", "runs"),
        )
        env = Environment(
            graph_from_adjacency_matrix(
                np.array(
                    [
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    dtype=np.int8,
                )
            ),
            seed=79,
        )
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=0,
            rng=np.random.default_rng(83),
            observers=(distance_observer,),
        )

        simulator.run(n_episodes=2, reset_observers=False)

        result = distance_observer.get()

        self.assertEqual(result.n_episodes, 2)
        np.testing.assert_array_equal(result.steps, np.array([0]))
        self.assertEqual(set(result.outputs), {"mean", "runs"})
        np.testing.assert_allclose(
            result.outputs["mean"],
            np.array([1.0], dtype=np.float32),
        )
        runs = result.outputs["runs"]
        self.assertEqual(len(runs), 2)
        np.testing.assert_allclose(runs[0], np.array([1.0], dtype=np.float32))

    def test_central_point_of_dominance_observer_aggregates_mean_and_runs(self) -> None:
        dominance_observer = CentralPointOfDominanceObserver(
            n_agents=0,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=("mean", "runs"),
        )
        env = Environment(
            graph_from_adjacency_matrix(
                np.array(
                    [
                        [0, 1, 1, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    dtype=np.int8,
                )
            ),
            seed=89,
        )
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=0,
            rng=np.random.default_rng(97),
            observers=(dominance_observer,),
        )

        simulator.run(n_episodes=2, reset_observers=False)

        result = dominance_observer.get()

        self.assertEqual(result.n_episodes, 2)
        np.testing.assert_array_equal(result.steps, np.array([0]))
        self.assertEqual(set(result.outputs), {"mean", "runs"})
        np.testing.assert_allclose(
            result.outputs["mean"],
            np.array([1.0], dtype=np.float32),
        )
        runs = result.outputs["runs"]
        self.assertEqual(len(runs), 2)
        np.testing.assert_allclose(runs[0], np.array([1.0], dtype=np.float32))

    def test_hub_persistence_observer_keeps_tied_hubs_and_uses_overlap_fraction(
        self,
    ) -> None:
        hub_persistence_observer = HubPersistenceObserver(
            n_agents=3,
            observation_sampling=ObservationSampling.periodic(1),
            window_size=1,
            aggregations=(),
        )
        env = Environment(create_empty_graph(3), seed=211)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=EpisodeCallPolicy(
                        [[
                            [Action(ActionType.ADD, source=0, target=1)],
                            [Action(ActionType.PASS, source=0)],
                        ]]
                    ),
                    seed=223,
                ),
                Agent(
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=227,
                ),
                Agent(
                    id=2,
                    visibility_distance=10,
                    policy=EpisodeCallPolicy(
                        [[
                            [Action(ActionType.ADD, source=2, target=1)],
                            [Action(ActionType.PASS, source=2)],
                        ]]
                    ),
                    seed=229,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, n_actors=3, seed=21),
            max_steps=2,
            rng=np.random.default_rng(233),
            observers=(hub_persistence_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(hub_persistence_observer.steps, np.arange(3))
        np.testing.assert_allclose(
            hub_persistence_observer.values,
            np.array([1.0 / 3.0, 1.0, np.nan], dtype=np.float32),
            equal_nan=True,
        )

    def test_hub_persistence_observer_uses_nan_borders_for_adjacent_ten_step_windows(
        self,
    ) -> None:
        hub_persistence_observer = HubPersistenceObserver(
            n_agents=3,
            observation_sampling=ObservationSampling.periodic(1),
            window_size=10,
            aggregations=(),
        )
        env = Environment(create_empty_graph(3), seed=239)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=241,
                ),
                Agent(
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=251,
                ),
                Agent(
                    id=2,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=2)),
                    seed=257,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=19,
            rng=np.random.default_rng(241),
            observers=(hub_persistence_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(hub_persistence_observer.steps, np.arange(20))
        np.testing.assert_allclose(
            hub_persistence_observer.values,
            np.array([np.nan] * 9 + [1.0] + [np.nan] * 10, dtype=np.float32),
            equal_nan=True,
        )

    def test_hub_persistence_observer_aggregates_series_outputs(self) -> None:
        hub_persistence_observer = HubPersistenceObserver(
            n_agents=3,
            observation_sampling=ObservationSampling.periodic(1),
            window_size=1,
            aggregations=("mean", "std", "runs"),
        )
        env = Environment(create_empty_graph(3), seed=251)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=EpisodeCallPolicy(
                        [
                            [
                                [Action(ActionType.ADD, source=0, target=1)],
                                [Action(ActionType.PASS, source=0)],
                            ],
                            [
                                [Action(ActionType.PASS, source=0)],
                                [Action(ActionType.PASS, source=0)],
                            ],
                        ]
                    ),
                    seed=257,
                ),
                Agent(
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=263,
                ),
                Agent(
                    id=2,
                    visibility_distance=10,
                    policy=EpisodeCallPolicy(
                        [
                            [
                                [Action(ActionType.ADD, source=2, target=1)],
                                [Action(ActionType.PASS, source=2)],
                            ],
                            [
                                [Action(ActionType.PASS, source=2)],
                                [Action(ActionType.PASS, source=2)],
                            ],
                        ]
                    ),
                    seed=269,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, n_actors=3, seed=21),
            max_steps=2,
            rng=np.random.default_rng(271),
            observers=(hub_persistence_observer,),
        )

        simulator.run(n_episodes=2, reset_observers=False)

        result = hub_persistence_observer.get()

        np.testing.assert_array_equal(result.steps, np.arange(3))
        self.assertEqual(result.n_episodes, 2)
        self.assertEqual(set(result.outputs), {"mean", "std", "runs"})
        np.testing.assert_allclose(
            result.outputs["mean"],
            np.array([2.0 / 3.0, 1.0, np.nan], dtype=np.float32),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            result.outputs["std"],
            np.array([1.0 / 3.0, 0.0, np.nan], dtype=np.float32),
            atol=1e-7,
            equal_nan=True,
        )
        runs = result.outputs["runs"]
        self.assertEqual(len(runs), 2)
        np.testing.assert_allclose(
            runs[0],
            np.array([1.0 / 3.0, 1.0, np.nan], dtype=np.float32),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            runs[1],
            np.array([1.0, 1.0, np.nan], dtype=np.float32),
            equal_nan=True,
        )

    def test_hub_persistence_observer_terminal_sampling_returns_latest_valid_score(
        self,
    ) -> None:
        hub_persistence_observer = HubPersistenceObserver(
            n_agents=3,
            observation_sampling=ObservationSampling.terminal(),
            window_size=1,
            aggregations=(),
        )
        env = Environment(create_empty_graph(3), seed=211)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=EpisodeCallPolicy(
                        [[
                            [Action(ActionType.ADD, source=0, target=1)],
                            [Action(ActionType.PASS, source=0)],
                        ]]
                    ),
                    seed=223,
                ),
                Agent(
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=227,
                ),
                Agent(
                    id=2,
                    visibility_distance=10,
                    policy=EpisodeCallPolicy(
                        [[
                            [Action(ActionType.ADD, source=2, target=1)],
                            [Action(ActionType.PASS, source=2)],
                        ]]
                    ),
                    seed=229,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, n_actors=3, seed=21),
            max_steps=2,
            rng=np.random.default_rng(233),
            observers=(hub_persistence_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(hub_persistence_observer.steps, np.array([1]))
        np.testing.assert_allclose(
            hub_persistence_observer.values,
            np.array([1.0], dtype=np.float32),
        )

    def test_hub_persistence_observer_terminal_sampling_falls_back_to_nan_at_step_zero(
        self,
    ) -> None:
        hub_persistence_observer = HubPersistenceObserver(
            n_agents=0,
            observation_sampling=ObservationSampling.terminal(),
            window_size=10,
            aggregations=(),
        )
        env = Environment(create_empty_graph(3), seed=239)
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=0,
            rng=np.random.default_rng(241),
            observers=(hub_persistence_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(hub_persistence_observer.steps, np.array([0]))
        np.testing.assert_allclose(
            hub_persistence_observer.values,
            np.array([np.nan], dtype=np.float32),
            equal_nan=True,
        )

    def test_density_observer_terminal_sampling_returns_last_step_value(self) -> None:
        density_observer = DensityObserver(
            n_agents=2,
            observation_sampling=ObservationSampling.terminal(),
            aggregations=(),
        )
        env = Environment(create_empty_graph(2), seed=43)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=47,
                ),
                Agent(
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=53,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=2,
            rng=np.random.default_rng(59),
            observers=(density_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(density_observer.steps, np.array([2]))
        np.testing.assert_allclose(
            density_observer.values,
            np.array([1.0], dtype=np.float32),
        )

    def test_series_observer_terminal_sampling_reuses_initial_sample_when_no_step_executes(
        self,
    ) -> None:
        density_observer = DensityObserver(
            n_agents=0,
            observation_sampling=ObservationSampling.terminal(),
            aggregations=(),
        )
        env = Environment(create_empty_graph(2), seed=61)
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=0,
            rng=np.random.default_rng(67),
            observers=(density_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(density_observer.steps, np.array([0]))
        np.testing.assert_allclose(
            density_observer.values,
            np.array([0.0], dtype=np.float32),
        )

    def test_density_observer_aggregates_selected_outputs_and_runs(self) -> None:
        density_observer = DensityObserver(
            n_agents=2,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=("mean", "runs"),
        )
        env = Environment(create_empty_graph(2), seed=61)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=67,
                ),
                Agent(
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=71,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=2,
            rng=np.random.default_rng(73),
            observers=(density_observer,),
        )

        simulator.run(n_episodes=2, reset_observers=False)

        result = density_observer.get()

        self.assertEqual(result.n_episodes, 2)
        np.testing.assert_array_equal(result.steps, np.array([0, 1, 2]))
        self.assertEqual(set(result.outputs), {"mean", "runs"})
        np.testing.assert_allclose(
            result.outputs["mean"],
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
        )
        runs = result.outputs["runs"]
        self.assertEqual(len(runs), 2)
        np.testing.assert_allclose(
            runs[0],
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
        )

    def test_density_observer_get_keeps_sampled_steps_for_aggregated_series(self) -> None:
        density_observer = DensityObserver(
            n_agents=2,
            observation_sampling=ObservationSampling.periodic(2),
            aggregations=("mean", "runs"),
        )
        env = Environment(create_empty_graph(2), seed=101)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=103,
                ),
                Agent(
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=107,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=5,
            rng=np.random.default_rng(109),
            observers=(density_observer,),
        )

        simulator.run(n_episodes=2, reset_observers=False)

        result = density_observer.get()

        np.testing.assert_array_equal(result.steps, np.array([0, 2, 4]))
        np.testing.assert_allclose(
            result.outputs["mean"],
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_density_observer_can_request_std_without_mean(self) -> None:
        density_observer = DensityObserver(
            n_agents=2,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=("std",),
        )
        env = Environment(create_empty_graph(2), seed=79)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=83,
                ),
                Agent(
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=89,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=2,
            rng=np.random.default_rng(97),
            observers=(density_observer,),
        )

        simulator.run(n_episodes=2, reset_observers=False)

        result = density_observer.get()

        self.assertEqual(set(result.outputs), {"std"})
        np.testing.assert_allclose(
            result.outputs["std"],
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_triad_census_observer_terminal_sampling_returns_last_tensor_value(self) -> None:
        triad_observer = TriadCensusObserver(
            n_agents=1,
            observation_sampling=ObservationSampling.terminal(),
            aggregations=(),
        )
        env = Environment(create_empty_graph(3), seed=17)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=19,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=1,
            rng=np.random.default_rng(23),
            observers=(triad_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        expected = np.zeros((1, len(TRIAD_TYPE_NAMES)), dtype=np.float32)
        expected[0, TRIAD_TYPE_NAMES.index("012")] = 1.0
        np.testing.assert_array_equal(triad_observer.steps, np.array([1]))
        np.testing.assert_allclose(triad_observer.values, expected)


if __name__ == "__main__":
    unittest.main()
