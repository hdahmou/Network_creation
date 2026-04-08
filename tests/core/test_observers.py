"""Tests for concrete simulator observers."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import numpy as np
from numpy.typing import NDArray

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.graph import create_empty_graph
from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    ObservationSampling,
    StateView,
    StepEvent,
)
from network_creation.core.model import (
    Action,
    ActionSpace,
    ActionType,
    Observation,
    PolicyDecision,
    ActorScheduler,
)
from network_creation.core.simulator import Simulator
from network_creation.observers.action_count_observer import (
    ActionCountObserver,
)
from network_creation.observers.action_print_observer import ActionPrintObserver
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
from network_creation.observers.grimoire import OBSERVER_GRIMOIRE
from network_creation.observers.state_history_observer import StateHistoryObserver
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)
from network_creation.observers.statistics.episode_tensor_series import (
    EpisodeSeriesTensorStatisticObserver,
)
from network_creation.observers.triad_census_observer import (
    TRIAD_TYPE_NAMES,
    TriadCensusObserver,
)
from network_creation.policies.base import Policy
from tests.graph_factories import graph_from_adjacency_matrix


class ConstantPolicy(Policy):
    def __init__(self, action: Action, value: dict[str, object] | None = None) -> None:
        self.action = action
        self.value = value

    def reset(self, seed: int) -> None:
        del seed
        return None

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        del observation
        del action_space
        return PolicyDecision(actions=(self.action,), value=self.value)


class ActionListPolicy(Policy):
    def __init__(self, actions: list[Action]) -> None:
        self.actions = actions

    def reset(self, seed: int) -> None:
        del seed
        return None

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        del observation
        del action_space
        return PolicyDecision(actions=tuple(self.actions), value=None)


class EpisodeCallPolicy(Policy):
    def __init__(self, episode_actions: list[list[list[Action]]]) -> None:
        self.episode_actions = episode_actions
        self._episode_index = -1
        self._call_index = 0

    def reset(self, seed: int) -> None:
        del seed
        self._episode_index = min(
            self._episode_index + 1,
            len(self.episode_actions) - 1,
        )
        self._call_index = 0

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        del observation
        del action_space
        current_episode_actions = self.episode_actions[self._episode_index]
        action_idx = min(self._call_index, len(current_episode_actions) - 1)
        self._call_index += 1
        return PolicyDecision(
            actions=tuple(current_episode_actions[action_idx]),
            value=None,
        )


class TensorMatrixObserver(EpisodeSeriesTensorStatisticObserver):
    label = "Tensor Matrix"
    x_label = "step"
    y_label = "matrix value"
    y_min = None
    y_max = None
    dimension_names = ("a", "b", "c", "d")

    def compute_initial_value(self, event: EpisodeStartEvent) -> NDArray[np.float32]:
        del event
        return np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    def compute_step_value(self, event: StepEvent) -> NDArray[np.float32]:
        base = 10.0 * (event.step + 1)
        return np.array(
            [[base, base + 1.0], [base + 2.0, base + 3.0]],
            dtype=np.float32,
        )


class ShapeMismatchObserver(EpisodeSeriesStatisticObserver):
    label = "Shape Mismatch"
    x_label = "step"
    y_label = "value"
    y_min = None
    y_max = None

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        del event
        return 0.0

    def compute_step_value(self, event: StepEvent) -> float:
        del event
        return 1.0


class TensorShapeMismatchObserver(EpisodeSeriesTensorStatisticObserver):
    label = "Shape Mismatch"
    x_label = "step"
    y_label = "value"
    y_min = None
    y_max = None
    dimension_names = ("a", "b")

    def compute_initial_value(self, event: EpisodeStartEvent) -> NDArray[np.float32]:
        del event
        return np.array([0.0, 1.0], dtype=np.float32)

    def compute_step_value(self, event: StepEvent) -> NDArray[np.float32]:
        del event
        return np.array([0.0, 1.0, 2.0], dtype=np.float32)


class TensorScalarValueObserver(EpisodeSeriesTensorStatisticObserver):
    label = "Scalar Tensor"
    x_label = "step"
    y_label = "value"
    y_min = None
    y_max = None
    dimension_names = ("a",)

    def compute_initial_value(self, event: EpisodeStartEvent) -> NDArray[np.float32]:
        del event
        return np.array(1.0, dtype=np.float32)

    def compute_step_value(self, event: StepEvent) -> NDArray[np.float32]:
        del event
        return np.array(1.0, dtype=np.float32)


class TensorWrongDimensionNamesObserver(EpisodeSeriesTensorStatisticObserver):
    label = "Wrong Names"
    x_label = "step"
    y_label = "value"
    y_min = None
    y_max = None
    dimension_names = ("a",)

    def compute_initial_value(self, event: EpisodeStartEvent) -> NDArray[np.float32]:
        del event
        return np.array([0.0, 1.0], dtype=np.float32)

    def compute_step_value(self, event: StepEvent) -> NDArray[np.float32]:
        del event
        return np.array([0.0, 1.0], dtype=np.float32)


def _graph_signature(graph) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    return (
        tuple(int(node_id) for node_id in graph.vs["name"]),
        tuple(sorted((int(source), int(target)) for source, target in graph.get_edgelist())),
    )


def _build_contract_test_observer(name: str):
    observer_class = OBSERVER_GRIMOIRE[name]
    periodic = ObservationSampling.periodic(1)
    kwargs_by_name = {
        "action_count": {"observation_sampling": periodic, "aggregations": ()},
        "action_print": {"observation_sampling": periodic, "print_empty_steps": False},
        "central_point_of_dominance": {
            "observation_sampling": periodic,
            "aggregations": (),
        },
        "density": {"observation_sampling": periodic, "aggregations": ()},
        "distance": {"observation_sampling": periodic, "aggregations": ()},
        "global_closeness_centrality": {
            "observation_sampling": periodic,
            "aggregations": (),
        },
        "hub_persistence": {
            "observation_sampling": periodic,
            "window_size": 1,
            "aggregations": (),
        },
        "indegree": {"observation_sampling": periodic, "aggregations": ()},
        "normalized_welfare": {
            "observation_sampling": periodic,
            "benefit_to_cost_ratio": 0.1,
            "distance_discount": 0.5,
            "aggregations": (),
        },
        "state_history": {"observation_sampling": periodic, "keep_runs": False},
        "triad_census": {"observation_sampling": periodic, "aggregations": ()},
    }
    return observer_class(**kwargs_by_name[name])


class ObserverTest(unittest.TestCase):
    def test_builtin_observers_do_not_mutate_event_graphs(self) -> None:
        bridge_decision = PolicyDecision(
            actions=(Action(ActionType.BRIDGE, source=0, target=1),),
            value=None,
        )

        for observer_name in OBSERVER_GRIMOIRE:
            with self.subTest(observer=observer_name):
                observer = _build_contract_test_observer(observer_name)
                initial_graph = create_empty_graph(3)
                post_graph = initial_graph.copy()
                post_graph.add_edge(0, 1)
                start_event = EpisodeStartEvent(
                    episode_index=0,
                    seed=11,
                    max_steps=1,
                    n_agents=3,
                    initial_state=StateView(graph=initial_graph),
                )
                step_event = StepEvent(
                    step=0,
                    decisions=(bridge_decision,),
                    modified=True,
                    pre_state=StateView(graph=initial_graph),
                    post_state=StateView(graph=post_graph),
                )
                end_event = EpisodeEndEvent(
                    episode_index=0,
                    seed=11,
                    current_step=1,
                    last_modified_step=0,
                    final_state=StateView(graph=post_graph),
                )

                with redirect_stdout(io.StringIO()):
                    before_initial = _graph_signature(initial_graph)
                    observer.on_episode_start(start_event)
                    self.assertEqual(
                        _graph_signature(initial_graph),
                        before_initial,
                        f"{observer_name} mutated the initial graph",
                    )

                    before_pre = _graph_signature(initial_graph)
                    before_post = _graph_signature(post_graph)
                    observer.on_step_end(step_event)
                    self.assertEqual(
                        _graph_signature(initial_graph),
                        before_pre,
                        f"{observer_name} mutated the pre-step graph",
                    )
                    self.assertEqual(
                        _graph_signature(post_graph),
                        before_post,
                        f"{observer_name} mutated the post-step graph",
                    )

                    before_final = _graph_signature(post_graph)
                    observer.on_episode_end(end_event)
                    self.assertEqual(
                        _graph_signature(post_graph),
                        before_final,
                        f"{observer_name} mutated the final graph",
                    )

    def test_action_print_observer_can_print_initial_empty_step(self) -> None:
        observer = ActionPrintObserver(ObservationSampling.periodic(1), True)
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            observer.on_episode_start(event=None)

        self.assertIn("step 0", stdout.getvalue())

    def test_action_print_observer_groups_actions_by_source_node(self) -> None:
        env = Environment(create_empty_graph(3), seed=3)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=5),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=7),
                Agent(id=2, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=2, target=0)), seed=11),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, fraction=1.0, seed=21),
            max_steps=1,
            rng=np.random.default_rng(13),
            observers=(ActionPrintObserver(ObservationSampling.periodic(1), False),),
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            simulator.run(n_episodes=1, reset_observers=False)

        output = stdout.getvalue()
        self.assertIn("step 1", output)
        self.assertIn("  0: BRIDGE->1", output)
        self.assertIn("  1: PASS", output)
        self.assertIn("  2: BRIDGE->0", output)

    def test_action_print_observer_shows_decision_value_on_same_line(self) -> None:
        env = Environment(create_empty_graph(2), seed=3)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1,
                    policy=ConstantPolicy(
                        Action(ActionType.BRIDGE, source=0, target=1),
                        value={"score": 1.25},
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
            actor_scheduler=ActorScheduler(n_agents=2, fraction=1.0, seed=21),
            max_steps=1,
            rng=np.random.default_rng(13),
            observers=(ActionPrintObserver(ObservationSampling.periodic(1), False),),
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            simulator.run(n_episodes=1, reset_observers=False)

        output = stdout.getvalue()
        self.assertIn("  0: BRIDGE->1 (value={score=1.25})", output)
        self.assertIn("  1: PASS", output)

    def test_density_observer_tracks_density_only(self) -> None:
        density_observer = DensityObserver(
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(2), seed=13)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=17),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=19),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, fraction=1.0, seed=21),
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

    def test_tensor_series_observer_tracks_matrix_values(self) -> None:
        tensor_observer = TensorMatrixObserver(
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(2), seed=13)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=17,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, fraction=1.0, seed=21),
            max_steps=2,
            rng=np.random.default_rng(23),
            observers=(tensor_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(tensor_observer.steps, np.array([0, 1, 2]))
        self.assertEqual(tensor_observer.sample_shape, (2, 2))
        self.assertEqual(tensor_observer.dimension_names, ("a", "b", "c", "d"))
        np.testing.assert_allclose(
            tensor_observer.values,
            np.array(
                [
                    [[0.0, 1.0], [2.0, 3.0]],
                    [[10.0, 11.0], [12.0, 13.0]],
                    [[20.0, 21.0], [22.0, 23.0]],
                ],
                dtype=np.float32,
            ),
        )

        result = tensor_observer.get()
        self.assertEqual(result.sample_shape, (2, 2))
        self.assertEqual(result.dimension_names, ("a", "b", "c", "d"))
        self.assertEqual(result.outputs, {})

    def test_scalar_series_observer_still_tracks_scalar_values(self) -> None:
        scalar_observer = ShapeMismatchObserver(
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(1), seed=13)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=17,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, fraction=1.0, seed=21),
            max_steps=1,
            rng=np.random.default_rng(19),
            observers=(scalar_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(scalar_observer.steps, np.array([0, 1]))
        np.testing.assert_allclose(
            scalar_observer.values,
            np.array([0.0, 1.0], dtype=np.float32),
        )

    def test_tensor_series_observer_rejects_mismatched_step_shape(self) -> None:
        mismatch_observer = TensorShapeMismatchObserver(
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(1), seed=13)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=17,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, fraction=1.0, seed=21),
            max_steps=1,
            rng=np.random.default_rng(19),
            observers=(mismatch_observer,),
        )

        with self.assertRaisesRegex(
            ValueError,
            "all sampled tensor values must share the same shape",
        ):
            simulator.run(n_episodes=1, reset_observers=False)

    def test_tensor_series_observer_rejects_scalar_values(self) -> None:
        scalar_value_observer = TensorScalarValueObserver(
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(1), seed=13)
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
            max_steps=0,
            rng=np.random.default_rng(19),
            observers=(scalar_value_observer,),
        )

        with self.assertRaisesRegex(
            ValueError,
            "require non-scalar",
        ):
            simulator.run(n_episodes=1, reset_observers=False)

    def test_tensor_series_observer_rejects_wrong_dimension_name_count(self) -> None:
        wrong_names_observer = TensorWrongDimensionNamesObserver(
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(1), seed=13)
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
            max_steps=0,
            rng=np.random.default_rng(19),
            observers=(wrong_names_observer,),
        )

        with self.assertRaisesRegex(
            ValueError,
            "dimension_names must match",
        ):
            simulator.run(n_episodes=1, reset_observers=False)

    def test_indegree_observer_uses_directed_adjacency_matrix(self) -> None:
        indegree_observer = IndegreeObserver(
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
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
            max_steps=0,
            rng=np.random.default_rng(19),
            observers=(indegree_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(
            indegree_observer.steps,
            np.array([0]),
        )
        np.testing.assert_allclose(
            indegree_observer.values,
            np.array([2.0 / 3.0], dtype=np.float32),
        )

    def test_triad_census_observer_returns_zero_when_no_triads_exist(self) -> None:
        triad_observer = TriadCensusObserver(
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(2), seed=17)
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
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
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(3), seed=17)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1,
                    policy=ConstantPolicy(
                        Action(ActionType.BRIDGE, source=0, target=1)
                    ),
                    seed=19,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
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
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
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
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(4), seed=31)
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
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
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=("mean", "std", "runs"),
        )
        env = Environment(create_empty_graph(3), seed=37)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1,
                    policy=ConstantPolicy(
                        Action(ActionType.BRIDGE, source=0, target=1)
                    ),
                    seed=41,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
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

    def test_distance_observer_uses_giant_component_and_full_graph_scale(
        self,
    ) -> None:
        distance_observer = DistanceObserver(
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
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
            max_steps=0,
            rng=np.random.default_rng(29),
            observers=(distance_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(
            distance_observer.steps,
            np.array([0]),
        )
        np.testing.assert_allclose(
            distance_observer.values,
            np.array([1.0], dtype=np.float32),
        )

    def test_global_closeness_centrality_observer_uses_full_graph_harmonic_scores(
        self,
    ) -> None:
        closeness_observer = GlobalClosenessCentralityObserver(
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
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
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

    def test_central_point_of_dominance_observer_matches_normalized_betweenness(
        self,
    ) -> None:
        dominance_observer = CentralPointOfDominanceObserver(
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
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
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

    def test_action_count_observer_normalizes_by_total_agents(self) -> None:
        action_count_observer = ActionCountObserver(
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(2), seed=41)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1,
                    policy=ActionListPolicy(
                        [Action(ActionType.BRIDGE, source=0, target=1)]
                    ),
                    seed=43,
                ),
                Agent(
                    id=1,
                    visibility_fraction=1,
                    policy=ActionListPolicy([Action(ActionType.PASS, source=1)]),
                    seed=47,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
            max_steps=1,
            rng=np.random.default_rng(53),
            observers=(action_count_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(action_count_observer.steps, np.array([0, 1]))
        np.testing.assert_allclose(
            action_count_observer.values,
            np.array([0.0, 0.5], dtype=np.float32),
        )

    def test_action_count_observer_excludes_pass_and_supports_multi_action(
        self,
    ) -> None:
        action_count_observer = ActionCountObserver(
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(3), seed=59)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1,
                    policy=ActionListPolicy(
                        [
                            Action(ActionType.BRIDGE, source=0, target=1),
                            Action(ActionType.BRIDGE, source=0, target=2),
                        ]
                    ),
                    seed=61,
                ),
                Agent(
                    id=1,
                    visibility_fraction=1,
                    policy=ActionListPolicy([Action(ActionType.PASS, source=1)]),
                    seed=67,
                ),
                Agent(
                    id=2,
                    visibility_fraction=1,
                    policy=ActionListPolicy(
                        [Action(ActionType.BRIDGE, source=2, target=1)]
                    ),
                    seed=71,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, fraction=1.0, seed=21),
            max_steps=1,
            rng=np.random.default_rng(73),
            observers=(action_count_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(action_count_observer.steps, np.array([0, 1]))
        np.testing.assert_allclose(
            action_count_observer.values,
            np.array([0.0, 1.0], dtype=np.float32),
        )

    def test_distance_observer_aggregates_mean_and_runs(self) -> None:
        distance_observer = DistanceObserver(
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
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
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
        np.testing.assert_allclose(
            runs[0],
            np.array([1.0], dtype=np.float32),
        )

    def test_central_point_of_dominance_observer_aggregates_mean_and_runs(
        self,
    ) -> None:
        dominance_observer = CentralPointOfDominanceObserver(
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
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
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
        np.testing.assert_allclose(
            runs[0],
            np.array([1.0], dtype=np.float32),
        )

    def test_hub_persistence_observer_keeps_tied_hubs_and_uses_overlap_fraction(
        self,
    ) -> None:
        hub_persistence_observer = HubPersistenceObserver(
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
                    visibility_fraction=1,
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
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=227,
                ),
                Agent(
                    id=2,
                    visibility_fraction=1,
                    policy=EpisodeCallPolicy(
                        [[
                            [Action(ActionType.ADD, source=2, target=1)],
                            [Action(ActionType.PASS, source=2)],
                        ]]
                    ),
                    seed=229,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, fraction=1.0, seed=21),
            max_steps=2,
            rng=np.random.default_rng(233),
            observers=(hub_persistence_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(
            hub_persistence_observer.steps,
            np.arange(3),
        )
        np.testing.assert_allclose(
            hub_persistence_observer.values,
            np.array([1.0 / 3.0, 1.0, np.nan], dtype=np.float32),
            equal_nan=True,
        )

    def test_hub_persistence_observer_uses_nan_borders_for_adjacent_ten_step_windows(
        self,
    ) -> None:
        hub_persistence_observer = HubPersistenceObserver(
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
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=241,
                ),
                Agent(
                    id=1,
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=251,
                ),
                Agent(
                    id=2,
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=2)),
                    seed=257,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
            max_steps=19,
            rng=np.random.default_rng(241),
            observers=(hub_persistence_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(
            hub_persistence_observer.steps,
            np.arange(20),
        )
        np.testing.assert_allclose(
            hub_persistence_observer.values,
            np.array([np.nan] * 9 + [1.0] + [np.nan] * 10, dtype=np.float32),
            equal_nan=True,
        )

    def test_hub_persistence_observer_aggregates_series_outputs(self) -> None:
        hub_persistence_observer = HubPersistenceObserver(
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
                    visibility_fraction=1,
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
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=263,
                ),
                Agent(
                    id=2,
                    visibility_fraction=1,
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
            actor_scheduler=ActorScheduler(n_agents=3, fraction=1.0, seed=21),
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
                    visibility_fraction=1,
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
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=227,
                ),
                Agent(
                    id=2,
                    visibility_fraction=1,
                    policy=EpisodeCallPolicy(
                        [[
                            [Action(ActionType.ADD, source=2, target=1)],
                            [Action(ActionType.PASS, source=2)],
                        ]]
                    ),
                    seed=229,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, fraction=1.0, seed=21),
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
            observation_sampling=ObservationSampling.terminal(),
            window_size=10,
            aggregations=(),
        )
        env = Environment(create_empty_graph(3), seed=239)
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
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

    def test_state_history_observer_tracks_graphs(self) -> None:
        state_history_observer = StateHistoryObserver(
            observation_sampling=ObservationSampling.periodic(1),
            keep_runs=False,
        )
        env = Environment(create_empty_graph(2), seed=29)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=31),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=37),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
            max_steps=2,
            rng=np.random.default_rng(41),
            observers=(state_history_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(state_history_observer.steps, np.array([0, 1, 2]))
        self.assertEqual(len(state_history_observer.graphs), 3)
        self.assertFalse(state_history_observer.graphs[0].are_adjacent(0, 1))
        self.assertTrue(state_history_observer.graphs[1].are_adjacent(0, 1))
        self.assertEqual(state_history_observer.graphs[0].ecount(), 0)
        self.assertEqual(state_history_observer.graphs[1].get_edgelist(), [(0, 1)])

    def test_state_history_observer_keeps_persistent_copies_of_graph_history(self) -> None:
        state_history_observer = StateHistoryObserver(
            observation_sampling=ObservationSampling.periodic(1),
            keep_runs=False,
        )
        env = Environment(create_empty_graph(2), seed=29)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=31),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=37),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
            max_steps=2,
            rng=np.random.default_rng(41),
            observers=(state_history_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        exported_graphs = state_history_observer.graphs
        simulator.env.state.graph.delete_edges([(0, 1)])
        self.assertTrue(state_history_observer.graphs[1].are_adjacent(0, 1))

        exported_graphs[1].delete_edges([(0, 1)])
        self.assertTrue(state_history_observer.graphs[1].are_adjacent(0, 1))

    def test_observers_record_sampled_step_ids_with_observation_sampling(self) -> None:
        density_observer = DensityObserver(
            observation_sampling=ObservationSampling.periodic(2),
            aggregations=(),
        )
        state_history_observer = StateHistoryObserver(
            observation_sampling=ObservationSampling.periodic(2),
            keep_runs=False,
        )
        env = Environment(create_empty_graph(2), seed=43)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=0)), seed=47),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=53),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
            max_steps=5,
            rng=np.random.default_rng(59),
            observers=(density_observer, state_history_observer),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(density_observer.steps, np.array([0, 2, 4]))
        np.testing.assert_allclose(
            density_observer.values,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(state_history_observer.steps, np.array([0, 2, 4]))
        self.assertEqual(len(state_history_observer.graphs), 3)

    def test_density_observer_terminal_sampling_returns_last_step_value(self) -> None:
        density_observer = DensityObserver(
            observation_sampling=ObservationSampling.terminal(),
            aggregations=(),
        )
        env = Environment(create_empty_graph(2), seed=43)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=47),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=53),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
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
            observation_sampling=ObservationSampling.terminal(),
            aggregations=(),
        )
        env = Environment(create_empty_graph(2), seed=61)
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
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

    def test_action_count_observer_terminal_sampling_uses_last_step_decisions(self) -> None:
        action_count_observer = ActionCountObserver(
            observation_sampling=ObservationSampling.terminal(),
            aggregations=(),
        )
        env = Environment(create_empty_graph(2), seed=41)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1,
                    policy=ActionListPolicy(
                        [Action(ActionType.BRIDGE, source=0, target=1)]
                    ),
                    seed=43,
                ),
                Agent(
                    id=1,
                    visibility_fraction=1,
                    policy=ActionListPolicy([Action(ActionType.PASS, source=1)]),
                    seed=47,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
            max_steps=1,
            rng=np.random.default_rng(53),
            observers=(action_count_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(action_count_observer.steps, np.array([1]))
        np.testing.assert_allclose(
            action_count_observer.values,
            np.array([0.5], dtype=np.float32),
        )

    def test_triad_census_observer_terminal_sampling_returns_last_tensor_value(self) -> None:
        triad_observer = TriadCensusObserver(
            observation_sampling=ObservationSampling.terminal(),
            aggregations=(),
        )
        env = Environment(create_empty_graph(3), seed=17)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1,
                    policy=ConstantPolicy(
                        Action(ActionType.BRIDGE, source=0, target=1)
                    ),
                    seed=19,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
            max_steps=1,
            rng=np.random.default_rng(23),
            observers=(triad_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        expected = np.zeros((1, len(TRIAD_TYPE_NAMES)), dtype=np.float32)
        expected[0, TRIAD_TYPE_NAMES.index("012")] = 1.0
        np.testing.assert_array_equal(triad_observer.steps, np.array([1]))
        np.testing.assert_allclose(triad_observer.values, expected)

    def test_state_history_observer_terminal_sampling_keeps_final_graph_only(self) -> None:
        state_history_observer = StateHistoryObserver(
            observation_sampling=ObservationSampling.terminal(),
            keep_runs=False,
        )
        env = Environment(create_empty_graph(2), seed=29)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=31),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=37),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
            max_steps=2,
            rng=np.random.default_rng(41),
            observers=(state_history_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(state_history_observer.steps, np.array([2]))
        self.assertEqual(len(state_history_observer.graphs), 1)
        self.assertTrue(state_history_observer.graphs[0].are_adjacent(0, 1))

    def test_action_print_observer_terminal_sampling_prints_only_last_completed_step(
        self,
    ) -> None:
        env = Environment(create_empty_graph(2), seed=3)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=5),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=7),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
            max_steps=2,
            rng=np.random.default_rng(13),
            observers=(ActionPrintObserver(ObservationSampling.terminal(), True),),
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            simulator.run(n_episodes=1, reset_observers=False)

        output = stdout.getvalue()
        self.assertNotIn("step 0", output)
        self.assertIn("step 2", output)

    def test_action_print_observer_terminal_sampling_prints_step_zero_once_when_no_step_executes(
        self,
    ) -> None:
        observer = ActionPrintObserver(ObservationSampling.terminal(), True)
        env = Environment(create_empty_graph(1), seed=3)
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
            max_steps=0,
            rng=np.random.default_rng(5),
            observers=(observer,),
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            simulator.run(n_episodes=1, reset_observers=False)

        self.assertEqual(stdout.getvalue().strip(), "step 0")

    def test_density_observer_aggregates_selected_outputs_and_runs(self) -> None:
        density_observer = DensityObserver(
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=("mean", "runs"),
        )
        env = Environment(create_empty_graph(2), seed=61)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=67),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=71),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
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
            observation_sampling=ObservationSampling.periodic(2),
            aggregations=("mean", "runs"),
        )
        env = Environment(create_empty_graph(2), seed=101)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=0)), seed=103),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=107),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
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
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=("std",),
        )
        env = Environment(create_empty_graph(2), seed=79)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=83),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=89),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
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

    def test_observers_keep_sampling_tail_steps_after_early_stopping(self) -> None:
        density_observer = DensityObserver(
            observation_sampling=ObservationSampling.periodic(2),
            aggregations=(),
        )
        state_history_observer = StateHistoryObserver(
            observation_sampling=ObservationSampling.periodic(2),
            keep_runs=False,
        )
        env = Environment(create_empty_graph(2), seed=113)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=0)), seed=127),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=131),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
            max_steps=5,
            rng=np.random.default_rng(137),
            observers=(density_observer, state_history_observer),
            stagnation_patience=1,
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(density_observer.steps, np.array([0, 2, 4]))
        np.testing.assert_allclose(
            density_observer.values,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(state_history_observer.steps, np.array([0, 2, 4]))
        self.assertEqual(len(state_history_observer.graphs), 3)


if __name__ == "__main__":
    unittest.main()
