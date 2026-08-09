"""Tests for observer contracts and statistic base behavior."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import numpy as np
from numpy.typing import NDArray

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.graph import create_empty_graph
from network_creation.core.model import (
    Action,
    ActionType,
    ActorScheduler,
    PolicyDecision,
)
from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    ObservationSampling,
    StateView,
    StepEvent,
)
from network_creation.core.simulator import Simulator
from network_creation.observers.grimoire import OBSERVER_GRIMOIRE
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)
from network_creation.observers.statistics.episode_tensor_series import (
    EpisodeSeriesTensorStatisticObserver,
)
from tests.core.simulation_support import ConstantPolicy


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
        "action_count": {
            "n_agents": 3,
            "observation_sampling": periodic,
            "aggregations": (),
        },
        "action_print": {"observation_sampling": periodic, "print_empty_steps": False},
        "central_point_of_dominance": {
            "n_agents": 3,
            "observation_sampling": periodic,
            "aggregations": (),
        },
        "density": {
            "n_agents": 3,
            "observation_sampling": periodic,
            "aggregations": (),
        },
        "distance": {
            "n_agents": 3,
            "observation_sampling": periodic,
            "aggregations": (),
        },
        "global_closeness_centrality": {
            "n_agents": 3,
            "observation_sampling": periodic,
            "aggregations": (),
        },
        "hub_persistence": {
            "n_agents": 3,
            "observation_sampling": periodic,
            "window_size": 1,
            "aggregations": (),
        },
        "hub_sponsorship": {
            "n_agents": 3,
            "observation_sampling": periodic,
            "aggregations": (),
        },
        "indegree": {
            "n_agents": 3,
            "observation_sampling": periodic,
            "aggregations": (),
        },
        "agent_payoff": {
            "n_agents": 3,
            "observation_sampling": periodic,
            "benefit_to_cost_ratio": 1.0,
            "distance_discount": 0.5,
            "aggregations": (),
        },
        "policy_welfare": {
            "n_agents": 3,
            "observation_sampling": periodic,
            "benefit_to_cost_ratio": 1.0,
            "distance_discount": 0.5,
            "aggregations": (),
            "policy_ids": ("first", "second"),
            "policy_slot_by_agent": np.asarray([0, 1, 0], dtype=np.int32),
        },
        "centered_mean_welfare": {
            "n_agents": 3,
            "observation_sampling": periodic,
            "benefit_to_cost_ratio": 1.0,
            "distance_discount": 0.5,
            "aggregations": (),
        },
        "state_history": {"observation_sampling": periodic, "keep_runs": False},
        "triad_census": {
            "n_agents": 3,
            "observation_sampling": periodic,
            "aggregations": (),
        },
    }
    return observer_class(**kwargs_by_name[name])


class ObserverContractTest(unittest.TestCase):
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

    def test_tensor_series_observer_tracks_matrix_values(self) -> None:
        tensor_observer = TensorMatrixObserver(
            n_agents=1,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(2), seed=13)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=17,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, n_actors=3, activation_weights=(1.0, 1.0, 1.0), seed=21),
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
            n_agents=1,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(1), seed=13)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=17,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, n_actors=3, activation_weights=(1.0, 1.0, 1.0), seed=21),
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
            n_agents=1,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(1), seed=13)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=17,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, n_actors=3, activation_weights=(1.0, 1.0, 1.0), seed=21),
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
            n_agents=0,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(1), seed=13)
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, activation_weights=(1.0,), seed=21),
            max_steps=0,
            rng=np.random.default_rng(19),
            observers=(scalar_value_observer,),
        )

        with self.assertRaisesRegex(ValueError, "require non-scalar"):
            simulator.run(n_episodes=1, reset_observers=False)

    def test_tensor_series_observer_rejects_wrong_dimension_name_count(self) -> None:
        wrong_names_observer = TensorWrongDimensionNamesObserver(
            n_agents=0,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(1), seed=13)
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, activation_weights=(1.0,), seed=21),
            max_steps=0,
            rng=np.random.default_rng(19),
            observers=(wrong_names_observer,),
        )

        with self.assertRaisesRegex(ValueError, "dimension_names must match"):
            simulator.run(n_episodes=1, reset_observers=False)


if __name__ == "__main__":
    unittest.main()
