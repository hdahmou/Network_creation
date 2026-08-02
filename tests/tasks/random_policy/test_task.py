"""Tests for the concrete policy task."""

from __future__ import annotations

import unittest

import numpy as np

from tasks.execution.schema import OutputArraySpec
from tasks.policies import prepare_policy_run
from tasks.policies.simulation import _pad_observer_time_axis
from tests.tasks.config_support import (
    activation_opportunities_per_agent,
    fixed,
    policy,
    raw_simulator_steps,
    sampled_discrete_range,
    scaled_inverse_agents,
    task_config,
)


class RandomPolicyTaskTest(unittest.TestCase):
    @staticmethod
    def _base_config() -> dict[str, object]:
        return task_config(            game={
                "benefit_to_cost_ratio": fixed(2.0),
                "distance_discount": fixed(0.5),
            },
            policies=[policy("random", "random")],
        )

    def test_prepare_builds_structured_output_schema_and_run_returns_arrays(self) -> None:
        prepared = prepare_policy_run(self._base_config())

        self.assertEqual(
            prepared.output_schema.metadata(),
            [
                {"name": "density.mean", "shape": [4], "dtype": "float32"},
                {"name": "centered_mean_welfare.mean", "shape": [4], "dtype": "float32"},
            ],
        )
        self.assertEqual(prepared.inputs, (0,))

        output = prepared.run(prepared.inputs[0])

        self.assertEqual(set(output), {"density.mean", "centered_mean_welfare.mean"})
        self.assertEqual(output["density.mean"].shape, (4,))
        self.assertEqual(output["centered_mean_welfare.mean"].shape, (4,))
        self.assertEqual(output["density.mean"].dtype, np.float32)
        self.assertEqual(output["centered_mean_welfare.mean"].dtype, np.float32)

    def test_prepare_saves_all_configured_observer_aggregations(self) -> None:
        config = self._base_config()
        config["observers"] = {
            "defaults": {
                "sampling": {"kind": "periodic", "every": 1},
                "aggregations": ["mean", "std"],
            },
            "outputs": [{"name": "density"}],
        }

        prepared = prepare_policy_run(config)

        self.assertEqual(
            prepared.output_schema.metadata(),
            [
                {"name": "density.mean", "shape": [4], "dtype": "float32"},
                {"name": "density.std", "shape": [4], "dtype": "float32"},
            ],
        )
        output = prepared.run(prepared.inputs[0])
        self.assertEqual(set(output), {"density.mean", "density.std"})
        self.assertEqual(output["density.mean"].shape, output["density.std"].shape)
        self.assertEqual(output["density.std"].shape, (4,))

    def test_prepared_run_is_deterministic_for_fixed_seed(self) -> None:
        prepared = prepare_policy_run(self._base_config())

        first = prepared.run(prepared.inputs[0])
        second = prepared.run(prepared.inputs[0])

        self.assertEqual(set(first), set(second))
        for field_name in first:
            np.testing.assert_allclose(first[field_name], second[field_name])

    def test_run_supports_n_agents_sweeps_and_auto_caps_actor_count(self) -> None:
        config = self._base_config()
        config["population"]["n_agents"] = [2, 4]

        prepared = prepare_policy_run(config)

        self.assertEqual(prepared.inputs, (0, 1))
        for sample_idx in prepared.inputs:
            output = prepared.run(sample_idx)
            self.assertEqual(set(output), {"density.mean", "centered_mean_welfare.mean"})

    def test_run_uses_scaled_inverse_benefit_to_cost_ratio_from_n_agents(self) -> None:
        sweep_config = self._base_config()
        sweep_config["population"]["n_agents"] = [5, 10]
        sweep_config["game"]["benefit_to_cost_ratio"] = scaled_inverse_agents(
            10.0, 2.0
        )

        small_config = self._base_config()
        small_config["population"]["n_agents"] = 5
        small_config["game"]["benefit_to_cost_ratio"] = scaled_inverse_agents(
            10.0, 2.0
        )

        large_config = self._base_config()
        large_config["population"]["n_agents"] = 10
        large_config["game"]["benefit_to_cost_ratio"] = scaled_inverse_agents(
            10.0, 2.0
        )

        sweep_prepared = prepare_policy_run(sweep_config)
        small_prepared = prepare_policy_run(small_config)
        large_prepared = prepare_policy_run(large_config)

        expected_outputs = (
            small_prepared.run(small_prepared.inputs[0]),
            large_prepared.run(large_prepared.inputs[0]),
        )

        self.assertEqual(sweep_prepared.inputs, (0, 1))
        for sample_idx, expected_output in zip(
            sweep_prepared.inputs,
            expected_outputs,
            strict=True,
        ):
            observed_output = sweep_prepared.run(sample_idx)
            self.assertEqual(set(observed_output), set(expected_output))
            for field_name in observed_output:
                np.testing.assert_allclose(
                    observed_output[field_name],
                    expected_output[field_name],
                )

    def test_run_fails_when_centered_mean_welfare_runtime_parameters_are_missing(self) -> None:
        config = self._base_config()
        config["game"] = {}

        prepared = prepare_policy_run(config)

        with self.assertRaisesRegex(KeyError, "benefit_to_cost_ratio"):
            prepared.run(prepared.inputs[0])

    def test_prepare_supports_series_and_tensor_observers_in_one_schema(self) -> None:
        config = self._base_config()
        config["observers"]["outputs"] = [
            {"name": "density"},
            {"name": "triad_census"},
        ]
        prepared = prepare_policy_run(config)

        self.assertEqual(
            prepared.output_schema.metadata(),
            [
                {"name": "density.mean", "shape": [4], "dtype": "float32"},
                {"name": "triad_census.mean", "shape": [4, 7], "dtype": "float32"},
            ],
        )
        output = prepared.run(prepared.inputs[0])
        self.assertEqual(output["density.mean"].shape, (4,))
        self.assertEqual(output["triad_census.mean"].shape, (4, 7))

    def test_activation_budget_pads_scalar_and_tensor_series_time_axes(self) -> None:
        observers = {
            "defaults": {
                "sampling": {"kind": "periodic", "every": 1},
                "aggregations": ["mean", "std"],
            },
            "outputs": [{"name": "density"}, {"name": "triad_census"}],
        }
        config = task_config(
            n_agents=[3, 5],
            n_actors=2,
            budget=activation_opportunities_per_agent(2.5),
            observers=observers,
        )
        native_config = task_config(
            n_agents=3,
            n_actors=2,
            budget=raw_simulator_steps(4),
            observers=observers,
        )

        prepared = prepare_policy_run(config)
        output = prepared.run(0)
        native_output = prepare_policy_run(native_config).run(0)

        self.assertEqual(output["density.mean"].shape, (8,))
        self.assertEqual(output["density.std"].shape, (8,))
        self.assertEqual(output["triad_census.mean"].shape, (8, 7))
        self.assertEqual(output["triad_census.std"].shape, (8, 7))
        for field_name, values in output.items():
            np.testing.assert_allclose(values[:5], native_output[field_name])
            self.assertTrue(np.isnan(values[5:]).all())

    def test_activation_budget_pads_runs_after_episode_axis(self) -> None:
        observers = {
            "defaults": {
                "sampling": {"kind": "periodic", "every": 1},
                "aggregations": ["runs"],
            },
            "outputs": [{"name": "density"}],
        }
        config = task_config(
            n_agents=[3, 5],
            n_actors=2,
            n_episodes=2,
            budget=activation_opportunities_per_agent(2.5),
            observers=observers,
        )
        native_config = task_config(
            n_agents=3,
            n_actors=2,
            n_episodes=2,
            budget=raw_simulator_steps(4),
            observers=observers,
        )

        prepared = prepare_policy_run(config)
        output = prepared.run(0)["density.runs"]
        native_output = prepare_policy_run(native_config).run(0)["density.runs"]

        self.assertEqual(output.shape, (2, 8))
        np.testing.assert_allclose(output[:, :5], native_output)
        self.assertTrue(np.isnan(output[:, 5:]).all())

    def test_observer_time_padding_rejects_invalid_shapes_and_leaves_terminal(
        self,
    ) -> None:
        field = OutputArraySpec(
            name="example.mean",
            shape=(3, 2),
            dtype=np.float32,
        )
        for values, expected_error in (
            (np.zeros((4, 2), dtype=np.float32), "longer"),
            (np.zeros((2, 3), dtype=np.float32), "non-time"),
        ):
            with self.subTest(shape=values.shape):
                with self.assertRaisesRegex(ValueError, expected_error):
                    _pad_observer_time_axis(
                        values,
                        field,
                        aggregation="mean",
                        is_terminal=False,
                    )

        terminal_field = OutputArraySpec(
            name="example.mean",
            shape=(1, 2),
            dtype=np.float32,
        )
        terminal = np.ones((1, 2), dtype=np.float32)
        self.assertIs(
            _pad_observer_time_axis(
                terminal,
                terminal_field,
                aggregation="mean",
                is_terminal=True,
            ),
            terminal,
        )

    def test_prepare_pads_agent_payoff_for_n_agents_sweeps(self) -> None:
        config = self._base_config()
        config["population"]["n_agents"] = [2, 4]
        config["observers"]["outputs"] = [
            {"name": "agent_payoff"},
        ]
        prepared = prepare_policy_run(config)

        self.assertEqual(
            prepared.output_schema.metadata(),
            [
                {"name": "agent_payoff.mean", "shape": [4, 4], "dtype": "float32"},
            ],
        )

        small_output = prepared.run(0)["agent_payoff.mean"]
        large_output = prepared.run(1)["agent_payoff.mean"]

        self.assertEqual(small_output.shape, (4, 4))
        self.assertEqual(large_output.shape, (4, 4))
        self.assertTrue(np.isnan(small_output[:, 2:]).all())

    def test_prepare_supports_terminal_observer_outputs(self) -> None:
        config = self._base_config()
        config["observers"] = {
            "defaults": {
                "sampling": {"kind": "terminal"},
                "aggregations": ["mean"],
            },
            "outputs": [{"name": "density"}],
        }
        prepared = prepare_policy_run(config)

        self.assertEqual(
            prepared.output_schema.metadata(),
            [
                {"name": "density.mean", "shape": [1], "dtype": "float32"},
            ],
        )
        output = prepared.run(prepared.inputs[0])
        self.assertEqual(output["density.mean"].shape, (1,))

    def test_prepare_supports_terminal_agent_payoff_runs(self) -> None:
        config = self._base_config()
        config["population"]["n_agents"] = [3, 5]
        config["simulator"]["actor_scheduler"]["n_actors"] = 2
        config["simulator"]["budget"] = activation_opportunities_per_agent(2.5)
        config["observers"]["outputs"] = [
            {
                "name": "agent_payoff",
                "sampling": {"kind": "terminal"},
                "aggregations": ["runs"],
            }
        ]
        prepared = prepare_policy_run(config)

        self.assertEqual(
            prepared.output_schema.metadata(),
            [
                {
                    "name": "agent_payoff.runs",
                    "shape": [2, 1, 5],
                    "dtype": "float32",
                },
            ],
        )

        small_output = prepared.run(0)["agent_payoff.runs"]
        large_output = prepared.run(1)["agent_payoff.runs"]

        self.assertEqual(small_output.shape, (2, 1, 5))
        self.assertEqual(large_output.shape, (2, 1, 5))
        self.assertTrue(np.isnan(small_output[..., 3:]).all())

    def test_run_supports_multiple_visibility_distance_samples(self) -> None:
        config = self._base_config()
        config["sampling"]["log2_budget"] = 1
        config["population"]["visibility_distance"] = sampled_discrete_range(
            1, 4
        )
        prepared = prepare_policy_run(config)

        self.assertEqual(prepared.inputs, (0, 1))
        for sample_idx in prepared.inputs:
            output = prepared.run(sample_idx)
            self.assertEqual(set(output), {"density.mean", "centered_mean_welfare.mean"})
