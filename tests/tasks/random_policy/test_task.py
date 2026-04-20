"""Tests for the concrete random-policy task."""

from __future__ import annotations

import unittest

import numpy as np

from tasks.policies import RandomPolicyRun


class RandomPolicyTaskTest(unittest.TestCase):
    @staticmethod
    def _base_config() -> dict[str, object]:
        return {
            "input_spec": {
                "log2_budget": 0,
                "sampler_seed": 123,
                "parameters": {"visibility_distance": {"type": "fixed", "payload": 4}},
            },
            "simulation": {
                "n_agents": 4,
                "actor_scheduler": {"n_actors": 4},
                "seed": 11,
                "max_steps": 3,
                "n_episodes": 2,
                "stagnation_patience": None,
                "initial_graph": {"kind": "empty"},
            },
            "observers": [
                {
                    "name": "density",
                    "output": "mean",
                    "kwargs": {
                        "observation_sampling": {"kind": "periodic", "every": 1},
                        "aggregations": ["mean"],
                    },
                },
                {
                    "name": "centered_mean_welfare",
                    "output": "mean",
                    "kwargs": {
                        "observation_sampling": {"kind": "periodic", "every": 1},
                        "aggregations": ["mean"],
                        "benefit_to_cost_ratio": 0.1,
                        "distance_discount": 0.5,
                    },
                },
            ],
        }

    def test_prepare_builds_structured_output_schema_and_run_returns_arrays(self) -> None:
        task = RandomPolicyRun()
        prepared = task.prepare(self._base_config())

        self.assertEqual(
            prepared.output_schema.metadata(),
            [
                {"name": "density.mean", "shape": [4], "dtype": "float32"},
                {"name": "centered_mean_welfare.mean", "shape": [4], "dtype": "float32"},
            ],
        )
        self.assertEqual(prepared.inputs, (0,))
        self.assertEqual(task.parameter_table.parameter_names, ("visibility_distance",))
        self.assertEqual(task.parameter_table.parameter_values.shape, (1, 1))

        output = prepared.run(prepared.inputs[0])

        self.assertEqual(set(output), {"density.mean", "centered_mean_welfare.mean"})
        self.assertEqual(output["density.mean"].shape, (4,))
        self.assertEqual(output["centered_mean_welfare.mean"].shape, (4,))
        self.assertEqual(output["density.mean"].dtype, np.float32)
        self.assertEqual(output["centered_mean_welfare.mean"].dtype, np.float32)

    def test_prepared_run_is_deterministic_for_fixed_seed(self) -> None:
        prepared = RandomPolicyRun().prepare(self._base_config())

        first = prepared.run(prepared.inputs[0])
        second = prepared.run(prepared.inputs[0])

        self.assertEqual(set(first), set(second))
        for field_name in first:
            np.testing.assert_allclose(first[field_name], second[field_name])

    def test_run_supports_n_agents_sweeps_and_auto_caps_actor_count(self) -> None:
        config = self._base_config()
        config["simulation"]["n_agents"] = [2, 4]

        prepared = RandomPolicyRun().prepare(config)

        self.assertEqual(prepared.inputs, (0, 1))
        for sample_idx in prepared.inputs:
            output = prepared.run(sample_idx)
            self.assertEqual(set(output), {"density.mean", "centered_mean_welfare.mean"})

    def test_run_uses_scaled_inverse_benefit_to_cost_ratio_from_n_agents(self) -> None:
        sweep_config = self._base_config()
        sweep_config["simulation"]["n_agents"] = [5, 10]
        sweep_config["input_spec"]["parameters"]["benefit_to_cost_ratio"] = {
            "type": "scaled_inverse_n_agents",
            "payload": {
                "numerator": 10.0,
                "factor": 2.0,
            },
        }
        del sweep_config["observers"][1]["kwargs"]["benefit_to_cost_ratio"]

        small_config = self._base_config()
        small_config["simulation"]["n_agents"] = 5
        small_config["input_spec"]["parameters"]["benefit_to_cost_ratio"] = {
            "type": "scaled_inverse_n_agents",
            "payload": {
                "numerator": 10.0,
                "factor": 2.0,
            },
        }
        del small_config["observers"][1]["kwargs"]["benefit_to_cost_ratio"]

        large_config = self._base_config()
        large_config["simulation"]["n_agents"] = 10
        large_config["input_spec"]["parameters"]["benefit_to_cost_ratio"] = {
            "type": "scaled_inverse_n_agents",
            "payload": {
                "numerator": 10.0,
                "factor": 2.0,
            },
        }
        del large_config["observers"][1]["kwargs"]["benefit_to_cost_ratio"]

        sweep_prepared = RandomPolicyRun().prepare(sweep_config)
        small_prepared = RandomPolicyRun().prepare(small_config)
        large_prepared = RandomPolicyRun().prepare(large_config)

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
        del config["observers"][1]["kwargs"]["benefit_to_cost_ratio"]
        del config["observers"][1]["kwargs"]["distance_discount"]

        prepared = RandomPolicyRun().prepare(config)

        with self.assertRaisesRegex(KeyError, "benefit_to_cost_ratio"):
            prepared.run(prepared.inputs[0])

    def test_prepare_supports_series_and_tensor_observers_in_one_schema(self) -> None:
        config = self._base_config()
        config["observers"] = [
            {
                "name": "density",
                "output": "mean",
                "kwargs": {
                    "observation_sampling": {"kind": "periodic", "every": 1},
                    "aggregations": ["mean"],
                },
            },
            {
                "name": "triad_census",
                "output": "mean",
                "kwargs": {
                    "observation_sampling": {"kind": "periodic", "every": 1},
                    "aggregations": ["mean"],
                },
            },
        ]
        prepared = RandomPolicyRun().prepare(config)

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

    def test_prepare_supports_terminal_observer_outputs(self) -> None:
        config = self._base_config()
        config["observers"] = [
            {
                "name": "density",
                "output": "mean",
                "kwargs": {
                    "observation_sampling": {"kind": "terminal"},
                    "aggregations": ["mean"],
                },
            },
        ]
        prepared = RandomPolicyRun().prepare(config)

        self.assertEqual(
            prepared.output_schema.metadata(),
            [
                {"name": "density.mean", "shape": [1], "dtype": "float32"},
            ],
        )
        output = prepared.run(prepared.inputs[0])
        self.assertEqual(output["density.mean"].shape, (1,))

    def test_run_supports_multiple_visibility_distance_samples(self) -> None:
        config = self._base_config()
        config["input_spec"] = {
            "log2_budget": 1,
            "sampler_seed": 123,
            "parameters": {
                "visibility_distance": {
                    "type": "sampled_discrete_range",
                    "payload": {"min": 1, "max": 4, "step": 1},
                }
            },
        }
        prepared = RandomPolicyRun().prepare(config)

        self.assertEqual(prepared.inputs, (0, 1))
        for sample_idx in prepared.inputs:
            output = prepared.run(sample_idx)
            self.assertEqual(set(output), {"density.mean", "centered_mean_welfare.mean"})
