"""Tests for the concrete best-response-policy task."""

from __future__ import annotations

import unittest

import numpy as np

from tasks.policies import BestResponsePolicyRun


class BestResponsePolicyTaskTest(unittest.TestCase):
    @staticmethod
    def _base_config() -> dict[str, object]:
        return {
            "input_spec": {
                "log2_budget": 0,
                "sampler_seed": 123,
                "parameters": {
                    "benefit_to_cost_ratio": {"type": "fixed", "payload": 0.1},
                    "distance_discount": {"type": "fixed", "payload": 0.5},
                    "bridge_weight": {"type": "fixed", "payload": 0.75},
                    "pass_bias": {"type": "fixed", "payload": 0.25},
                    "exploration": {"type": "fixed", "payload": 0.5},
                    "opportunity_sensitivity": {"type": "fixed", "payload": 0.5},
                    "visibility_distance": {"type": "fixed", "payload": 4},
                },
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
                    "name": "normalized_welfare",
                    "output": "mean",
                    "kwargs": {
                        "observation_sampling": {"kind": "periodic", "every": 1},
                        "aggregations": ["mean"],
                    },
                },
            ],
        }

    def test_prepare_builds_structured_output_schema_and_run_returns_arrays(self) -> None:
        task = BestResponsePolicyRun()
        prepared = task.prepare(self._base_config())

        self.assertEqual(
            prepared.output_schema.metadata(),
            [
                {"name": "density.mean", "shape": [4], "dtype": "float32"},
                {"name": "normalized_welfare.mean", "shape": [4], "dtype": "float32"},
            ],
        )
        self.assertEqual(prepared.inputs, (0,))
        self.assertEqual(
            task.parameter_table.parameter_names,
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "bridge_weight",
                "pass_bias",
                "exploration",
                "opportunity_sensitivity",
                "visibility_distance",
            ),
        )

        output = prepared.run(prepared.inputs[0])

        self.assertEqual(set(output), {"density.mean", "normalized_welfare.mean"})
        self.assertEqual(output["density.mean"].shape, (4,))
        self.assertEqual(output["normalized_welfare.mean"].shape, (4,))
        self.assertEqual(output["density.mean"].dtype, np.float32)
        self.assertEqual(output["normalized_welfare.mean"].dtype, np.float32)

    def test_prepared_run_is_deterministic_for_fixed_seed(self) -> None:
        prepared = BestResponsePolicyRun().prepare(self._base_config())

        first = prepared.run(prepared.inputs[0])
        second = prepared.run(prepared.inputs[0])

        self.assertEqual(set(first), set(second))
        for field_name in first:
            np.testing.assert_allclose(first[field_name], second[field_name])

    def test_run_matches_explicit_normalized_welfare_kwargs(self) -> None:
        inferred_prepared = BestResponsePolicyRun().prepare(self._base_config())

        explicit_config = self._base_config()
        explicit_config["observers"][1]["kwargs"]["benefit_to_cost_ratio"] = 0.1
        explicit_config["observers"][1]["kwargs"]["distance_discount"] = 0.5
        explicit_prepared = BestResponsePolicyRun().prepare(explicit_config)

        inferred_output = inferred_prepared.run(inferred_prepared.inputs[0])
        explicit_output = explicit_prepared.run(explicit_prepared.inputs[0])

        np.testing.assert_allclose(
            inferred_output["normalized_welfare.mean"],
            explicit_output["normalized_welfare.mean"],
        )

    def test_run_uses_sampled_n_agents_from_sweeps(self) -> None:
        sweep_config = self._base_config()
        sweep_config["simulation"]["n_agents"] = [2, 4]

        small_config = self._base_config()
        small_config["simulation"]["n_agents"] = 2

        large_config = self._base_config()
        large_config["simulation"]["n_agents"] = 4

        sweep_prepared = BestResponsePolicyRun().prepare(sweep_config)
        small_prepared = BestResponsePolicyRun().prepare(small_config)
        large_prepared = BestResponsePolicyRun().prepare(large_config)

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

        small_config = self._base_config()
        small_config["simulation"]["n_agents"] = 5
        small_config["input_spec"]["parameters"]["benefit_to_cost_ratio"] = {
            "type": "scaled_inverse_n_agents",
            "payload": {
                "numerator": 10.0,
                "factor": 2.0,
            },
        }

        large_config = self._base_config()
        large_config["simulation"]["n_agents"] = 10
        large_config["input_spec"]["parameters"]["benefit_to_cost_ratio"] = {
            "type": "scaled_inverse_n_agents",
            "payload": {
                "numerator": 10.0,
                "factor": 2.0,
            },
        }

        sweep_prepared = BestResponsePolicyRun().prepare(sweep_config)
        small_prepared = BestResponsePolicyRun().prepare(small_config)
        large_prepared = BestResponsePolicyRun().prepare(large_config)

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

    def test_prepare_supports_multiple_parameter_samples(self) -> None:
        config = self._base_config()
        config["input_spec"] = {
            "log2_budget": 1,
            "sampler_seed": 123,
            "parameters": {
                "benefit_to_cost_ratio": {"type": "fixed", "payload": 0.1},
                "distance_discount": {"type": "fixed", "payload": 0.5},
                "bridge_weight": {
                    "type": "continuous",
                    "payload": {"min": 0.0, "max": 1.0},
                },
                "pass_bias": {
                    "type": "continuous",
                    "payload": {"min": 0.0, "max": 1.0},
                },
                "exploration": {
                    "type": "continuous",
                    "payload": {"min": 0.0, "max": 1.0},
                },
                "opportunity_sensitivity": {
                    "type": "continuous",
                    "payload": {"min": 0.0, "max": 1.0},
                },
                "visibility_distance": {"type": "fixed", "payload": 4},
            },
        }
        prepared = BestResponsePolicyRun().prepare(config)

        self.assertEqual(prepared.inputs, (0, 1))
        for sample_idx in prepared.inputs:
            output = prepared.run(sample_idx)
            self.assertEqual(set(output), {"density.mean", "normalized_welfare.mean"})
