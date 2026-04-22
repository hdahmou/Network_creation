"""Tests for best-response-policy task preparation."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import qmc

from tasks.policies import BestResponsePolicyRun
from tasks.policies.support import generate_parameter_table


class BestResponsePolicyPreparationTest(unittest.TestCase):
    @staticmethod
    def _base_config() -> dict[str, object]:
        return {
            "input_spec": {
                "log2_budget": 0,
                "sampler_seed": 123,
                "parameters": {
                    "benefit_to_cost_ratio": {"type": "fixed", "payload": 0.05},
                    "distance_discount": {"type": "fixed", "payload": 0.5},
                    "outside_optimism": {"type": "fixed", "payload": 0.25},
                    "exploration": {"type": "fixed", "payload": 0.5},
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
            ],
        }

    def test_generate_parameter_table_preserves_order_and_repeats_fixed_columns(self) -> None:
        input_spec = {
            "log2_budget": 2,
            "sampler_seed": 123,
            "parameters": {
                "benefit_to_cost_ratio": {"type": "fixed", "payload": 0.05},
                "distance_discount": {
                    "type": "continuous",
                    "payload": {"min": 0.2, "max": 0.8},
                },
                "outside_optimism": {
                    "type": "continuous",
                    "payload": {"min": 0.0, "max": 1.0},
                },
                "exploration": {"type": "fixed", "payload": 0.5},
                "visibility_distance": {"type": "fixed", "payload": 4},
            },
        }

        parameter_table = generate_parameter_table(input_spec)

        self.assertEqual(
            parameter_table.parameter_names,
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "outside_optimism",
                "exploration",
                "visibility_distance",
            ),
        )
        self.assertEqual(parameter_table.parameter_values.shape, (4, 5))
        np.testing.assert_allclose(parameter_table.parameter_values[:, 0], 0.05)
        np.testing.assert_allclose(parameter_table.parameter_values[:, 3], 0.5)
        np.testing.assert_allclose(parameter_table.parameter_values[:, 4], 4.0)

    def test_generate_parameter_table_samples_continuous_parameters_jointly(self) -> None:
        input_spec = {
            "log2_budget": 2,
            "sampler_seed": 123,
            "parameters": {
                "distance_discount": {
                    "type": "continuous",
                    "payload": {"min": 0.2, "max": 0.8},
                },
                "outside_optimism": {
                    "type": "continuous",
                    "payload": {"min": 0.0, "max": 1.0},
                },
                "exploration": {
                    "type": "continuous",
                    "payload": {"min": 1e-3, "max": 1.0},
                },
                "visibility_distance": {"type": "fixed", "payload": 4},
            },
        }

        parameter_table = generate_parameter_table(input_spec)
        sampler = qmc.Sobol(
            d=3,
            scramble=True,
            optimization="random-cd",
            rng=123,
        )
        unit_samples = np.asarray(sampler.random(n=4), dtype=np.float32)
        expected = np.column_stack(
            (
                0.2 + (unit_samples[:, 0] * 0.6),
                unit_samples[:, 1],
                1e-3 + (unit_samples[:, 2] * 0.999),
                np.full(4, 4.0, dtype=np.float32),
            )
        ).astype(np.float32, copy=False)

        self.assertEqual(
            parameter_table.parameter_names,
            (
                "distance_discount",
                "outside_optimism",
                "exploration",
                "visibility_distance",
            ),
        )
        np.testing.assert_allclose(parameter_table.parameter_values, expected)

    def test_prepare_generates_row_index_inputs_from_input_spec(self) -> None:
        task = BestResponsePolicyRun()
        prepared = task.prepare(self._base_config())

        self.assertEqual(prepared.inputs, (0,))
        self.assertEqual(task.simulation_sweep.configuration_values.shape, (1, 1, 6))
        self.assertEqual(
            task.parameter_table.parameter_names,
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "outside_optimism",
                "exploration",
                "visibility_distance",
            ),
        )
        np.testing.assert_allclose(
            task.parameter_table.parameter_values,
            np.array(
                [[0.05, 0.5, 0.25, 0.5, 4.0]],
                dtype=np.float32,
            ),
        )

    def test_prepare_rejects_missing_visibility_distance(self) -> None:
        config = self._base_config()
        parameters = dict(config["input_spec"]["parameters"])
        del parameters["visibility_distance"]
        config["input_spec"] = {
            "log2_budget": 0,
            "sampler_seed": 123,
            "parameters": parameters,
        }

        with self.assertRaisesRegex(ValueError, "visibility_distance is required"):
            BestResponsePolicyRun().prepare(config)

    def test_prepare_rejects_scaled_inverse_benefit_to_cost_ratio_above_one(
        self,
    ) -> None:
        config = self._base_config()
        config["simulation"]["n_agents"] = 4
        config["input_spec"] = {
            "log2_budget": 0,
            "sampler_seed": 123,
            "parameters": {
                "benefit_to_cost_ratio": {
                    "type": "scaled_inverse_n_agents",
                    "payload": {
                        "numerator": 10.0,
                        "factor": 2.0,
                    },
                },
                "distance_discount": {"type": "fixed", "payload": 0.5},
                "outside_optimism": {"type": "fixed", "payload": 0.25},
                "exploration": {"type": "fixed", "payload": 0.5},
                "visibility_distance": {"type": "fixed", "payload": 4},
            },
        }

        with self.assertRaisesRegex(
            ValueError,
            r"input_spec.parameters.benefit_to_cost_ratio must satisfy "
            r"benefit_to_cost_ratio in \(0\.0,1\.0\]",
        ):
            BestResponsePolicyRun().prepare(config)

    def test_prepare_rejects_non_positive_exploration(self) -> None:
        config = self._base_config()
        config["input_spec"]["parameters"]["exploration"] = {
            "type": "fixed",
            "payload": 0.0,
        }

        with self.assertRaisesRegex(
            ValueError,
            r"input_spec.parameters.exploration must satisfy "
            r"exploration in \(0\.0,inf\)",
        ):
            BestResponsePolicyRun().prepare(config)
