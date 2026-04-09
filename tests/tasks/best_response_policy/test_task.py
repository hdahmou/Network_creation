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
                    "bridge_value_scale": {"type": "fixed", "payload": 1.0},
                    "inertia_term": {"type": "fixed", "payload": 0.0},
                    "temperature": {"type": "fixed", "payload": 1.0},
                    "visibility_fraction": {"type": "fixed", "payload": 1.0},
                },
            },
            "simulation": {
                "n_agents": 4,
                "actor_scheduler": {"fraction": 1.0},
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
                        "benefit_to_cost_ratio": 0.1,
                        "distance_discount": 0.5,
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
                "bridge_value_scale",
                "inertia_term",
                "temperature",
                "visibility_fraction",
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

    def test_prepare_supports_multiple_parameter_samples(self) -> None:
        config = self._base_config()
        config["input_spec"] = {
            "log2_budget": 1,
            "sampler_seed": 123,
            "parameters": {
                "benefit_to_cost_ratio": {"type": "fixed", "payload": 0.1},
                "distance_discount": {"type": "fixed", "payload": 0.5},
                "bridge_value_scale": {
                    "type": "continuous",
                    "payload": {"min": 0.2, "max": 1.0},
                },
                "inertia_term": {
                    "type": "continuous",
                    "payload": {"min": 0.0, "max": 0.5},
                },
                "temperature": {"type": "fixed", "payload": 1.0},
                "visibility_fraction": {"type": "fixed", "payload": 1.0},
            },
        }
        prepared = BestResponsePolicyRun().prepare(config)

        self.assertEqual(prepared.inputs, (0, 1))
        for sample_idx in prepared.inputs:
            output = prepared.run(sample_idx)
            self.assertEqual(set(output), {"density.mean", "normalized_welfare.mean"})
