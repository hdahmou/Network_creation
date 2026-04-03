"""Tests for random-policy task preparation."""

from __future__ import annotations

import unittest

from tasks.policies import RandomPolicyRun


class RandomPolicyPreparationTest(unittest.TestCase):
    @staticmethod
    def _base_config() -> dict[str, object]:
        return {
            "input_spec": {
                "log2_budget": 0,
                "sampler_seed": 123,
                "parameters": {},
            },
            "simulation": {
                "n_agents": 4,
                "radius": 2,
                "mode": "SIMULTANEOUS",
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
                        "observation_frequency": 1,
                        "aggregations": ["mean"],
                    },
                },
            ],
        }

    def test_prepare_generates_singleton_row_index_input(self) -> None:
        task = RandomPolicyRun()
        prepared = task.prepare(self._base_config())

        self.assertEqual(prepared.inputs, (0,))
        self.assertEqual(task.parameter_table.parameter_names, ())
        self.assertEqual(task.parameter_table.parameter_values.shape, (1, 0))

    def test_prepare_rejects_non_empty_parameters(self) -> None:
        config = self._base_config()
        config["input_spec"] = {
            "log2_budget": 0,
            "sampler_seed": 123,
            "parameters": {"bridge_probability": 1.0},
        }

        with self.assertRaisesRegex(ValueError, "parameters"):
            RandomPolicyRun().prepare(config)

    def test_prepare_rejects_non_singleton_budget(self) -> None:
        config = self._base_config()
        config["input_spec"] = {
            "log2_budget": 1,
            "sampler_seed": 123,
            "parameters": {},
        }

        with self.assertRaisesRegex(ValueError, "log2_budget"):
            RandomPolicyRun().prepare(config)
