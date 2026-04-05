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
                "parameters": {"radius": {"type": "fixed", "payload": 2}},
            },
            "simulation": {
                "n_agents": 4,
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
                        "observation_sampling": {"kind": "periodic", "every": 1},
                        "aggregations": ["mean"],
                    },
                },
            ],
        }

    def test_prepare_generates_singleton_row_index_input(self) -> None:
        task = RandomPolicyRun()
        prepared = task.prepare(self._base_config())

        self.assertEqual(prepared.inputs, (0,))
        self.assertEqual(task.parameter_table.parameter_names, ("radius",))
        self.assertEqual(task.parameter_table.parameter_values.shape, (1, 1))

    def test_prepare_accepts_parameters_other_than_radius(self) -> None:
        config = self._base_config()
        config["input_spec"] = {
            "log2_budget": 0,
            "sampler_seed": 123,
            "parameters": {
                "radius": {"type": "fixed", "payload": 2},
                "bridge_probability": {"type": "fixed", "payload": 1.0},
            },
        }

        prepared = RandomPolicyRun().prepare(config)

        self.assertEqual(prepared.inputs, (0,))

    def test_prepare_supports_sampled_radius_range(self) -> None:
        config = self._base_config()
        config["input_spec"] = {
            "log2_budget": 1,
            "sampler_seed": 123,
            "parameters": {
                "radius": {
                    "type": "sampled_discrete_range",
                    "payload": {"min": 1, "max": 2},
                }
            },
        }

        prepared = RandomPolicyRun().prepare(config)

        self.assertEqual(prepared.inputs, (0, 1))
