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
                "parameters": {"visibility_fraction": {"type": "fixed", "payload": 1.0}},
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
            ],
        }

    def test_prepare_generates_singleton_row_index_input(self) -> None:
        task = RandomPolicyRun()
        prepared = task.prepare(self._base_config())

        self.assertEqual(prepared.inputs, (0,))
        self.assertEqual(task.parameter_table.parameter_names, ("visibility_fraction",))
        self.assertEqual(task.parameter_table.parameter_values.shape, (1, 1))

    def test_prepare_accepts_parameters_other_than_visibility_fraction(self) -> None:
        config = self._base_config()
        config["input_spec"] = {
            "log2_budget": 0,
            "sampler_seed": 123,
            "parameters": {
                "visibility_fraction": {"type": "fixed", "payload": 1.0},
                "bridge_probability": {"type": "fixed", "payload": 1.0},
            },
        }

        prepared = RandomPolicyRun().prepare(config)

        self.assertEqual(prepared.inputs, (0,))

    def test_prepare_rejects_visibility_fraction_outside_unit_interval(self) -> None:
        for visibility_fraction in (-0.1, 1.1):
            with self.subTest(visibility_fraction=visibility_fraction):
                config = self._base_config()
                config["input_spec"] = {
                    "log2_budget": 0,
                    "sampler_seed": 123,
                    "parameters": {
                        "visibility_fraction": {
                            "type": "fixed",
                            "payload": visibility_fraction,
                        },
                    },
                }

                with self.assertRaisesRegex(ValueError, "0 <= visibility_fraction <= 1"):
                    RandomPolicyRun().prepare(config)

    def test_prepare_supports_sampled_visibility_fraction_range(self) -> None:
        config = self._base_config()
        config["input_spec"] = {
            "log2_budget": 1,
            "sampler_seed": 123,
            "parameters": {
                "visibility_fraction": {
                    "type": "continuous",
                    "payload": {"min": 0.0, "max": 1.0},
                }
            },
        }

        prepared = RandomPolicyRun().prepare(config)

        self.assertEqual(prepared.inputs, (0, 1))
