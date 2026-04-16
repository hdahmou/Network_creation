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
            ],
        }

    def test_prepare_generates_singleton_row_index_input(self) -> None:
        task = RandomPolicyRun()
        prepared = task.prepare(self._base_config())

        self.assertEqual(prepared.inputs, (0,))
        self.assertEqual(task.parameter_table.parameter_names, ("visibility_distance",))
        self.assertEqual(task.parameter_table.parameter_values.shape, (1, 1))

    def test_prepare_accepts_parameters_other_than_visibility_distance(self) -> None:
        config = self._base_config()
        config["input_spec"] = {
            "log2_budget": 0,
            "sampler_seed": 123,
            "parameters": {
                "visibility_distance": {"type": "fixed", "payload": 4},
                "bridge_probability": {"type": "fixed", "payload": 1.0},
            },
        }

        prepared = RandomPolicyRun().prepare(config)

        self.assertEqual(prepared.inputs, (0,))

    def test_prepare_rejects_non_positive_visibility_distance(self) -> None:
        for visibility_distance in (-1, 0):
            with self.subTest(visibility_distance=visibility_distance):
                config = self._base_config()
                config["input_spec"] = {
                    "log2_budget": 0,
                    "sampler_seed": 123,
                    "parameters": {
                        "visibility_distance": {
                            "type": "fixed",
                            "payload": visibility_distance,
                        },
                    },
                }

                with self.assertRaisesRegex(ValueError, "visibility_distance >= 1"):
                    RandomPolicyRun().prepare(config)

    def test_prepare_rejects_non_integer_visibility_distance(self) -> None:
        config = self._base_config()
        config["input_spec"] = {
            "log2_budget": 0,
            "sampler_seed": 123,
            "parameters": {
                "visibility_distance": {
                    "type": "fixed",
                    "payload": 1.5,
                },
            },
        }

        with self.assertRaisesRegex(ValueError, "contain only integers"):
            RandomPolicyRun().prepare(config)

    def test_prepare_supports_sampled_visibility_distance_range(self) -> None:
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

    def test_prepare_rejects_invalid_n_agents_lists(self) -> None:
        invalid_values = (
            [],
            [0, 4],
            [2.5, 4],
            [True, 4],
        )

        for invalid_n_agents in invalid_values:
            with self.subTest(n_agents=invalid_n_agents):
                config = self._base_config()
                config["simulation"]["n_agents"] = invalid_n_agents

                with self.assertRaisesRegex(ValueError, "simulation.n_agents"):
                    RandomPolicyRun().prepare(config)
