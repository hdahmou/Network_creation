"""Tests for policy task preparation."""

from __future__ import annotations

import unittest

from tasks.policies import prepare_policy_run
from tests.tasks.config_support import (
    continuous,
    fixed,
    policy,
    sampled_discrete_range,
    task_config,
)


class RandomPolicyPreparationTest(unittest.TestCase):
    @staticmethod
    def _base_config() -> dict[str, object]:
        return task_config(
            policies=[policy("random", "random")],
            observers={
                "defaults": {
                    "sampling": {"kind": "periodic", "every": 1},
                    "aggregations": ["mean"],
                },
                "outputs": [{"name": "density", "output": "mean"}],
            },
        )

    def test_prepare_generates_singleton_row_index_input(self) -> None:
        task = prepare_policy_run(self._base_config())
        prepared = task

        self.assertEqual(prepared.inputs, (0,))
        self.assertEqual(
            task.parameter_samples.parameter_names,
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
            ),
        )
        self.assertEqual(task.parameter_samples.parameter_values.shape, (1, 3))

    def test_prepare_collects_explicit_p_act_parameter(self) -> None:
        config = self._base_config()
        config["sampling"]["log2_budget"] = 1
        config["population"]["policies"][0]["parameters"]["p_act"] = continuous(
            0.0, 1.0
        )

        task = prepare_policy_run(config)

        self.assertEqual(task.inputs, (0, 1))
        self.assertEqual(
            task.parameter_samples.parameter_names,
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
                "p_act",
            ),
        )
        self.assertEqual(task.parameter_samples.parameter_values.shape, (2, 4))

    def test_prepare_accepts_parameters_other_than_visibility_distance(self) -> None:
        config = self._base_config()
        config["population"]["extra_random_knob"] = fixed(1.0)

        prepared = prepare_policy_run(config)

        self.assertEqual(prepared.inputs, (0,))

    def test_prepare_supports_sampled_visibility_distance_range(self) -> None:
        config = self._base_config()
        config["sampling"]["log2_budget"] = 1
        config["population"]["visibility_distance"] = sampled_discrete_range(
            1, 4
        )

        prepared = prepare_policy_run(config)

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
                config["population"]["n_agents"] = invalid_n_agents

                with self.assertRaisesRegex(ValueError, "population.n_agents"):
                    prepare_policy_run(config)
