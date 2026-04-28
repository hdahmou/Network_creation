"""Tests for multi-policy simulation runs."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.policies.myopic_payoff import MyopicPayoffPolicy
from network_creation.policies.random import RandomPolicy
from tasks.policies import prepare_policy_run
from tests.tasks.config_support import continuous, fixed, policy, task_config


class PolicyRunTest(unittest.TestCase):
    @staticmethod
    def _base_config() -> dict[str, object]:
        return task_config(
            n_agents=[5, 10],
            n_actors=2,
            log2_budget=1,
            game={
                "benefit_to_cost_ratio": fixed(0.5),
                "distance_discount": fixed(0.5),
            },
            policies=[
                policy(
                    "myopic",
                    "myopic",
                    {"bridge_probability": continuous(0.0, 1.0)},
                ),
                policy("random", "random"),
            ],
            policy_proportions={"policy": "myopic", "values": [0.0, 0.5, 1.0]},
            observers={
                "defaults": {
                    "sampling": {"kind": "periodic", "every": 1},
                    "aggregations": ["mean"],
                },
                "outputs": [{"name": "density", "output": "mean"}],
            },
        )

    def test_prepare_expands_n_agents_proportions_and_parameter_rows(self) -> None:
        task = prepare_policy_run(self._base_config())
        prepared = task

        self.assertEqual(len(prepared.inputs), 12)
        self.assertEqual(task.sweep.total_inputs, 12)
        self.assertEqual(task.sweep.configuration_values.shape, (2, 3, 2, 8))
        self.assertEqual(task.sweep.coordinates(0), (0, 0, 0))
        self.assertEqual(task.sweep.coordinates(1), (0, 0, 1))
        self.assertEqual(task.sweep.coordinates(2), (0, 1, 0))
        self.assertEqual(task.sweep.coordinates(11), (1, 2, 1))
        self.assertEqual(
            task.parameter_samples.parameter_names,
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
                "myopic.bridge_probability",
            ),
        )

        myopic_count_index = task.sweep.column_index_by_name[
            "myopic_count"
        ]
        random_count_index = task.sweep.column_index_by_name[
            "random_count"
        ]
        np.testing.assert_allclose(
            task.sweep.configuration_values[:, :, 0, myopic_count_index],
            np.asarray([[0.0, 2.0, 5.0], [0.0, 5.0, 10.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            task.sweep.configuration_values[:, :, 0, random_count_index],
            np.asarray([[5.0, 3.0, 0.0], [10.0, 5.0, 0.0]], dtype=np.float32),
        )

    def test_build_agent_assigns_policies_by_id_order(self) -> None:
        config = self._base_config()
        config["sampling"]["log2_budget"] = 0
        config["population"]["n_agents"] = 5
        config["population"]["policy_proportions"] = {
            "policy": "myopic",
            "values": [0.5],
        }
        config["population"]["policies"][0]["parameters"][
            "bridge_probability"
        ] = fixed(1.0)

        task = prepare_policy_run(config)
        configuration = task.sweep.configuration_at(0)
        simulation_settings = task.sweep.resolve(configuration)

        self.assertIsInstance(
            task.build_agent(0, configuration, simulation_settings).policy,
            MyopicPayoffPolicy,
        )
        self.assertIsInstance(
            task.build_agent(1, configuration, simulation_settings).policy,
            MyopicPayoffPolicy,
        )
        self.assertIsInstance(
            task.build_agent(2, configuration, simulation_settings).policy,
            RandomPolicy,
        )

    def test_single_policy_assigns_every_agent_to_the_same_policy(self) -> None:
        config = self._base_config()
        config["sampling"]["log2_budget"] = 0
        config["population"]["n_agents"] = 3
        config["population"]["policies"] = [
            policy("random", "random"),
        ]
        config["population"]["policy_proportions"] = {
            "policy": "random",
            "values": [1.0],
        }

        task = prepare_policy_run(config)
        configuration = task.sweep.configuration_at(0)
        simulation_settings = task.sweep.resolve(configuration)

        self.assertEqual(task.sweep.total_inputs, 1)
        for node_id in range(3):
            self.assertIsInstance(
                task.build_agent(node_id, configuration, simulation_settings).policy,
                RandomPolicy,
            )

    def test_prepared_run_is_deterministic_for_fixed_seed(self) -> None:
        config = self._base_config()
        config["sampling"]["log2_budget"] = 0
        config["population"]["n_agents"] = 4
        config["population"]["policy_proportions"] = {
            "policy": "myopic",
            "values": [0.5],
        }
        config["population"]["policies"][0]["parameters"][
            "bridge_probability"
        ] = fixed(1.0)
        prepared = prepare_policy_run(config)

        first = prepared.run(prepared.inputs[0])
        second = prepared.run(prepared.inputs[0])

        self.assertEqual(set(first), set(second))
        for field_name in first:
            np.testing.assert_allclose(first[field_name], second[field_name])

    def test_prepare_rejects_invalid_policy_count(self) -> None:
        cases = []

        no_policies = self._base_config()
        no_policies["population"]["policies"] = []
        cases.append((no_policies, "at least one policy"))

        too_many_policies = self._base_config()
        too_many_policies["population"]["policies"] = [
            *too_many_policies["population"]["policies"],
            policy("random_2", "random"),
        ]
        cases.append((too_many_policies, "at most two policies"))

        for config, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    prepare_policy_run(config)


if __name__ == "__main__":
    unittest.main()
