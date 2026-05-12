"""Tests for multi-policy simulation runs."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.policies.myopic_payoff import MyopicPayoffPolicy
from network_creation.policies.random import RandomPolicy
from network_creation.policies.threshold_link import ThresholdLinkPolicy
from tasks.policies import prepare_policy_run
from tasks.policies.agent_parameters import build_sampled_agent_parameter_plan
from tasks.policies.agents import PolicyAgentBuilder
from tasks.policies.outputs import (
    SAMPLED_AGENT_PARAMETER_NAMES_ATTR,
    SAMPLED_AGENT_PARAMETERS_FIELD,
)
from tasks.policies.sweep import (
    build_sweep,
    collect_parameter_specs,
    parse_policy_entries,
    sample_parameters,
)
from tests.tasks.config_support import continuous, fixed, policy, task_config


def _policy_components(config: dict[str, object]):
    policy_entries = parse_policy_entries(config)
    input_spec = collect_parameter_specs(config)
    parameter_specs = input_spec["parameters"]
    parameter_samples = sample_parameters(input_spec)
    sweep = build_sweep(
        config,
        parameter_samples,
        parameter_specs=parameter_specs,
        policy_ids=tuple(policy_id for policy_id, _ in policy_entries),
    )
    sampled_parameter_plan = build_sampled_agent_parameter_plan(
        parameter_specs,
        configuration_column_by_name=sweep.column_index_by_name,
        policy_entries=policy_entries,
    )
    agent_builder = PolicyAgentBuilder(
        policy_entries=policy_entries,
        configuration_names=sweep.configuration_names,
        column_index_by_name=sweep.column_index_by_name,
        sampled_parameters_by_policy_id=sampled_parameter_plan.parameters_by_policy_id,
    )
    return parameter_samples, sweep, sampled_parameter_plan, agent_builder


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
        prepared = prepare_policy_run(self._base_config())
        parameter_samples, sweep, _, _ = _policy_components(self._base_config())

        self.assertEqual(len(prepared.inputs), 12)
        self.assertEqual(sweep.total_inputs, 12)
        self.assertEqual(sweep.configuration_values.shape, (2, 3, 2, 8))
        self.assertEqual(sweep.coordinates(0), (0, 0, 0))
        self.assertEqual(sweep.coordinates(1), (0, 0, 1))
        self.assertEqual(sweep.coordinates(2), (0, 1, 0))
        self.assertEqual(sweep.coordinates(11), (1, 2, 1))
        self.assertEqual(
            parameter_samples.parameter_names,
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
                "myopic.bridge_probability",
            ),
        )

        myopic_count_index = sweep.column_index_by_name["myopic_count"]
        random_count_index = sweep.column_index_by_name["random_count"]
        np.testing.assert_allclose(
            sweep.configuration_values[:, :, 0, myopic_count_index],
            np.asarray([[0.0, 2.0, 5.0], [0.0, 5.0, 10.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            sweep.configuration_values[:, :, 0, random_count_index],
            np.asarray([[5.0, 3.0, 0.0], [10.0, 5.0, 0.0]], dtype=np.float32),
        )

    def test_prepare_namespaces_random_policy_p_act_in_multi_policy_runs(self) -> None:
        config = self._base_config()
        config["population"]["policies"][1]["parameters"]["p_act"] = continuous(
            0.0, 1.0
        )

        parameter_samples, sweep, _, _ = _policy_components(config)

        self.assertEqual(sweep.configuration_values.shape, (2, 3, 2, 9))
        self.assertEqual(
            parameter_samples.parameter_names,
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
                "myopic.bridge_probability",
                "random.p_act",
            ),
        )

    def test_run_records_sampled_agent_parameters_for_nonzero_std(self) -> None:
        config = task_config(
            n_agents=[2, 4],
            n_actors=2,
            log2_budget=0,
            policies=[
                policy(
                    "myopic",
                    "myopic",
                    {"bridge_probability": continuous(0.0, 1.0, std=0.1)},
                ),
                policy(
                    "random",
                    "random",
                    {"p_act": continuous(0.0, 1.0, std=0.1)},
                ),
            ],
            policy_proportions={"policy": "myopic", "values": [0.5]},
            observers={
                "defaults": {
                    "sampling": {"kind": "periodic", "every": 1},
                    "aggregations": ["mean"],
                },
                "outputs": [{"name": "density", "output": "mean"}],
            },
        )

        prepared = prepare_policy_run(config)
        _, sweep, sampled_parameter_plan, agent_builder = _policy_components(config)
        configuration = sweep.configuration_at(0)
        simulation_settings = sweep.resolve(configuration)
        sampled = sampled_parameter_plan.sample(
            0,
            configuration,
            simulation_settings,
            max(sweep.n_agents_values),
            agent_builder.policy_entry_for_node,
        )
        myopic_agent = agent_builder.build_agent(
            0,
            configuration,
            simulation_settings,
            sampled_agent_parameters=sampled,
        )
        random_agent = agent_builder.build_agent(
            1,
            configuration,
            simulation_settings,
            sampled_agent_parameters=sampled,
        )

        self.assertEqual(
            prepared.result_attrs[SAMPLED_AGENT_PARAMETER_NAMES_ATTR],
            ["myopic.bridge_probability", "random.p_act"],
        )
        self.assertIn(
            {
                "name": SAMPLED_AGENT_PARAMETERS_FIELD,
                "shape": [4, 2],
                "dtype": "float32",
            },
            prepared.output_schema.metadata(),
        )
        self.assertEqual(sampled.shape, (4, 2))
        self.assertTrue(np.isfinite(sampled[0, 0]))
        self.assertTrue(np.isnan(sampled[0, 1]))
        self.assertTrue(np.isnan(sampled[1, 0]))
        self.assertTrue(np.isfinite(sampled[1, 1]))
        self.assertTrue(np.isnan(sampled[2:]).all())
        self.assertAlmostEqual(myopic_agent.policy.bridge_probability, sampled[0, 0])
        self.assertAlmostEqual(random_agent.policy.p_act, sampled[1, 1])

        first = prepared.run(0)[SAMPLED_AGENT_PARAMETERS_FIELD]
        second = prepared.run(0)[SAMPLED_AGENT_PARAMETERS_FIELD]
        np.testing.assert_allclose(first, second, equal_nan=True)

    def test_agent_builder_supports_threshold_link_policy_parameters(self) -> None:
        config = task_config(
            n_agents=3,
            log2_budget=0,
            policies=[
                policy(
                    "threshold",
                    "threshold-link",
                    {
                        "link_threshold": fixed(0.5),
                        "p_act": fixed(0.75),
                    },
                ),
            ],
            policy_proportions={"policy": "threshold", "values": [1.0]},
        )

        parameter_samples, sweep, sampled_parameter_plan, agent_builder = (
            _policy_components(config)
        )
        configuration = sweep.configuration_at(0)
        simulation_settings = sweep.resolve(configuration)
        sampled = sampled_parameter_plan.sample(
            0,
            configuration,
            simulation_settings,
            max(sweep.n_agents_values),
            agent_builder.policy_entry_for_node,
        )
        agent_policy = agent_builder.build_agent(
            0,
            configuration,
            simulation_settings,
            sampled_agent_parameters=sampled,
        ).policy

        self.assertEqual(
            parameter_samples.parameter_names,
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
                "link_threshold",
                "p_act",
            ),
        )
        self.assertIsInstance(agent_policy, ThresholdLinkPolicy)
        self.assertEqual(agent_policy.link_threshold, 0.5)
        self.assertEqual(agent_policy.p_act, 0.75)

    def test_agent_builder_assigns_policies_by_id_order(self) -> None:
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

        _, sweep, sampled_parameter_plan, agent_builder = _policy_components(config)
        configuration = sweep.configuration_at(0)
        simulation_settings = sweep.resolve(configuration)
        sampled = sampled_parameter_plan.sample(
            0,
            configuration,
            simulation_settings,
            max(sweep.n_agents_values),
            agent_builder.policy_entry_for_node,
        )

        self.assertIsInstance(
            agent_builder.build_agent(
                0,
                configuration,
                simulation_settings,
                sampled_agent_parameters=sampled,
            ).policy,
            MyopicPayoffPolicy,
        )
        self.assertIsInstance(
            agent_builder.build_agent(
                1,
                configuration,
                simulation_settings,
                sampled_agent_parameters=sampled,
            ).policy,
            MyopicPayoffPolicy,
        )
        self.assertIsInstance(
            agent_builder.build_agent(
                2,
                configuration,
                simulation_settings,
                sampled_agent_parameters=sampled,
            ).policy,
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

        _, sweep, sampled_parameter_plan, agent_builder = _policy_components(config)
        configuration = sweep.configuration_at(0)
        simulation_settings = sweep.resolve(configuration)
        sampled = sampled_parameter_plan.sample(
            0,
            configuration,
            simulation_settings,
            max(sweep.n_agents_values),
            agent_builder.policy_entry_for_node,
        )

        self.assertEqual(sweep.total_inputs, 1)
        for node_id in range(3):
            self.assertIsInstance(
                agent_builder.build_agent(
                    node_id,
                    configuration,
                    simulation_settings,
                    sampled_agent_parameters=sampled,
                ).policy,
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
