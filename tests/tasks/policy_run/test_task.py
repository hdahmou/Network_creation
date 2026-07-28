"""Tests for multi-policy simulation runs."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.policies.best_response import BestResponsePolicy
from network_creation.policies.myopic_payoff import MyopicPayoffPolicy
from network_creation.policies.random import RandomPolicy
from network_creation.policies.threshold_link import ThresholdLinkPolicy
from tasks.policies import prepare_policy_run
from tasks.policies.agent_parameters import build_sampled_agent_parameter_plan
from tasks.policies.agent_attributes import build_agent_attribute_plan
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
from tests.tasks.config_support import (
    continuous,
    discrete_offset_heterogeneity,
    fixed,
    normal_heterogeneity,
    policy,
    sampled_discrete_range,
    scaled_other_agents,
    task_config,
)


class AlternatingPolicyAgentBuilder(PolicyAgentBuilder):
    def policy_entry_for_node(
        self,
        node_id: int,
        configuration: np.ndarray,
    ):
        del configuration
        return self.policy_entries[node_id % len(self.policy_entries)]


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
        input_spec["policy_parameters"],
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


def _agent_attribute_components(config: dict[str, object]):
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
    plan = build_agent_attribute_plan(
        input_spec["agent_attribute_definitions"],
        configuration_column_by_name=sweep.column_index_by_name,
        policy_entries=policy_entries,
        n_agents_values=sweep.n_agents_values,
        n_actors=sweep.n_actors,
    )
    builder = PolicyAgentBuilder(
        policy_entries=policy_entries,
        configuration_names=sweep.configuration_names,
        column_index_by_name=sweep.column_index_by_name,
        sampled_parameters_by_policy_id={
            policy_id: () for policy_id, _ in policy_entries
        },
    )
    return parameter_samples, sweep, plan, builder


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
                "outputs": [{"name": "density"}],
            },
        )

    def test_prepare_expands_n_agents_proportions_and_parameter_rows(self) -> None:
        prepared = prepare_policy_run(self._base_config())
        parameter_samples, sweep, _, _ = _policy_components(self._base_config())

        self.assertEqual(len(prepared.inputs), 12)
        self.assertEqual(sweep.total_inputs, 12)
        self.assertEqual(sweep.configuration_values.shape, (2, 3, 2, 9))
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
                "agent.activation_weight",
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

    def test_prepare_balances_policy_counts_over_parameter_rows(self) -> None:
        config = task_config(
            n_agents=[2, 4],
            n_actors=1,
            log2_budget=3,
            policies=[
                policy("best_response", "random"),
                policy("random", "random"),
            ],
            policy_counts={"policy": "best_response", "kind": "balanced"},
        )

        prepared = prepare_policy_run(config)
        _, sweep, _, _ = _policy_components(config)

        self.assertEqual(len(prepared.inputs), 16)
        self.assertEqual(sweep.configuration_values.shape[:3], (2, 1, 8))
        self.assertEqual(prepared.result_attrs["sweep_layout"], "sampled_composition")
        best_response_count = sweep.column_index_by_name["best_response_count"]
        random_count = sweep.column_index_by_name["random_count"]
        for n_agents_index, n_agents in enumerate((2, 4)):
            counts = sweep.configuration_values[
                n_agents_index, 0, :, best_response_count
            ]
            levels, frequencies = np.unique(counts, return_counts=True)
            np.testing.assert_array_equal(levels, np.arange(n_agents + 1))
            self.assertLessEqual(int(frequencies.max() - frequencies.min()), 1)
            np.testing.assert_array_equal(
                counts
                + sweep.configuration_values[
                    n_agents_index, 0, :, random_count
                ],
                np.full(8, n_agents, dtype=np.float32),
            )

        repeated = prepare_policy_run(config)
        np.testing.assert_array_equal(
            prepared.result_arrays["parameters"],
            repeated.result_arrays["parameters"],
        )

    def test_balanced_policy_counts_require_enough_rows(self) -> None:
        config = task_config(
            n_agents=4,
            log2_budget=2,
            policies=[policy("first", "random"), policy("second", "random")],
            policy_counts={"policy": "first", "kind": "balanced"},
        )

        with self.assertRaisesRegex(ValueError, r"at least max\(n_agents\) \+ 1"):
            prepare_policy_run(config)

    def test_prepare_namespaces_random_policy_activation_weight(self) -> None:
        config = self._base_config()
        config["simulator"]["actor_scheduler"]["n_actors"] = 1
        config["population"]["policies"][1]["agent_parameters"] = {
            "activation_weight": continuous(0.01, 1.0)
        }

        parameter_samples, sweep, _, _ = _policy_components(config)

        self.assertEqual(sweep.configuration_values.shape, (2, 3, 2, 10))
        self.assertEqual(
            parameter_samples.parameter_names,
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
                "myopic.bridge_probability",
                "agent.activation_weight",
                "random.agent.activation_weight",
            ),
        )

    def test_policy_assignment_uses_policy_selection_logic(self) -> None:
        builder = AlternatingPolicyAgentBuilder(
            policy_entries=(("first", "random"), ("second", "random")),
            configuration_names=("n_agents",),
            column_index_by_name={"n_agents": 0},
            sampled_parameters_by_policy_id={"first": (), "second": ()},
        )

        assignment = builder.policy_assignment_for_configuration(
            np.asarray([5.0], dtype=np.float32),
            n_agents=5,
        )

        self.assertEqual(assignment.policy_ids, ("first", "second"))
        np.testing.assert_array_equal(
            assignment.policy_slot_by_agent,
            np.asarray([0, 1, 0, 1, 0], dtype=np.int32),
        )

    def test_run_records_policy_welfare_observer(self) -> None:
        config = task_config(
            n_agents=4,
            n_actors=2,
            log2_budget=0,
            n_episodes=1,
            policies=[
                policy("first", "random"),
                policy("second", "random"),
            ],
            policy_proportions={"policy": "first", "values": [0.5]},
            observers={
                "defaults": {
                    "sampling": {"kind": "periodic", "every": 1},
                    "aggregations": ["mean"],
                },
                "outputs": [{"name": "policy_welfare"}],
            },
        )

        output = prepare_policy_run(config).run(0)

        self.assertEqual(output["policy_welfare.mean"].shape, (4, 2))
        np.testing.assert_allclose(output["policy_welfare.mean"][0], 0.0)
        self.assertTrue(np.isfinite(output["policy_welfare.mean"]).all())

    def test_run_records_sampled_agent_parameters_for_normal_heterogeneity(self) -> None:
        config = task_config(
            n_agents=[2, 4],
            n_actors=2,
            log2_budget=0,
            policies=[
                policy(
                    "myopic",
                    "myopic",
                    {
                        "bridge_probability": continuous(
                            0.0,
                            1.0,
                            heterogeneity=normal_heterogeneity(0.1),
                        )
                    },
                ),
                policy(
                    "random",
                    "random",
                ),
            ],
            policy_proportions={"policy": "myopic", "values": [0.5]},
            observers={
                "defaults": {
                    "sampling": {"kind": "periodic", "every": 1},
                    "aggregations": ["mean"],
                },
                "outputs": [{"name": "density"}],
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
            activation_weight=1.0,
        )
        random_agent = agent_builder.build_agent(
            1,
            configuration,
            simulation_settings,
            sampled_agent_parameters=sampled,
            activation_weight=1.0,
        )

        self.assertEqual(
            prepared.result_attrs[SAMPLED_AGENT_PARAMETER_NAMES_ATTR],
            ["myopic.bridge_probability"],
        )
        self.assertIn(
            {
                "name": SAMPLED_AGENT_PARAMETERS_FIELD,
                "shape": [4, 1],
                "dtype": "float32",
            },
            prepared.output_schema.metadata(),
        )
        self.assertEqual(sampled.shape, (4, 1))
        self.assertTrue(np.isfinite(sampled[0, 0]))
        self.assertTrue(np.isnan(sampled[1, 0]))
        self.assertTrue(np.isnan(sampled[2:]).all())
        self.assertAlmostEqual(myopic_agent.policy.bridge_probability, sampled[0, 0])
        self.assertIsInstance(random_agent.policy, RandomPolicy)

        first = prepared.run(0)[SAMPLED_AGENT_PARAMETERS_FIELD]
        second = prepared.run(0)[SAMPLED_AGENT_PARAMETERS_FIELD]
        np.testing.assert_allclose(first, second, equal_nan=True)

    def test_visibility_distance_uses_three_point_agent_heterogeneity(self) -> None:
        config = task_config(
            n_agents=24,
            n_actors=1,
            log2_budget=0,
            visibility_distance=sampled_discrete_range(
                2,
                5,
                heterogeneity=discrete_offset_heterogeneity((-1, 0, 1)),
            ),
            policies=[policy("random", "random")],
        )
        prepared = prepare_policy_run(config)
        _, sweep, sampled_parameter_plan, agent_builder = _policy_components(config)
        configuration = sweep.configuration_at(0)
        settings = sweep.resolve(configuration)
        sampled = sampled_parameter_plan.sample(
            0,
            configuration,
            settings,
            max(sweep.n_agents_values),
            agent_builder.policy_entry_for_node,
        )
        center = configuration[sweep.column_index_by_name["visibility_distance"]]
        realized = sampled[:, 0]

        self.assertEqual(
            prepared.result_attrs[SAMPLED_AGENT_PARAMETER_NAMES_ATTR],
            ["visibility_distance"],
        )
        np.testing.assert_array_equal(realized, np.rint(realized))
        self.assertEqual(set(realized - center), {-1.0, 0.0, 1.0})
        self.assertGreaterEqual(float(realized.min()), 1.0)
        self.assertLessEqual(float(realized.max()), 6.0)
        for node_id in range(settings.n_agents):
            agent = agent_builder.build_agent(
                node_id,
                configuration,
                settings,
                sampled_agent_parameters=sampled,
                activation_weight=1.0,
            )
            self.assertEqual(agent.visibility_distance, int(realized[node_id]))

        np.testing.assert_array_equal(
            prepared.run(0)[SAMPLED_AGENT_PARAMETERS_FIELD][:, 0],
            realized,
        )

    def test_zero_normal_heterogeneity_saves_agent_level_centers(self) -> None:
        config = task_config(
            n_agents=3,
            log2_budget=0,
            policies=[
                policy(
                    "myopic",
                    "myopic",
                    {
                        "bridge_probability": continuous(
                            0.25,
                            0.75,
                            heterogeneity=normal_heterogeneity(0.0),
                        )
                    },
                )
            ],
        )
        prepared = prepare_policy_run(config)
        _, sweep, sampled_parameter_plan, agent_builder = _policy_components(config)
        configuration = sweep.configuration_at(0)
        settings = sweep.resolve(configuration)
        sampled = sampled_parameter_plan.sample(
            0,
            configuration,
            settings,
            max(sweep.n_agents_values),
            agent_builder.policy_entry_for_node,
        )
        center = configuration[sweep.column_index_by_name["bridge_probability"]]

        self.assertEqual(
            prepared.result_attrs[SAMPLED_AGENT_PARAMETER_NAMES_ATTR],
            ["bridge_probability"],
        )
        np.testing.assert_allclose(sampled[:, 0], center)
        np.testing.assert_allclose(
            prepared.run(0)[SAMPLED_AGENT_PARAMETERS_FIELD][:, 0],
            center,
        )

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
            activation_weight=1.0,
        ).policy

        self.assertEqual(
            parameter_samples.parameter_names,
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
                "link_threshold",
                "p_act",
                "agent.activation_weight",
            ),
        )
        self.assertIsInstance(agent_policy, ThresholdLinkPolicy)
        self.assertEqual(agent_policy.link_threshold, 0.5)
        self.assertEqual(agent_policy.p_act, 0.75)

    def test_scaled_other_agents_resolves_constant_share_across_sizes(self) -> None:
        config = task_config(
            n_agents=[10, 50, 100],
            n_actors=1,
            log2_budget=0,
            n_episodes=1,
            max_steps=1,
            agent_parameters={"activation_weight": fixed(1.0)},
            policies=[
                policy(
                    "threshold_link",
                    "threshold-link",
                    {
                        "link_threshold": fixed(0.5),
                        "p_act": fixed(1.0),
                    },
                    agent_parameters={
                        "activation_weight": scaled_other_agents(fixed(0.1))
                    },
                ),
                policy(
                    "best_response",
                    "best-response",
                    {
                        "outside_optimism": fixed(0.25),
                        "exploration": fixed(0.01),
                    },
                ),
            ],
            policy_counts={"policy": "threshold_link", "values": [1]},
        )
        parameter_samples, sweep, plan, builder = _agent_attribute_components(config)

        self.assertEqual(
            parameter_samples.parameter_names[-2:],
            ("agent.activation_weight", "threshold_link.agent.activation_share"),
        )
        for input_index, expected_weight in enumerate((1.0, 49.0 / 9.0, 11.0)):
            configuration = sweep.configuration_at(input_index)
            settings = sweep.resolve(configuration)
            resolved = plan.resolve(
                configuration,
                settings,
                parameter_row_index=0,
                max_n_agents=100,
                policy_for_node=builder.policy_entry_for_node,
            )

            self.assertAlmostEqual(
                float(resolved.activation_weights[0]),
                expected_weight,
                places=5,
            )
            np.testing.assert_allclose(resolved.activation_weights[1:], 1.0)
            probability = resolved.activation_weights[0] / np.sum(
                resolved.activation_weights
            )
            self.assertAlmostEqual(float(probability), 0.1, places=6)

    def test_sampled_activation_weights_pair_unchanged_agents_by_parameter_row(
        self,
    ) -> None:
        config = task_config(
            n_agents=5,
            n_actors=1,
            log2_budget=0,
            n_episodes=1,
            max_steps=1,
            agent_parameters={
                "activation_weight": continuous(
                    0.8,
                    1.2,
                    heterogeneity=normal_heterogeneity(0.05),
                )
            },
            policies=[
                policy(
                    "threshold_link",
                    "threshold-link",
                    {
                        "link_threshold": fixed(0.5),
                        "p_act": fixed(1.0),
                    },
                    agent_parameters={
                        "activation_weight": scaled_other_agents(
                            continuous(
                                0.08,
                                0.12,
                                heterogeneity=normal_heterogeneity(0.01),
                            )
                        )
                    },
                ),
                policy(
                    "best_response",
                    "random",
                ),
            ],
            policy_counts={"policy": "threshold_link", "values": [0, 1]},
        )
        _, sweep, plan, builder = _agent_attribute_components(config)
        baseline_configuration = sweep.configuration_at(0)
        treatment_configuration = sweep.configuration_at(1)
        settings = sweep.resolve(baseline_configuration)

        baseline = plan.resolve(
            baseline_configuration,
            settings,
            parameter_row_index=0,
            max_n_agents=5,
            policy_for_node=builder.policy_entry_for_node,
        )
        treatment = plan.resolve(
            treatment_configuration,
            settings,
            parameter_row_index=0,
            max_n_agents=5,
            policy_for_node=builder.policy_entry_for_node,
        )

        np.testing.assert_allclose(
            baseline.activation_weights[1:],
            treatment.activation_weights[1:],
        )
        self.assertNotEqual(
            float(baseline.activation_weights[0]),
            float(treatment.activation_weights[0]),
        )
        self.assertEqual(
            set(plan.sampled_names),
            {
                "agent.activation_weight",
                "threshold_link.agent.activation_weight",
            },
        )

        prepared = prepare_policy_run(config)
        baseline_output = prepared.run(0)
        output = prepared.run(1)
        self.assertEqual(
            set(prepared.result_attrs[SAMPLED_AGENT_PARAMETER_NAMES_ATTR]),
            set(plan.sampled_names),
        )
        self.assertEqual(
            output[SAMPLED_AGENT_PARAMETERS_FIELD].shape,
            (5, 2),
        )
        sampled_names = prepared.result_attrs[SAMPLED_AGENT_PARAMETER_NAMES_ATTR]
        for name in ("agent.activation_weight",):
            column = sampled_names.index(name)
            np.testing.assert_allclose(
                baseline_output[SAMPLED_AGENT_PARAMETERS_FIELD][1:, column],
                output[SAMPLED_AGENT_PARAMETERS_FIELD][1:, column],
            )

    def test_agent_attribute_validation_rejects_invalid_specs(self) -> None:
        cases = (
            (
                {"activation_weight": fixed(0.0)},
                "activation_weight.*> 0",
            ),
            (
                {
                    "activation_weight": scaled_other_agents(fixed(1.0)),
                },
                "activation_share.*\\(0, 1\\)",
            ),
            (
                {"unknown": fixed(1.0)},
                "unsupported agent parameter",
            ),
        )
        for agent_parameters, message in cases:
            with self.subTest(agent_parameters=agent_parameters):
                config = task_config(
                    n_agents=3,
                    n_actors=1,
                    agent_parameters=agent_parameters,
                )
                with self.assertRaisesRegex(ValueError, message):
                    prepare_policy_run(config)

        single_agent_config = task_config(
            n_agents=1,
            n_actors=1,
            policies=[
                policy(
                    "random",
                    "random",
                    agent_parameters={
                        "activation_weight": scaled_other_agents(fixed(0.1))
                    },
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "requires n_agents >= 2"):
            prepare_policy_run(single_agent_config)

        multi_actor_config = task_config(
            n_agents=3,
            n_actors=2,
            policies=[
                policy(
                    "weighted",
                    "random",
                    agent_parameters={
                        "activation_weight": scaled_other_agents(fixed(0.1))
                    },
                ),
                policy("neutral", "random"),
            ],
            policy_proportions={"policy": "weighted", "values": [0.5]},
        )
        with self.assertRaisesRegex(ValueError, "unequal.*n_actors == 1"):
            prepare_policy_run(multi_actor_config)

        uniform_multi_actor_config = task_config(
            n_agents=3,
            n_actors=2,
            agent_parameters={
                "activation_weight": scaled_other_agents(fixed(0.1))
            },
        )
        prepare_policy_run(uniform_multi_actor_config)

    def test_fully_shadowed_agent_default_is_not_a_sampled_parameter(self) -> None:
        config = task_config(
            n_agents=3,
            n_actors=1,
            agent_parameters={"activation_weight": fixed(2.0)},
            policies=[
                policy(
                    "first",
                    "random",
                    agent_parameters={"activation_weight": fixed(1.0)},
                ),
                policy(
                    "second",
                    "random",
                    agent_parameters={"activation_weight": fixed(3.0)},
                ),
            ],
            policy_proportions={"policy": "first", "values": [0.5]},
        )

        parameter_samples, _, _, _ = _agent_attribute_components(config)

        self.assertNotIn("agent.activation_weight", parameter_samples.parameter_names)
        self.assertIn("first.agent.activation_weight", parameter_samples.parameter_names)
        self.assertIn("second.agent.activation_weight", parameter_samples.parameter_names)

    def test_policy_counts_assign_exact_target_count_across_n_agents(self) -> None:
        config = task_config(
            n_agents=[10, 50],
            n_actors=2,
            log2_budget=0,
            policies=[
                policy(
                    "threshold_link",
                    "threshold-link",
                    {
                        "link_threshold": fixed(0.5),
                        "p_act": fixed(1.0),
                    },
                ),
                policy(
                    "best_response",
                    "best-response",
                    {
                        "outside_optimism": fixed(0.25),
                        "exploration": fixed(0.01),
                    },
                ),
            ],
            policy_counts={"policy": "threshold_link", "values": [1]},
        )

        prepared = prepare_policy_run(config)
        _, sweep, sampled_parameter_plan, agent_builder = _policy_components(config)

        self.assertEqual(prepared.inputs, (0, 1))
        self.assertEqual(
            sweep.configuration_names[:3],
            ("n_agents", "threshold_link_count", "best_response_count"),
        )
        np.testing.assert_allclose(
            sweep.configuration_values[
                :,
                0,
                0,
                sweep.column_index_by_name["threshold_link_count"],
            ],
            np.asarray([1.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            sweep.configuration_values[
                :,
                0,
                0,
                sweep.column_index_by_name["best_response_count"],
            ],
            np.asarray([9.0, 49.0], dtype=np.float32),
        )

        for input_index, last_agent_id in ((0, 9), (1, 49)):
            configuration = sweep.configuration_at(input_index)
            simulation_settings = sweep.resolve(configuration)
            sampled = sampled_parameter_plan.sample(
                input_index,
                configuration,
                simulation_settings,
                max(sweep.n_agents_values),
                agent_builder.policy_entry_for_node,
            )

            first_agent = agent_builder.build_agent(
                0,
                configuration,
                simulation_settings,
                sampled_agent_parameters=sampled,
                activation_weight=1.0,
            )
            last_agent = agent_builder.build_agent(
                last_agent_id,
                configuration,
                simulation_settings,
                sampled_agent_parameters=sampled,
                activation_weight=1.0,
            )
            self.assertIsInstance(first_agent.policy, ThresholdLinkPolicy)
            self.assertIsInstance(last_agent.policy, BestResponsePolicy)

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
                activation_weight=1.0,
            ).policy,
            MyopicPayoffPolicy,
        )
        self.assertIsInstance(
            agent_builder.build_agent(
                1,
                configuration,
                simulation_settings,
                sampled_agent_parameters=sampled,
                activation_weight=1.0,
            ).policy,
            MyopicPayoffPolicy,
        )
        self.assertIsInstance(
            agent_builder.build_agent(
                2,
                configuration,
                simulation_settings,
                sampled_agent_parameters=sampled,
                activation_weight=1.0,
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
                    activation_weight=1.0,
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
