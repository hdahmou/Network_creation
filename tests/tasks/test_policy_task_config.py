"""Tests for policy task config handling."""

from __future__ import annotations

from typing import Any, cast
import unittest

import numpy as np

from tasks.policies import prepare_policy_run
from tasks.policies.sweep import (
    build_sweep,
    collect_parameter_specs,
    parse_policy_entries,
    sample_parameters,
)
from tests.tasks.config_support import (
    activation_opportunities_per_agent,
    raw_simulator_steps,
    task_config,
)


class PolicyTaskConfigTest(unittest.TestCase):
    def test_prepare_policy_run_uses_root_policy_config(self) -> None:
        policy_task = prepare_policy_run(
            {
                "sampling": {"log2_budget": 0, "seed": 1},
                "simulator": {
                    "seed": 1,
                    "budget": {
                        "kind": "raw_simulator_steps",
                        "value": 1,
                    },
                    "n_episodes": 1,
                    "stagnation_patience": None,
                    "actor_scheduler": {"n_actors": 1},
                },
                "game": {
                    "benefit_to_cost_ratio": {"kind": "fixed", "value": 0.5},
                    "distance_discount": {"kind": "fixed", "value": 0.5},
                },
                "population": {
                    "n_agents": 2,
                    "visibility_distance": {"kind": "fixed", "value": 1},
                    "agent_parameters": {
                        "activation_weight": {"kind": "fixed", "value": 1.0}
                    },
                    "policies": [
                        {"id": "random", "kind": "random", "parameters": {}},
                    ],
                    "policy_proportions": {
                        "policy": "random",
                        "values": [1.0],
                    },
                },
                "observers": {
                    "defaults": {
                        "sampling": {"kind": "periodic", "every": 1},
                        "aggregations": ["mean"],
                    },
                    "outputs": [{"name": "density"}],
                },
            }
        )

        self.assertEqual(policy_task.inputs, (0,))
        self.assertEqual(
            policy_task.output_schema.metadata(),
            [{"name": "density.mean", "shape": [2], "dtype": "float32"}],
        )

    @staticmethod
    def _build_sweep(config: dict[str, object]):
        input_spec = collect_parameter_specs(config)
        parameter_samples = sample_parameters(input_spec)
        policy_entries = parse_policy_entries(config)
        return build_sweep(
            config,
            parameter_samples,
            parameter_specs=input_spec["parameters"],
            policy_ids=tuple(policy_id for policy_id, _ in policy_entries),
        )

    def test_activation_budget_resolves_population_specific_raw_horizons(
        self,
    ) -> None:
        config = task_config(
            n_agents=[10, 50, 100],
            n_actors=1,
            budget=activation_opportunities_per_agent(40),
        )

        sweep = self._build_sweep(config)

        self.assertEqual(sweep.budget_kind, "activation_opportunities_per_agent")
        self.assertEqual(sweep.budget_value, 40.0)
        self.assertEqual(sweep.raw_step_horizons, (400, 2000, 4000))
        self.assertEqual(sweep.storage_max_steps, 4000)
        self.assertEqual(
            tuple(
                sweep.resolve(sweep.configuration_values[index, 0, 0]).max_steps
                for index in range(3)
            ),
            (400, 2000, 4000),
        )

    def test_activation_budget_rounds_each_raw_horizon_up(self) -> None:
        config = task_config(
            n_agents=[3, 5],
            n_actors=2,
            budget=activation_opportunities_per_agent(2.5),
        )

        sweep = self._build_sweep(config)
        prepared = prepare_policy_run(config)

        self.assertEqual(sweep.raw_step_horizons, (4, 7))
        self.assertEqual(sweep.storage_max_steps, 7)
        np.testing.assert_array_equal(
            prepared.result_arrays["n_agents"],
            np.asarray([3, 5], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            prepared.result_arrays["raw_step_horizons"],
            np.asarray([4, 7], dtype=np.int64),
        )
        self.assertEqual(
            prepared.result_arrays["raw_step_horizons"].dtype,
            np.dtype(np.int64),
        )

    def test_activation_budget_ceiling_uses_exact_float_value(self) -> None:
        config = task_config(
            n_agents=3,
            n_actors=1,
            budget=activation_opportunities_per_agent(0.33333333333333337),
        )

        sweep = self._build_sweep(config)

        self.assertEqual(sweep.raw_step_horizons, (2,))

    def test_activation_budget_preserves_integral_values_above_float_precision(
        self,
    ) -> None:
        value = 2**53 + 1
        config = task_config(
            n_agents=1,
            n_actors=1,
            budget=activation_opportunities_per_agent(value),
        )

        sweep = self._build_sweep(config)

        self.assertEqual(sweep.budget_value, value)
        self.assertIs(type(sweep.budget_value), int)
        self.assertEqual(sweep.raw_step_horizons, (value,))

    def test_activation_budget_rejects_integral_horizon_above_int64(self) -> None:
        value = int(np.iinfo(np.int64).max) + 1
        config = task_config(
            n_agents=1,
            n_actors=1,
            budget=activation_opportunities_per_agent(value),
        )

        with self.assertRaisesRegex(ValueError, "invalid raw horizon"):
            self._build_sweep(config)

    def test_raw_budget_repeats_exact_horizon_for_each_population(self) -> None:
        config = task_config(
            n_agents=[3, 5],
            n_actors=2,
            budget=raw_simulator_steps(6),
        )

        sweep = self._build_sweep(config)
        settings = tuple(
            sweep.resolve(sweep.configuration_values[index, 0, 0])
            for index in range(2)
        )

        self.assertEqual(sweep.budget_kind, "raw_simulator_steps")
        self.assertEqual(sweep.budget_value, 6)
        self.assertEqual(sweep.raw_step_horizons, (6, 6))
        self.assertEqual(sweep.storage_max_steps, 6)
        self.assertEqual(tuple(setting.max_steps for setting in settings), (6, 6))
        self.assertTrue(all(type(setting.max_steps) is int for setting in settings))

    def test_simulator_budget_validates_only_required_numeric_contract(self) -> None:
        cases = (
            ({"kind": "unknown", "value": 1}, "simulator.budget.kind"),
            (raw_simulator_steps(True), "non-negative integer"),
            ({"kind": "raw_simulator_steps", "value": 1.5}, "non-negative integer"),
            (
                activation_opportunities_per_agent(float("inf")),
                "finite non-negative number",
            ),
            (
                {"kind": "activation_opportunities_per_agent", "value": True},
                "finite non-negative number",
            ),
            (
                activation_opportunities_per_agent(-0.5),
                "finite non-negative number",
            ),
        )
        for budget, expected_error in cases:
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, expected_error):
                    prepare_policy_run(task_config(budget=budget))

    def test_simulator_requires_positive_integral_configured_actor_count(
        self,
    ) -> None:
        for n_actors in (0, True, 1.5):
            with self.subTest(n_actors=n_actors):
                config = task_config()
                config["simulator"]["actor_scheduler"]["n_actors"] = n_actors
                with self.assertRaisesRegex(ValueError, "n_actors.*positive integer"):
                    prepare_policy_run(config)

    def test_simulator_max_steps_is_not_a_budget_alias(self) -> None:
        config = task_config()
        del config["simulator"]["budget"]
        config["simulator"]["max_steps"] = 3

        with self.assertRaisesRegex(KeyError, "budget"):
            prepare_policy_run(config)

    def test_prepare_policy_run_requires_activation_weight(self) -> None:
        config = cast(dict[str, Any], task_config())
        population = cast(dict[str, Any], config["population"])
        del population["agent_parameters"]

        with self.assertRaisesRegex(ValueError, "must define 'activation_weight'"):
            prepare_policy_run(config)

    def test_prepare_policy_run_accepts_erdos_renyi_initial_graph(self) -> None:
        config = cast(dict[str, Any], task_config())
        simulator = cast(dict[str, Any], config["simulator"])
        simulator["initial_graph"] = {
            "kind": "erdos_renyi",
            "p": 0.25,
        }

        input_spec = collect_parameter_specs(config)
        parameter_samples = sample_parameters(input_spec)
        policy_entries = parse_policy_entries(config)
        sweep = build_sweep(
            config,
            parameter_samples,
            parameter_specs=input_spec["parameters"],
            policy_ids=tuple(policy_id for policy_id, _ in policy_entries),
        )
        policy_task = prepare_policy_run(config)
        settings = sweep.resolve(sweep.configuration_at(0))

        self.assertEqual(policy_task.inputs, (0,))
        self.assertEqual(sweep.initial_graph_kind, "erdos_renyi")
        self.assertEqual(sweep.initial_graph_p, 0.25)
        self.assertEqual(settings.initial_graph_kind, "erdos_renyi")
        self.assertEqual(settings.initial_graph_p, 0.25)

    def test_prepare_policy_run_rejects_invalid_erdos_renyi_probability(self) -> None:
        invalid_initial_graphs = (
            {"kind": "erdos_renyi"},
            {"kind": "erdos_renyi", "p": None},
            {"kind": "erdos_renyi", "p": -0.1},
            {"kind": "erdos_renyi", "p": 1.1},
            {"kind": "erdos_renyi", "p": True},
            {"kind": "erdos_renyi", "p": "0.1"},
        )

        for initial_graph in invalid_initial_graphs:
            with self.subTest(initial_graph=initial_graph):
                config = cast(dict[str, Any], task_config())
                simulator = cast(dict[str, Any], config["simulator"])
                simulator["initial_graph"] = initial_graph

                with self.assertRaisesRegex(ValueError, "simulator.initial_graph.p"):
                    prepare_policy_run(config)


if __name__ == "__main__":
    unittest.main()
