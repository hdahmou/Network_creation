"""Tests for policy task config handling."""

from __future__ import annotations

from typing import Any, cast
import unittest

from tasks.policies import prepare_policy_run
from tasks.policies.sweep import (
    build_sweep,
    collect_parameter_specs,
    parse_policy_entries,
    sample_parameters,
)
from tests.tasks.config_support import task_config


class PolicyTaskConfigTest(unittest.TestCase):
    def test_prepare_policy_run_uses_root_policy_config(self) -> None:
        policy_task = prepare_policy_run(
            {
                "sampling": {"log2_budget": 0, "seed": 1},
                "simulator": {
                    "seed": 1,
                    "max_steps": 1,
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
                    "outputs": [{"name": "density", "output": "mean"}],
                },
            }
        )

        self.assertEqual(policy_task.inputs, (0,))

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
