"""Tests for policy task config handling."""

from __future__ import annotations

import unittest

from tasks.policies import prepare_policy_run


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


if __name__ == "__main__":
    unittest.main()
