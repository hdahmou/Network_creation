"""Tests for myopic-policy task preparation."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.core.model import SchedulingMode
from tasks.policies.myopic_policy import myopicPolicyRun
from tasks.policies.utils import generate_parameter_table, load_simulation_settings


class MyopicPolicyPreparationTest(unittest.TestCase):
    @staticmethod
    def _base_config() -> dict[str, object]:
        return {
            "input_spec": {
                "log2_budget": 0,
                "sampler_seed": 123,
                "parameters": {
                    "link_cost": 0.5,
                    "decay": 0.5,
                    "per_link_benefit": 10.0,
                    "bridge_probability": 0.0,
                },
            },
            "simulation": {
                "n_agents": 4,
                "radius": 2,
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
                        "observation_frequency": 1,
                        "aggregations": ["mean"],
                    },
                },
                {
                    "name": "social_welfare",
                    "output": "mean",
                    "kwargs": {
                        "observation_frequency": 1,
                        "aggregations": ["mean"],
                        "link_cost": 1.0,
                        "decay": 0.5,
                        "per_link_benefit": 10.0,
                    },
                },
            ],
        }

    def test_load_simulation_settings(self) -> None:
        settings = load_simulation_settings(self._base_config()["simulation"])
        self.assertEqual(settings.n_agents, 4)
        self.assertEqual(settings.radius, 2)
        self.assertEqual(settings.mode, SchedulingMode.SIMULTANEOUS)
        self.assertEqual(settings.seed, 11)
        self.assertEqual(settings.max_steps, 3)
        self.assertEqual(settings.n_episodes, 2)
        self.assertIsNone(settings.stagnation_patience)
        self.assertEqual(settings.initial_graph_kind, "empty")

    def test_generate_parameter_table_preserves_order_and_repeats_fixed_columns(self) -> None:
        input_spec = {
            "log2_budget": 2,
            "sampler_seed": 123,
            "parameters": {
                "link_cost": 0.5,
                "decay": {
                    "min": 0.2,
                    "max": 0.8,
                },
                "per_link_benefit": 10.0,
                "bridge_probability": {
                    "min": 0.0,
                    "max": 1.0,
                },
            },
        }
        parameter_table = generate_parameter_table(input_spec)

        self.assertEqual(
            parameter_table.parameter_names,
            ("link_cost", "decay", "per_link_benefit", "bridge_probability"),
        )
        self.assertEqual(parameter_table.parameter_values.shape, (4, 4))
        np.testing.assert_allclose(parameter_table.parameter_values[:, 0], 0.5)
        np.testing.assert_allclose(parameter_table.parameter_values[:, 2], 10.0)
        self.assertTrue(np.all(parameter_table.parameter_values[:, 1] >= 0.2))
        self.assertTrue(np.all(parameter_table.parameter_values[:, 1] <= 0.8))
        self.assertTrue(np.all(parameter_table.parameter_values[:, 3] >= 0.0))
        self.assertTrue(np.all(parameter_table.parameter_values[:, 3] <= 1.0))
        self.assertGreater(
            np.unique(parameter_table.parameter_values[:, 1]).size,
            1,
        )
        self.assertGreater(
            np.unique(parameter_table.parameter_values[:, 3]).size,
            1,
        )

    def test_generate_parameter_table_collapses_all_fixed_parameters_to_one_row(self) -> None:
        input_spec = {
            "log2_budget": 3,
            "sampler_seed": 123,
            "parameters": {
                "link_cost": 0.5,
                "decay": 0.5,
                "per_link_benefit": 10.0,
                "bridge_probability": 0.0,
            },
        }

        parameter_table = generate_parameter_table(input_spec)

        self.assertEqual(parameter_table.parameter_values.shape, (1, 4))
        np.testing.assert_allclose(
            parameter_table.parameter_values,
            np.array([[0.5, 0.5, 10.0, 0.0]], dtype=np.float32),
        )

    def test_prepare_generates_row_index_inputs_from_input_spec(self) -> None:
        task = myopicPolicyRun()
        prepared = task.prepare(self._base_config())

        self.assertEqual(prepared.inputs, (0,))
        self.assertEqual(
            task.parameter_table.parameter_names,
            ("link_cost", "decay", "per_link_benefit", "bridge_probability"),
        )
        np.testing.assert_allclose(
            task.parameter_table.parameter_values,
            np.array([[0.5, 0.5, 10.0, 0.0]], dtype=np.float32),
        )

    def test_generate_parameter_table_rejects_negative_log2_budget(self) -> None:
        input_spec = {
            "log2_budget": -1,
            "sampler_seed": 123,
            "parameters": {
                "link_cost": 0.5,
                "decay": 0.5,
                "per_link_benefit": 10.0,
                "bridge_probability": 0.0,
            },
        }

        with self.assertRaisesRegex(ValueError, "log2_budget"):
            generate_parameter_table(input_spec)
