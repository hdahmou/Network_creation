"""Tests for myopic-policy task preparation."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import qmc

from tasks.policies import MyopicPolicyRun
from tasks.policies.support import generate_parameter_table, load_simulation_settings


class MyopicPolicyPreparationTest(unittest.TestCase):
    @staticmethod
    def _base_config() -> dict[str, object]:
        return {
            "input_spec": {
                "log2_budget": 0,
                "sampler_seed": 123,
                "parameters": {
                    "benefit_to_cost_ratio": {"type": "fixed", "payload": 0.05},
                    "distance_discount": {"type": "fixed", "payload": 0.5},
                    "bridge_probability": {"type": "fixed", "payload": 0.0},
                    "visibility_distance": {"type": "fixed", "payload": 4},
                },
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
                {
                    "name": "normalized_welfare",
                    "output": "mean",
                    "kwargs": {
                        "observation_sampling": {"kind": "periodic", "every": 1},
                        "aggregations": ["mean"],
                        "benefit_to_cost_ratio": 0.1,
                        "distance_discount": 0.5,
                    },
                },
            ],
        }

    def test_load_simulation_settings(self) -> None:
        settings = load_simulation_settings(self._base_config()["simulation"])
        self.assertEqual(settings.n_agents, 4)
        self.assertEqual(settings.n_actors, 4)
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
                "benefit_to_cost_ratio": {"type": "fixed", "payload": 0.05},
                "distance_discount": {
                    "type": "continuous",
                    "payload": {
                        "min": 0.2,
                        "max": 0.8,
                    },
                },
                "bridge_probability": {
                    "type": "continuous",
                    "payload": {
                        "min": 0.0,
                        "max": 1.0,
                    },
                },
                "visibility_distance": {"type": "fixed", "payload": 4},
            },
        }
        parameter_table = generate_parameter_table(input_spec)

        self.assertEqual(
            parameter_table.parameter_names,
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "bridge_probability",
                "visibility_distance",
            ),
        )
        self.assertEqual(parameter_table.parameter_values.shape, (4, 4))
        np.testing.assert_allclose(parameter_table.parameter_values[:, 0], 0.05)
        np.testing.assert_allclose(parameter_table.parameter_values[:, 3], 4.0)
        self.assertTrue(np.all(parameter_table.parameter_values[:, 1] >= 0.2))
        self.assertTrue(np.all(parameter_table.parameter_values[:, 1] <= 0.8))
        self.assertTrue(np.all(parameter_table.parameter_values[:, 2] >= 0.0))
        self.assertTrue(np.all(parameter_table.parameter_values[:, 2] <= 1.0))
        self.assertGreater(
            np.unique(parameter_table.parameter_values[:, 1]).size,
            1,
        )
        self.assertGreater(
            np.unique(parameter_table.parameter_values[:, 2]).size,
            1,
        )

    def test_generate_parameter_table_samples_continuous_parameters_jointly(self) -> None:
        input_spec = {
            "log2_budget": 2,
            "sampler_seed": 123,
            "parameters": {
                "distance_discount": {
                    "type": "continuous",
                    "payload": {"min": 0.2, "max": 0.8},
                },
                "bridge_probability": {
                    "type": "continuous",
                    "payload": {"min": 0.0, "max": 1.0},
                },
                "visibility_distance": {"type": "fixed", "payload": 4},
            },
        }

        parameter_table = generate_parameter_table(input_spec)
        sampler = qmc.Sobol(
            d=2,
            scramble=True,
            optimization="random-cd",
            rng=123,
        )
        unit_samples = np.asarray(sampler.random(n=4), dtype=np.float32)
        expected = np.column_stack(
            (
                0.2 + (unit_samples[:, 0] * 0.6),
                unit_samples[:, 1],
                np.full(4, 4.0, dtype=np.float32),
            )
        ).astype(np.float32, copy=False)

        self.assertEqual(
            parameter_table.parameter_names,
            ("distance_discount", "bridge_probability", "visibility_distance"),
        )
        np.testing.assert_allclose(parameter_table.parameter_values, expected)

    def test_generate_parameter_table_collapses_all_fixed_parameters_to_one_row(self) -> None:
        input_spec = {
            "log2_budget": 3,
            "sampler_seed": 123,
            "parameters": {
                "benefit_to_cost_ratio": {"type": "fixed", "payload": 0.05},
                "distance_discount": {"type": "fixed", "payload": 0.5},
                "bridge_probability": {"type": "fixed", "payload": 0.0},
                "visibility_distance": {"type": "fixed", "payload": 4},
            },
        }

        parameter_table = generate_parameter_table(input_spec)

        self.assertEqual(parameter_table.parameter_values.shape, (1, 4))
        np.testing.assert_allclose(
            parameter_table.parameter_values,
            np.array([[0.05, 0.5, 0.0, 4.0]], dtype=np.float32),
        )

    def test_prepare_generates_row_index_inputs_from_input_spec(self) -> None:
        task = MyopicPolicyRun()
        prepared = task.prepare(self._base_config())

        self.assertEqual(prepared.inputs, (0,))
        self.assertEqual(
            task.parameter_table.parameter_names,
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "bridge_probability",
                "visibility_distance",
            ),
        )
        np.testing.assert_allclose(
            task.parameter_table.parameter_values,
            np.array([[0.05, 0.5, 0.0, 4.0]], dtype=np.float32),
        )

    def test_prepare_rejects_missing_visibility_distance(self) -> None:
        config = self._base_config()
        parameters = dict(config["input_spec"]["parameters"])
        del parameters["visibility_distance"]
        config["input_spec"] = {
            "log2_budget": 0,
            "sampler_seed": 123,
            "parameters": parameters,
        }

        with self.assertRaisesRegex(ValueError, "visibility_distance is required"):
            MyopicPolicyRun().prepare(config)

    def test_generate_parameter_table_supports_sampled_discrete_visibility_distance_ranges(
        self,
    ) -> None:
        input_spec = {
            "log2_budget": 2,
            "sampler_seed": 123,
            "parameters": {
                "benefit_to_cost_ratio": {"type": "fixed", "payload": 0.05},
                "distance_discount": {"type": "fixed", "payload": 0.5},
                "bridge_probability": {"type": "fixed", "payload": 0.0},
                "visibility_distance": {
                    "type": "sampled_discrete_range",
                    "payload": {"min": 1, "max": 4, "step": 1},
                },
            },
        }

        parameter_table = generate_parameter_table(input_spec)
        sampled_visibility_distance = parameter_table.parameter_values[:, 3]
        expected_visibility_distance = np.random.default_rng(123 + 3).choice(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            size=4,
            replace=True,
        )

        self.assertEqual(parameter_table.parameter_values.shape, (4, 4))
        self.assertTrue(np.all(sampled_visibility_distance >= 1.0))
        self.assertTrue(np.all(sampled_visibility_distance <= 4.0))
        self.assertTrue(
            np.all(sampled_visibility_distance == np.floor(sampled_visibility_distance))
        )
        np.testing.assert_allclose(
            sampled_visibility_distance,
            expected_visibility_distance,
        )

    def test_generate_parameter_table_samples_float_and_integer_ranges_rowwise(self) -> None:
        input_spec = {
            "log2_budget": 2,
            "sampler_seed": 123,
            "parameters": {
                "benefit_to_cost_ratio": {
                    "type": "continuous",
                    "payload": {"min": 0.2, "max": 0.8},
                },
                "distance_discount": {"type": "fixed", "payload": 0.5},
                "bridge_probability": {"type": "fixed", "payload": 0.0},
                "visibility_distance": {
                    "type": "sampled_discrete_range",
                    "payload": {"min": 1, "max": 4, "step": 1},
                },
            },
        }

        parameter_table = generate_parameter_table(input_spec)
        unit_samples = np.asarray(
            qmc.Sobol(
                d=1,
                scramble=True,
                optimization="random-cd",
                rng=123,
            ).random(n=4),
            dtype=np.float32,
        )
        expected_visibility_distance = np.random.default_rng(123 + 3).choice(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            size=4,
            replace=True,
        )
        expected = np.column_stack(
            (
                0.2 + (unit_samples[:, 0] * 0.6),
                np.full(4, 0.5, dtype=np.float32),
                np.full(4, 0.0, dtype=np.float32),
                expected_visibility_distance,
            )
        ).astype(np.float32, copy=False)

        self.assertEqual(parameter_table.parameter_values.shape, (4, 4))
        np.testing.assert_allclose(parameter_table.parameter_values, expected)

    def test_generate_parameter_table_rejects_negative_log2_budget(self) -> None:
        input_spec = {
            "log2_budget": -1,
            "sampler_seed": 123,
            "parameters": {
                "benefit_to_cost_ratio": {"type": "fixed", "payload": 0.05},
                "distance_discount": {"type": "fixed", "payload": 0.5},
                "bridge_probability": {"type": "fixed", "payload": 0.0},
                "visibility_distance": {"type": "fixed", "payload": 4},
            },
        }

        with self.assertRaisesRegex(ValueError, "log2_budget"):
            generate_parameter_table(input_spec)
