"""Tests for policy task preparation."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import qmc

from tasks.policies import prepare_policy_run
from tasks.policies.sweep import (
    collect_parameter_specs,
    sample_parameters,
    build_sweep,
)
from tests.tasks.config_support import (
    continuous,
    fixed,
    policy,
    sampled_discrete_range,
    scaled_inverse_agents,
    task_config,
)


class MyopicPolicyPreparationTest(unittest.TestCase):
    @staticmethod
    def _base_config() -> dict[str, object]:
        return task_config(            n_agents=4,
            game={
                "benefit_to_cost_ratio": fixed(0.05),
                "distance_discount": fixed(0.5),
            },
            policies=[
                policy("myopic", "myopic", {"bridge_probability": fixed(0.0)})
            ],
            observers={
                "defaults": {
                    "sampling": {"kind": "periodic", "every": 1},
                    "aggregations": ["mean"],
                },
                "outputs": [{"name": "density", "output": "mean"}],
            },
        )

    def test_collect_parameter_specs_collects_game_population_and_policy_parameters(self) -> None:
        input_spec = collect_parameter_specs(self._base_config())
        self.assertEqual(input_spec["log2_budget"], 0)
        self.assertEqual(input_spec["seed"], 123)
        self.assertEqual(
            tuple(input_spec["parameters"]),
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
                "bridge_probability",
            ),
        )

    def test_build_sweep_supports_scalar_and_list_n_agents(self) -> None:
        config = self._base_config()
        config["population"]["n_agents"] = [2, 4]
        input_spec = collect_parameter_specs(config)
        parameter_samples = sample_parameters(input_spec)
        sweep = build_sweep(
            config,
            parameter_samples,
            parameter_specs=input_spec["parameters"],
            policy_ids=("myopic",),
        )

        self.assertEqual(sweep.n_agents_values, (2, 4))
        self.assertEqual(sweep.configuration_names[0], "n_agents")
        self.assertEqual(sweep.configuration_values.shape, (2, 1, 1, 7))
        self.assertEqual(sweep.resolve(sweep.configuration_at(0)).n_actors, 2)
        self.assertEqual(sweep.resolve(sweep.configuration_at(1)).n_actors, 4)

    def test_build_sweep_supports_discrete_range_n_agents(self) -> None:
        config = self._base_config()
        config["population"]["n_agents"] = {
            "kind": "sampled_discrete_range",
            "min": 10,
            "max": 40,
            "step": 10,
        }
        input_spec = collect_parameter_specs(config)
        parameter_samples = sample_parameters(input_spec)
        sweep = build_sweep(
            config,
            parameter_samples,
            parameter_specs=input_spec["parameters"],
            policy_ids=("myopic",),
        )

        self.assertEqual(sweep.n_agents_values, (10, 20, 30, 40))
        self.assertEqual(sweep.configuration_values.shape, (4, 1, 1, 7))
        np.testing.assert_allclose(
            sweep.configuration_values[:, 0, 0, 0],
            np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        )

    def test_prepare_resolves_scaled_inverse_benefit_to_cost_ratio_from_n_agents(
        self,
    ) -> None:
        config = self._base_config()
        config["population"]["n_agents"] = [5, 10]
        config["game"]["benefit_to_cost_ratio"] = scaled_inverse_agents(
            10.0, 2.0
        )

        task = prepare_policy_run(config)

        column = task.sweep.column_index_by_name[
            "benefit_to_cost_ratio"
        ]
        np.testing.assert_allclose(
            task.sweep.configuration_values[:, 0, 0, column],
            np.asarray([1.0, 0.5], dtype=np.float32),
        )

    def test_sample_parameters_samples_continuous_parameters_jointly(self) -> None:
        input_spec = {
            "log2_budget": 2,
            "seed": 123,
            "parameters": {
                "distance_discount": continuous(0.2, 0.8),
                "bridge_probability": continuous(0.0, 1.0),
                "visibility_distance": fixed(4),
            },
        }

        parameter_samples = sample_parameters(input_spec)
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
            parameter_samples.parameter_names,
            (
                "distance_discount",
                "bridge_probability",
                "visibility_distance",
            ),
        )
        np.testing.assert_allclose(parameter_samples.parameter_values, expected)

    def test_sample_parameters_collapses_all_fixed_parameters_to_one_row(self) -> None:
        input_spec = collect_parameter_specs(self._base_config())

        parameter_samples = sample_parameters(input_spec)

        self.assertEqual(parameter_samples.parameter_values.shape, (1, 4))
        np.testing.assert_allclose(
            parameter_samples.parameter_values,
            np.array([[0.05, 0.5, 4.0, 0.0]], dtype=np.float32),
        )

    def test_prepare_expands_n_agents_sweeps_cartesian_product(self) -> None:
        config = self._base_config()
        config["population"]["n_agents"] = [2, 5, 7]
        config["sampling"]["log2_budget"] = 1
        config["population"]["policies"][0]["parameters"][
            "bridge_probability"
        ] = continuous(0.0, 1.0)

        task = prepare_policy_run(config)
        prepared = task

        self.assertEqual(prepared.inputs, (0, 1, 2, 3, 4, 5))
        self.assertEqual(task.sweep.configuration_values.shape, (3, 1, 2, 7))
        np.testing.assert_allclose(
            task.sweep.configuration_values[:, 0, :, 0],
            np.array(
                [
                    [2.0, 2.0],
                    [5.0, 5.0],
                    [7.0, 7.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_sample_parameters_supports_sampled_discrete_ranges(self) -> None:
        input_spec = {
            "log2_budget": 2,
            "seed": 123,
            "parameters": {
                "benefit_to_cost_ratio": continuous(0.2, 0.8),
                "distance_discount": fixed(0.5),
                "bridge_probability": fixed(0.0),
                "visibility_distance": sampled_discrete_range(1, 4),
            },
        }

        parameter_samples = sample_parameters(input_spec)
        expected_visibility_distance = np.random.default_rng(123 + 3).choice(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            size=4,
            replace=True,
        )

        self.assertEqual(parameter_samples.parameter_values.shape, (4, 4))
        np.testing.assert_allclose(
            parameter_samples.parameter_values[:, 3],
            expected_visibility_distance,
        )

    def test_sample_parameters_rejects_negative_log2_budget(self) -> None:
        input_spec = collect_parameter_specs(self._base_config())
        input_spec["log2_budget"] = -1

        with self.assertRaisesRegex(ValueError, "log2_budget"):
            sample_parameters(input_spec)


if __name__ == "__main__":
    unittest.main()
