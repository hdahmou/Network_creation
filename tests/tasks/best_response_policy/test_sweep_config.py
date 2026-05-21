"""Tests for policy task preparation."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import qmc

from tasks.policies import prepare_policy_run
from tasks.policies.sweep import build_sweep, collect_parameter_specs, sample_parameters
from tests.tasks.config_support import (
    continuous,
    fixed,
    policy,
    task_config,
)


class BestResponsePolicyPreparationTest(unittest.TestCase):
    @staticmethod
    def _base_config() -> dict[str, object]:
        return task_config(            game={
                "benefit_to_cost_ratio": fixed(0.05),
                "distance_discount": fixed(0.5),
            },
            policies=[
                policy(
                    "best_response",
                    "best-response",
                    {
                        "outside_optimism": fixed(0.25),
                        "exploration": fixed(0.5),
                    },
                )
            ],
            observers={
                "defaults": {
                    "sampling": {"kind": "periodic", "every": 1},
                    "aggregations": ["mean"],
                },
                "outputs": [{"name": "density"}],
            },
        )

    def test_sample_parameters_samples_continuous_parameters_jointly(self) -> None:
        input_spec = {
            "log2_budget": 2,
            "seed": 123,
            "parameters": {
                "distance_discount": continuous(0.2, 0.8),
                "outside_optimism": continuous(0.0, 1.0),
                "exploration": continuous(1e-3, 1.0),
                "visibility_distance": fixed(4),
            },
        }

        parameter_samples = sample_parameters(input_spec)
        sampler = qmc.Sobol(
            d=3,
            scramble=True,
            optimization="random-cd",
            rng=123,
        )
        unit_samples = np.asarray(sampler.random(n=4), dtype=np.float32)
        expected = np.column_stack(
            (
                0.2 + (unit_samples[:, 0] * 0.6),
                unit_samples[:, 1],
                1e-3 + (unit_samples[:, 2] * 0.999),
                np.full(4, 4.0, dtype=np.float32),
            )
        ).astype(np.float32, copy=False)

        self.assertEqual(
            parameter_samples.parameter_names,
            (
                "distance_discount",
                "outside_optimism",
                "exploration",
                "visibility_distance",
            ),
        )
        np.testing.assert_allclose(parameter_samples.parameter_values, expected)

    def test_prepare_generates_row_index_inputs_from_task_spec(self) -> None:
        config = self._base_config()
        input_spec = collect_parameter_specs(config)
        parameter_samples = sample_parameters(input_spec)
        sweep = build_sweep(
            config,
            parameter_samples,
            parameter_specs=input_spec["parameters"],
            policy_ids=("best_response",),
        )
        prepared = prepare_policy_run(config)

        self.assertEqual(prepared.inputs, (0,))
        self.assertEqual(sweep.configuration_values.shape, (1, 1, 1, 8))
        self.assertEqual(
            parameter_samples.parameter_names,
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
                "outside_optimism",
                "exploration",
            ),
        )
        np.testing.assert_allclose(
            parameter_samples.parameter_values,
            np.array(
                [[0.05, 0.5, 4.0, 0.25, 0.5]],
                dtype=np.float32,
            ),
        )

    def test_collect_parameter_specs_uses_readable_paths(self) -> None:
        input_spec = collect_parameter_specs(self._base_config())

        self.assertEqual(
            tuple(input_spec["parameters"]),
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
                "outside_optimism",
                "exploration",
            ),
        )


if __name__ == "__main__":
    unittest.main()
