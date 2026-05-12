"""Tests for the concrete policy task."""

from __future__ import annotations

import unittest

import numpy as np

from tasks.policies import prepare_policy_run
from tests.tasks.config_support import (
    continuous,
    fixed,
    policy,
    scaled_inverse_agents,
    task_config,
)


class BestResponsePolicyTaskTest(unittest.TestCase):
    @staticmethod
    def _base_config() -> dict[str, object]:
        return task_config(            game={
                "benefit_to_cost_ratio": fixed(0.8),
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
        )

    def test_prepare_builds_structured_output_schema_and_run_returns_arrays(self) -> None:
        prepared = prepare_policy_run(self._base_config())

        self.assertEqual(
            prepared.output_schema.metadata(),
            [
                {"name": "density.mean", "shape": [4], "dtype": "float32"},
                {"name": "centered_mean_welfare.mean", "shape": [4], "dtype": "float32"},
            ],
        )
        self.assertEqual(prepared.inputs, (0,))

        output = prepared.run(prepared.inputs[0])

        self.assertEqual(set(output), {"density.mean", "centered_mean_welfare.mean"})
        self.assertEqual(output["density.mean"].shape, (4,))
        self.assertEqual(output["centered_mean_welfare.mean"].shape, (4,))
        self.assertEqual(output["density.mean"].dtype, np.float32)
        self.assertEqual(output["centered_mean_welfare.mean"].dtype, np.float32)

    def test_prepared_run_is_deterministic_for_fixed_seed(self) -> None:
        prepared = prepare_policy_run(self._base_config())

        first = prepared.run(prepared.inputs[0])
        second = prepared.run(prepared.inputs[0])

        self.assertEqual(set(first), set(second))
        for field_name in first:
            np.testing.assert_allclose(first[field_name], second[field_name])

    def test_run_matches_explicit_centered_mean_welfare_settings(self) -> None:
        inferred_prepared = prepare_policy_run(self._base_config())

        explicit_config = self._base_config()
        explicit_config["observers"]["outputs"][1]["settings"] = {
            "benefit_to_cost_ratio": 0.8,
            "distance_discount": 0.5,
        }
        explicit_prepared = prepare_policy_run(explicit_config)

        inferred_output = inferred_prepared.run(inferred_prepared.inputs[0])
        explicit_output = explicit_prepared.run(explicit_prepared.inputs[0])

        np.testing.assert_allclose(
            inferred_output["centered_mean_welfare.mean"],
            explicit_output["centered_mean_welfare.mean"],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_run_uses_sampled_n_agents_from_sweeps(self) -> None:
        sweep_config = self._base_config()
        sweep_config["population"]["n_agents"] = [4, 6]

        small_config = self._base_config()
        small_config["population"]["n_agents"] = 4

        large_config = self._base_config()
        large_config["population"]["n_agents"] = 6

        sweep_prepared = prepare_policy_run(sweep_config)
        small_prepared = prepare_policy_run(small_config)
        large_prepared = prepare_policy_run(large_config)

        expected_outputs = (
            small_prepared.run(small_prepared.inputs[0]),
            large_prepared.run(large_prepared.inputs[0]),
        )

        self.assertEqual(sweep_prepared.inputs, (0, 1))
        for sample_idx, expected_output in zip(
            sweep_prepared.inputs,
            expected_outputs,
            strict=True,
        ):
            observed_output = sweep_prepared.run(sample_idx)
            self.assertEqual(set(observed_output), set(expected_output))
            for field_name in observed_output:
                np.testing.assert_allclose(
                    observed_output[field_name],
                    expected_output[field_name],
                )

    def test_run_uses_scaled_inverse_benefit_to_cost_ratio_from_n_agents(self) -> None:
        sweep_config = self._base_config()
        sweep_config["population"]["n_agents"] = [5, 10]
        sweep_config["game"]["benefit_to_cost_ratio"] = scaled_inverse_agents(
            10.0, 2.0
        )

        small_config = self._base_config()
        small_config["population"]["n_agents"] = 5
        small_config["game"]["benefit_to_cost_ratio"] = scaled_inverse_agents(
            10.0, 2.0
        )

        large_config = self._base_config()
        large_config["population"]["n_agents"] = 10
        large_config["game"]["benefit_to_cost_ratio"] = scaled_inverse_agents(
            10.0, 2.0
        )

        sweep_prepared = prepare_policy_run(sweep_config)
        small_prepared = prepare_policy_run(small_config)
        large_prepared = prepare_policy_run(large_config)

        expected_outputs = (
            small_prepared.run(small_prepared.inputs[0]),
            large_prepared.run(large_prepared.inputs[0]),
        )

        self.assertEqual(sweep_prepared.inputs, (0, 1))
        for sample_idx, expected_output in zip(
            sweep_prepared.inputs,
            expected_outputs,
            strict=True,
        ):
            observed_output = sweep_prepared.run(sample_idx)
            self.assertEqual(set(observed_output), set(expected_output))
            for field_name in observed_output:
                np.testing.assert_allclose(
                    observed_output[field_name],
                    expected_output[field_name],
                )

    def test_prepare_supports_multiple_parameter_samples(self) -> None:
        config = self._base_config()
        config["sampling"]["log2_budget"] = 1
        config["population"]["policies"][0]["parameters"][
            "outside_optimism"
        ] = continuous(0.0, 1.0)
        config["population"]["policies"][0]["parameters"][
            "exploration"
        ] = continuous(1e-3, 1.0)
        prepared = prepare_policy_run(config)

        self.assertEqual(prepared.inputs, (0, 1))
        for sample_idx in prepared.inputs:
            output = prepared.run(sample_idx)
            self.assertEqual(set(output), {"density.mean", "centered_mean_welfare.mean"})


if __name__ == "__main__":
    unittest.main()
