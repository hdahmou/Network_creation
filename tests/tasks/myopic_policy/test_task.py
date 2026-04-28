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


class MyopicPolicyTaskTest(unittest.TestCase):
    @staticmethod
    def _base_config() -> dict[str, object]:
        return task_config(            game={
                "benefit_to_cost_ratio": fixed(2.0),
                "distance_discount": fixed(0.5),
            },
            policies=[
                policy("myopic", "myopic", {"bridge_probability": fixed(1.0)})
            ],
        )

    def test_prepare_builds_structured_output_schema_and_run_returns_arrays(self) -> None:
        task = prepare_policy_run(self._base_config())
        prepared = task

        self.assertEqual(
            prepared.output_schema.metadata(),
            [
                {"name": "density.mean", "shape": [4], "dtype": "float32"},
                {"name": "centered_mean_welfare.mean", "shape": [4], "dtype": "float32"},
            ],
        )
        self.assertEqual(prepared.inputs, (0,))
        self.assertEqual(
            task.parameter_samples.parameter_names,
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
                "bridge_probability",
            ),
        )

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

    def test_run_supports_n_agents_sweeps(self) -> None:
        config = self._base_config()
        config["population"]["n_agents"] = [3, 4]

        prepared = prepare_policy_run(config)

        self.assertEqual(prepared.inputs, (0, 1))
        for sample_idx in prepared.inputs:
            output = prepared.run(sample_idx)
            self.assertEqual(set(output), {"density.mean", "centered_mean_welfare.mean"})

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

    def test_run_matches_explicit_centered_mean_welfare_settings(self) -> None:
        inferred_prepared = prepare_policy_run(self._base_config())

        explicit_config = self._base_config()
        explicit_config["observers"]["outputs"][1]["settings"] = {
            "benefit_to_cost_ratio": 2.0,
            "distance_discount": 0.5,
        }
        explicit_prepared = prepare_policy_run(explicit_config)

        inferred_output = inferred_prepared.run(inferred_prepared.inputs[0])
        explicit_output = explicit_prepared.run(explicit_prepared.inputs[0])

        np.testing.assert_allclose(
            inferred_output["centered_mean_welfare.mean"],
            explicit_output["centered_mean_welfare.mean"],
        )

    def test_changing_bridge_probability_changes_output(self) -> None:
        config = self._base_config()
        config["sampling"]["log2_budget"] = 1
        config["sampling"]["seed"] = 55
        config["population"]["policies"][0]["parameters"][
            "bridge_probability"
        ] = continuous(0.0, 1.0)
        prepared = prepare_policy_run(config)
        self.assertEqual(prepared.inputs, (0, 1))

        no_bridge_output = prepared.run(prepared.inputs[0])
        bridge_output = prepared.run(prepared.inputs[1])

        self.assertFalse(
            np.array_equal(no_bridge_output["density.mean"], bridge_output["density.mean"])
        )

    def test_prepare_supports_series_and_tensor_observers_in_one_schema(self) -> None:
        config = self._base_config()
        config["observers"]["outputs"] = [
            {"name": "density", "output": "mean"},
            {"name": "triad_census", "output": "mean"},
        ]
        prepared = prepare_policy_run(config)

        self.assertEqual(
            prepared.output_schema.metadata(),
            [
                {"name": "density.mean", "shape": [4], "dtype": "float32"},
                {"name": "triad_census.mean", "shape": [4, 7], "dtype": "float32"},
            ],
        )
        output = prepared.run(prepared.inputs[0])
        self.assertEqual(output["density.mean"].shape, (4,))
        self.assertEqual(output["triad_census.mean"].shape, (4, 7))

    def test_prepare_supports_terminal_tensor_observer_outputs(self) -> None:
        config = self._base_config()
        config["observers"] = {
            "defaults": {
                "sampling": {"kind": "terminal"},
                "aggregations": ["mean"],
            },
            "outputs": [{"name": "triad_census", "output": "mean"}],
        }
        prepared = prepare_policy_run(config)

        self.assertEqual(
            prepared.output_schema.metadata(),
            [
                {"name": "triad_census.mean", "shape": [1, 7], "dtype": "float32"},
            ],
        )
        output = prepared.run(prepared.inputs[0])
        self.assertEqual(output["triad_census.mean"].shape, (1, 7))


if __name__ == "__main__":
    unittest.main()
