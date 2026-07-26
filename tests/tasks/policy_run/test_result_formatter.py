"""Tests for multi-policy result formatting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.execution.zarr_store import ZarrResultStore
from tasks.policies import load_policy_results, prepare_policy_run
from tests.tasks.config_support import continuous, fixed, policy, task_config


class PolicyRunResultFormatterTest(unittest.TestCase):
    @staticmethod
    def _config() -> dict[str, object]:
        return task_config(
            n_agents=4,
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
            policy_proportions={"policy": "myopic", "values": [0.25, 0.75]},
            observers={
                "defaults": {
                    "sampling": {"kind": "periodic", "every": 1},
                    "aggregations": ["mean"],
                },
                "outputs": [{"name": "density"}],
            },
        )

    def test_from_zarr_reconstructs_policy_proportion_axis(self) -> None:
        task = prepare_policy_run(self._config())
        prepared = task

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "policy-run.zarr"
            writer = ZarrResultStore(
                str(store_path),
                output_schema=prepared.output_schema,
                batch_size=2,
                total_inputs=len(prepared.inputs),
                metadata_attrs={
                    **dict(prepared.result_attrs),
                    "config_path": str(store_path.with_suffix(".run_config.json")),
                },
                metadata_arrays=prepared.result_arrays,
            )
            writer.add_success(
                input_index=0,
                output={
                    "density.mean": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
                },
            )
            writer.add_success(
                input_index=2,
                output={
                    "density.mean": np.asarray(
                        [20.0, 21.0, 22.0, 23.0],
                        dtype=np.float32,
                    )
                },
            )
            writer.add_success(
                input_index=3,
                output={
                    "density.mean": np.asarray(
                        [30.0, 31.0, 32.0, 33.0],
                        dtype=np.float32,
                    )
                },
            )
            writer.flush()

            ds = load_policy_results(store_path)

        self.assertEqual(
            tuple(ds.coords["theta_parameter"].values),
            (
                "n_agents",
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
                "myopic.bridge_probability",
                "agent.activation_weight",
            ),
        )
        np.testing.assert_allclose(ds.coords["myopic_proportion"].values, [0.25, 0.75])
        np.testing.assert_array_equal(
            ds.coords["parameter_row"].values,
            np.asarray([0, 1], dtype=np.int64),
        )
        self.assertEqual(ds["x"].shape, (1, 2, 2, 4, 1))
        np.testing.assert_allclose(
            ds["x"].isel(step=-1, observable=0).values,
            np.asarray([[[3.0, np.nan], [23.0, 33.0]]], dtype=np.float32),
            equal_nan=True,
        )

    def test_from_zarr_reconstructs_policy_count_axis(self) -> None:
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
            observers={
                "defaults": {
                    "sampling": {"kind": "periodic", "every": 1},
                    "aggregations": ["mean"],
                },
                "outputs": [{"name": "density"}],
            },
        )
        prepared = prepare_policy_run(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "policy-count-run.zarr"
            writer = ZarrResultStore(
                str(store_path),
                output_schema=prepared.output_schema,
                batch_size=2,
                total_inputs=len(prepared.inputs),
                metadata_attrs={
                    **dict(prepared.result_attrs),
                    "config_path": str(store_path.with_suffix(".run_config.json")),
                },
                metadata_arrays=prepared.result_arrays,
            )
            writer.add_success(
                input_index=0,
                output={
                    "density.mean": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
                },
            )
            writer.add_success(
                input_index=1,
                output={
                    "density.mean": np.asarray(
                        [10.0, 11.0, 12.0, 13.0],
                        dtype=np.float32,
                    )
                },
            )
            writer.flush()

            ds = load_policy_results(store_path)

        np.testing.assert_allclose(ds.coords["threshold_link_count"].values, [1.0])
        self.assertEqual(ds["x"].shape, (2, 1, 1, 4, 1))
        self.assertNotIn("threshold_link_count", ds.coords["theta_parameter"].values)
        self.assertNotIn("best_response_count", ds.coords["theta_parameter"].values)

    def test_from_zarr_preserves_policy_welfare_axis(self) -> None:
        config = task_config(
            n_agents=4,
            n_actors=2,
            log2_budget=0,
            policies=[
                policy(
                    "myopic",
                    "myopic",
                    {"bridge_probability": fixed(0.5)},
                ),
                policy("random", "random"),
            ],
            policy_proportions={"policy": "myopic", "values": [0.5]},
            observers={
                "defaults": {
                    "sampling": {"kind": "periodic", "every": 1},
                    "aggregations": ["mean"],
                },
                "outputs": [{"name": "policy_welfare"}],
            },
        )
        prepared = prepare_policy_run(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "policy-welfare-run.zarr"
            writer = ZarrResultStore(
                str(store_path),
                output_schema=prepared.output_schema,
                batch_size=2,
                total_inputs=len(prepared.inputs),
                metadata_attrs={
                    **dict(prepared.result_attrs),
                    "config_path": str(store_path.with_suffix(".run_config.json")),
                },
                metadata_arrays=prepared.result_arrays,
            )
            writer.add_success(
                input_index=0,
                output={
                    "policy_welfare.mean": np.asarray(
                        [
                            [0.0, 0.0],
                            [0.1, 0.2],
                            [0.3, 0.4],
                            [0.5, 0.6],
                        ],
                        dtype=np.float32,
                    )
                },
            )
            writer.flush()

            ds = load_policy_results(store_path)

        self.assertEqual(
            ds["policy_welfare.mean"].dims,
            (
                "n_agents",
                "myopic_proportion",
                "parameter_row",
                "step",
                "policy_class",
            ),
        )
        self.assertEqual(tuple(ds.coords["policy_class"].values), ("myopic", "random"))
        self.assertEqual(ds["policy_welfare.mean"].shape, (1, 1, 1, 4, 2))
        self.assertEqual(tuple(ds.coords["observable"].values), ())
        np.testing.assert_allclose(
            ds["policy_welfare.mean"].isel(step=-1).values,
            np.asarray([[[[0.5, 0.6]]]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
