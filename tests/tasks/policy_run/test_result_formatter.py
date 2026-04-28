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
                "outputs": [{"name": "density", "output": "mean"}],
            },
        )

    def test_from_zarr_reconstructs_policy_proportion_axis_in_theta(self) -> None:
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
                duration_ms=1.0,
            )
            writer.add_success(
                input_index=2,
                output={
                    "density.mean": np.asarray(
                        [20.0, 21.0, 22.0, 23.0],
                        dtype=np.float32,
                    )
                },
                duration_ms=3.0,
            )
            writer.add_success(
                input_index=3,
                output={
                    "density.mean": np.asarray(
                        [30.0, 31.0, 32.0, 33.0],
                        dtype=np.float32,
                    )
                },
                duration_ms=4.0,
            )
            writer.flush()

            ds = load_policy_results(store_path)

        self.assertEqual(
            tuple(ds.coords["theta_parameter"].values),
            (
                "n_agents",
                "myopic_proportion",
                "myopic_count",
                "random_count",
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
                "myopic.bridge_probability",
            ),
        )
        np.testing.assert_array_equal(
            ds.coords["policy_proportion_index"].values,
            np.asarray([0, 1, 1], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            ds.coords["parameter_row"].values,
            np.asarray([0, 0, 1], dtype=np.int64),
        )
        np.testing.assert_allclose(
            ds["theta"].sel(theta_parameter="myopic_proportion").values,
            np.asarray([0.25, 0.75, 0.75], dtype=np.float32),
        )
        np.testing.assert_allclose(
            ds["theta"].sel(theta_parameter="myopic_count").values,
            np.asarray([1.0, 3.0, 3.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            ds["theta"].sel(theta_parameter="random_count").values,
            np.asarray([3.0, 1.0, 1.0], dtype=np.float32),
        )
        self.assertEqual(ds["x"].shape, (3, 4, 1))
        np.testing.assert_allclose(
            ds["x"].isel(step=-1, observable=0).values,
            np.asarray([3.0, 23.0, 33.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
