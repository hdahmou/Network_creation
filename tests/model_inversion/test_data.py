from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from network_creation.model_inversion.data import (
    filter_inversion_dataset,
    inversion_dataset_metadata,
)


class ModelDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workdir = Path(self.temp_dir.name)
        self.config_path = self.workdir / "run.run_config.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "sampling": {"log2_budget": 2, "seed": 123},
                    "game": {
                        "alpha": {
                            "kind": "continuous",
                            "min": 0.0,
                            "max": 1.0,
                        },
                        "beta": {"kind": "fixed", "value": 5.0},
                    }
                }
            ),
            encoding="utf-8",
        )

    def formatted_results(self) -> xr.Dataset:
        theta = np.asarray(
            [
                [10.0, 0.1, 5.0],
                [10.0, 0.3, 5.0],
                [20.0, 0.4, 5.0],
                [20.0, 0.5, 5.0],
                [20.0, 0.6, 5.0],
            ],
            dtype=np.float32,
        )
        return xr.Dataset(
            data_vars={
                "theta": (("sample", "theta_parameter"), theta),
                "prior_low": (("theta_parameter",), np.asarray([10.0, 0.0, 5.0])),
                "prior_high": (("theta_parameter",), np.asarray([20.0, 1.0, 5.0])),
                "x": (
                    ("sample", "step", "observable"),
                    np.arange(30, dtype=np.float32).reshape(5, 2, 3),
                ),
            },
            coords={
                "sample": np.arange(5, dtype=np.int64),
                "input_index": ("sample", np.asarray([0, 2, 3, 4, 5], dtype=np.int64)),
                "parameter_row": ("sample", np.asarray([0, 2, 0, 1, 2], dtype=np.int64)),
                "theta_parameter": ("theta_parameter", ["n_agents", "alpha", "beta"]),
                "observable": (
                    "observable",
                    [
                        "density.mean",
                        "distance.mean",
                        "indegree.mean",
                    ],
                ),
                "step": np.asarray([0, 5], dtype=np.int64),
            },
            attrs={
                "result_store_path": str(self.workdir / "run.zarr"),
                "config_path": str(self.config_path),
                "has_failures": False,
            },
        )

    def filter_results(self, **kwargs: object):
        return filter_inversion_dataset(self.formatted_results(), **kwargs)

    def test_filter_inversion_dataset_returns_final_tensors_and_metadata(self) -> None:
        dataset = self.filter_results(
            observers=("density", "distance.mean"),
        )
        self.assertIsInstance(dataset, xr.Dataset)
        theta = dataset["theta"].values
        x = dataset["x"].values
        metadata = inversion_dataset_metadata(dataset)

        np.testing.assert_allclose(
            theta,
            np.asarray(
                [
                    [10.0, 0.1],
                    [10.0, 0.3],
                    [20.0, 0.4],
                    [20.0, 0.5],
                    [20.0, 0.6],
                ],
                dtype=np.float32,
            ),
        )
        expected_x = self.formatted_results()["x"].isel(step=-1).values[
            np.asarray(dataset["kept_mask"].values, dtype=np.bool_)
        ][:, [0, 1]]
        np.testing.assert_allclose(x, expected_x)
        self.assertEqual(x.shape, (5, 2))
        self.assertEqual(
            set(metadata),
            {
                "result_store_path",
                "config_path",
                "theta_names",
                "x_names",
                "constant_theta",
                "prior_low",
                "prior_high",
                "x_event_shape",
                "x_representation",
                "n_simulations",
            },
        )
        self.assertEqual(metadata["theta_names"], ["n_agents", "alpha"])
        self.assertEqual(
            list(dataset.coords["theta_parameter"].values),
            ["n_agents", "alpha"],
        )
        self.assertEqual(metadata["x_names"], ["density.mean", "distance.mean"])
        self.assertEqual(
            list(dataset.coords["observable"].values),
            ["density.mean", "distance.mean"],
        )
        self.assertEqual(metadata["constant_theta"], {"beta": 5.0})
        self.assertEqual(metadata["prior_low"], [10.0, 0.0])
        self.assertEqual(metadata["prior_high"], [20.0, 1.0])
        self.assertEqual(metadata["x_event_shape"], [2])
        self.assertEqual(metadata["x_representation"], "final")
        self.assertEqual(metadata["n_simulations"], 5)

    def test_filter_inversion_dataset_keeps_requested_theta_subset(self) -> None:
        dataset = self.filter_results(
            observers=("density.mean",),
            theta_parameters=("beta",),
            drop_constant_theta=False,
        )
        theta = dataset["theta"].values
        metadata = inversion_dataset_metadata(dataset)

        np.testing.assert_allclose(theta, np.full((5, 1), 5.0, dtype=np.float32))
        self.assertEqual(metadata["theta_names"], ["beta"])
        self.assertEqual(metadata["constant_theta"], {})
        self.assertEqual(metadata["prior_low"], [5.0])
        self.assertEqual(metadata["prior_high"], [5.0])

    def test_filter_inversion_dataset_returns_timeseries_tensors(self) -> None:
        dataset = self.filter_results(
            observers=("indegree", "density.mean"),
            x_representation="timeseries",
        )
        x = dataset["x"].values
        metadata = inversion_dataset_metadata(dataset)

        raw = self.formatted_results()["x"].values
        expected_first = raw[0, :, :][:, [2, 0]]
        self.assertEqual(x.shape, (5, 2, 2))
        np.testing.assert_allclose(x[0], expected_first)
        self.assertEqual(dataset["x"].dims, ("sample", "step", "observable"))
        self.assertEqual(metadata["x_names"], ["indegree.mean", "density.mean"])
        self.assertEqual(metadata["x_event_shape"], [2, 2])
        self.assertEqual(metadata["x_representation"], "timeseries")
        self.assertEqual(metadata["step_labels"], [0, 5])
        self.assertEqual(metadata["n_simulations"], 5)


if __name__ == "__main__":
    unittest.main()
