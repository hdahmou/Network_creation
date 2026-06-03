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
    move_discrete_theta_to_end,
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
                            "std": 0.0,
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
                "observer_names",
                "input_parameters",
                "continuous_input_parameters",
                "discrete_input_parameters",
                "discrete_input_values",
                "constant_theta",
                "prior_low",
                "prior_high",
                "x_event_shape",
                "x_representation",
                "final_step_label",
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
            metadata["observer_names"],
            ["density.mean", "distance.mean"],
        )
        self.assertEqual(metadata["input_parameters"], [])
        self.assertEqual(metadata["continuous_input_parameters"], [])
        self.assertEqual(metadata["discrete_input_parameters"], [])
        self.assertEqual(metadata["discrete_input_values"], [])
        self.assertEqual(
            list(dataset.coords["observable"].values),
            ["density.mean", "distance.mean"],
        )
        self.assertEqual(metadata["constant_theta"], {"beta": 5.0})
        self.assertEqual(metadata["prior_low"], [10.0, 0.0])
        self.assertEqual(metadata["prior_high"], [20.0, 1.0])
        self.assertEqual(metadata["x_event_shape"], [2])
        self.assertEqual(metadata["x_representation"], "final")
        self.assertEqual(metadata["final_step_label"], 5)
        self.assertEqual(metadata["n_simulations"], 5)

    def test_filter_inversion_dataset_appends_final_continuous_input_parameter(
        self,
    ) -> None:
        dataset = self.filter_results(
            observers=("density.mean",),
            theta_parameters=("alpha",),
            input_parameters=("n_agents",),
            drop_constant_theta=False,
        )
        x = dataset["x"].values
        metadata = inversion_dataset_metadata(dataset)

        expected_density = self.formatted_results()["x"].isel(step=-1).values[:, [0]]
        expected_n_agents = self.formatted_results()["theta"].sel(
            theta_parameter=["n_agents"],
        ).values
        np.testing.assert_allclose(
            x,
            np.concatenate((expected_density, expected_n_agents), axis=1),
        )
        self.assertEqual(metadata["theta_names"], ["alpha"])
        self.assertEqual(metadata["x_names"], ["density.mean", "n_agents"])
        self.assertEqual(metadata["observer_names"], ["density.mean"])
        self.assertEqual(metadata["input_parameters"], ["n_agents"])
        self.assertEqual(metadata["continuous_input_parameters"], ["n_agents"])
        self.assertEqual(metadata["discrete_input_parameters"], [])
        self.assertEqual(metadata["discrete_input_values"], [])
        self.assertEqual(metadata["x_event_shape"], [2])

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
        self.assertEqual(
            metadata["observer_names"],
            ["indegree.mean", "density.mean"],
        )
        self.assertEqual(metadata["input_parameters"], [])
        self.assertEqual(metadata["continuous_input_parameters"], [])
        self.assertEqual(metadata["discrete_input_parameters"], [])
        self.assertEqual(metadata["discrete_input_values"], [])
        self.assertEqual(metadata["x_event_shape"], [2, 2])
        self.assertEqual(metadata["x_representation"], "timeseries")
        self.assertEqual(metadata["step_labels"], [0, 5])
        self.assertEqual(metadata["n_simulations"], 5)

    def test_filter_inversion_dataset_appends_timeseries_continuous_input_parameter(
        self,
    ) -> None:
        dataset = self.filter_results(
            observers=("indegree",),
            theta_parameters=("alpha",),
            input_parameters=("n_agents",),
            x_representation="timeseries",
            drop_constant_theta=False,
        )
        x = dataset["x"].values
        metadata = inversion_dataset_metadata(dataset)

        raw = self.formatted_results()["x"].values
        expected_first = np.asarray(
            [
                [raw[0, 0, 2], 10.0],
                [raw[0, 1, 2], 10.0],
            ],
            dtype=np.float32,
        )
        self.assertEqual(x.shape, (5, 2, 2))
        np.testing.assert_allclose(x[0], expected_first)
        self.assertEqual(dataset["x"].dims, ("sample", "step", "observable"))
        self.assertEqual(metadata["theta_names"], ["alpha"])
        self.assertEqual(metadata["x_names"], ["indegree.mean", "n_agents"])
        self.assertEqual(metadata["observer_names"], ["indegree.mean"])
        self.assertEqual(metadata["input_parameters"], ["n_agents"])
        self.assertEqual(metadata["continuous_input_parameters"], ["n_agents"])
        self.assertEqual(metadata["discrete_input_parameters"], [])
        self.assertEqual(metadata["discrete_input_values"], [])
        self.assertEqual(metadata["x_event_shape"], [2, 2])

    def test_filter_inversion_dataset_one_hot_encodes_final_discrete_input(
        self,
    ) -> None:
        dataset = self.filter_results(
            observers=("density.mean",),
            theta_parameters=("alpha",),
            input_parameters=("n_agents",),
            discrete_input_parameters=("n_agents",),
            drop_constant_theta=False,
        )
        x = dataset["x"].values
        metadata = inversion_dataset_metadata(dataset)

        expected_density = self.formatted_results()["x"].isel(step=-1).values[:, [0]]
        expected_input = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            x,
            np.concatenate((expected_density, expected_input), axis=1),
        )
        self.assertEqual(
            metadata["x_names"],
            ["density.mean", "n_agents=10", "n_agents=20"],
        )
        self.assertEqual(metadata["input_parameters"], ["n_agents"])
        self.assertEqual(metadata["continuous_input_parameters"], [])
        self.assertEqual(metadata["discrete_input_parameters"], ["n_agents"])
        self.assertEqual(metadata["discrete_input_values"], [[10, 20]])
        self.assertEqual(metadata["x_event_shape"], [3])

    def test_filter_inversion_dataset_one_hot_encodes_timeseries_discrete_input(
        self,
    ) -> None:
        dataset = self.filter_results(
            observers=("indegree",),
            theta_parameters=("alpha",),
            input_parameters=("n_agents",),
            discrete_input_parameters=("n_agents",),
            x_representation="timeseries",
            drop_constant_theta=False,
        )
        x = dataset["x"].values
        metadata = inversion_dataset_metadata(dataset)

        raw = self.formatted_results()["x"].values
        expected_first = np.asarray(
            [
                [raw[0, 0, 2], 1.0, 0.0],
                [raw[0, 1, 2], 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.assertEqual(x.shape, (5, 2, 3))
        np.testing.assert_allclose(x[0], expected_first)
        self.assertEqual(
            metadata["x_names"],
            ["indegree.mean", "n_agents=10", "n_agents=20"],
        )
        self.assertEqual(metadata["discrete_input_values"], [[10, 20]])
        self.assertEqual(metadata["x_event_shape"], [2, 3])

    def test_filter_inversion_dataset_rejects_theta_input_overlap(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "input parameters cannot also be inferred theta parameters",
        ):
            self.filter_results(
                observers=("density.mean",),
                theta_parameters=("n_agents", "alpha"),
                input_parameters=("n_agents",),
                drop_constant_theta=False,
            )

    def test_filter_inversion_dataset_rejects_non_integer_discrete_input(
        self,
    ) -> None:
        raw = self.formatted_results().assign(
            theta=(
                ("sample", "theta_parameter"),
                np.asarray(
                    [
                        [10.5, 0.1, 5.0],
                        [10.0, 0.3, 5.0],
                        [20.0, 0.4, 5.0],
                        [20.0, 0.5, 5.0],
                        [20.0, 0.6, 5.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "discrete input parameter 'n_agents' contains non-integer values",
        ):
            filter_inversion_dataset(
                raw,
                observers=("density.mean",),
                theta_parameters=("alpha",),
                input_parameters=("n_agents",),
                discrete_input_parameters=("n_agents",),
                drop_constant_theta=False,
            )

    def test_filter_inversion_dataset_preserves_normalized_time_metadata(self) -> None:
        raw = self.formatted_results().assign_coords(
            step=("step", np.asarray([0.0, 0.5], dtype=np.float64)),
        ).assign_attrs(
            normalize_time=True,
            time_axis="activation_opportunities_per_agent",
            raw_step_labels=[0.0, 5.0],
            common_normalized_time_max=0.5,
            normalized_time_resolution=0.5,
        )

        dataset = filter_inversion_dataset(
            raw,
            observers=("density.mean",),
            x_representation="timeseries",
        )
        metadata = inversion_dataset_metadata(dataset)

        np.testing.assert_allclose(dataset.coords["step"].values, [0.0, 0.5])
        self.assertEqual(metadata["step_labels"], [0.0, 0.5])
        self.assertEqual(metadata["normalize_time"], True)
        self.assertEqual(metadata["time_axis"], "activation_opportunities_per_agent")
        self.assertEqual(metadata["raw_step_labels"], [0.0, 5.0])
        self.assertEqual(metadata["common_normalized_time_max"], 0.5)
        self.assertEqual(metadata["normalized_time_resolution"], 0.5)

    def test_filter_inversion_dataset_flattens_dense_policy_axes(self) -> None:
        theta = np.asarray(
            [
                [[[10.0, 0.1], [10.0, 0.2]]],
                [[[20.0, 0.3], [20.0, 0.4]]],
            ],
            dtype=np.float32,
        )
        x = np.arange(16, dtype=np.float32).reshape(2, 1, 2, 2, 2)
        x[1, 0, 1, -1, 0] = np.nan
        dense_results = xr.Dataset(
            data_vars={
                "theta": (
                    (
                        "n_agents",
                        "best_response_proportion",
                        "parameter_row",
                        "theta_parameter",
                    ),
                    theta,
                ),
                "prior_low": (("theta_parameter",), np.asarray([10.0, 0.0])),
                "prior_high": (("theta_parameter",), np.asarray([20.0, 1.0])),
                "x": (
                    (
                        "n_agents",
                        "best_response_proportion",
                        "parameter_row",
                        "step",
                        "observable",
                    ),
                    x,
                ),
            },
            coords={
                "n_agents": ("n_agents", [10, 20]),
                "best_response_proportion": ("best_response_proportion", [0.5]),
                "parameter_row": ("parameter_row", [0, 1]),
                "theta_parameter": ("theta_parameter", ["n_agents", "alpha"]),
                "observable": ("observable", ["density.mean", "distance.mean"]),
                "step": ("step", [0, 5]),
            },
            attrs={
                "result_store_path": str(self.workdir / "dense.zarr"),
                "config_path": str(self.config_path),
                "has_failures": True,
            },
        )

        dataset = filter_inversion_dataset(
            dense_results,
            observers=("density.mean",),
        )

        np.testing.assert_allclose(
            dataset["theta"].values,
            np.asarray(
                [[10.0, 0.1], [10.0, 0.2], [20.0, 0.3]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            dataset["x"].values,
            np.asarray([[2.0], [6.0], [10.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            dataset.coords["n_agents"].values,
            np.asarray([10, 10, 20]),
        )
        np.testing.assert_allclose(
            dataset.coords["best_response_proportion"].values,
            np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            dataset.coords["parameter_row"].values,
            np.asarray([0, 1, 0]),
        )
        self.assertEqual(dataset.attrs["has_failures"], True)

    def test_move_discrete_theta_to_end_reorders_and_records_metadata(self) -> None:
        raw = self.formatted_results().assign(
            theta=(
                ("sample", "theta_parameter"),
                np.asarray(
                    [
                        [10.0, 0.1, 1.0],
                        [10.0, 0.3, 2.0],
                        [20.0, 0.4, 4.0],
                        [20.0, 0.5, 1.0],
                        [20.0, 0.6, 2.0],
                    ],
                    dtype=np.float32,
                ),
            ),
            prior_low=("theta_parameter", np.asarray([10.0, 0.0, 1.0])),
            prior_high=("theta_parameter", np.asarray([20.0, 1.0, 4.0])),
        ).assign_coords(
            theta_parameter=["n_agents", "alpha", "visibility_distance"],
        )
        dataset = filter_inversion_dataset(
            raw,
            observers=("density.mean",),
            theta_parameters=("n_agents", "alpha", "visibility_distance"),
            drop_constant_theta=False,
        )

        mixed_dataset = move_discrete_theta_to_end(
            dataset,
            discrete_theta_parameters=("n_agents", "visibility_distance"),
        )
        metadata = inversion_dataset_metadata(mixed_dataset)

        self.assertEqual(
            list(mixed_dataset.coords["theta_parameter"].values),
            ["alpha", "n_agents", "visibility_distance"],
        )
        np.testing.assert_allclose(
            mixed_dataset["theta"].values,
            np.asarray(
                [
                    [0.1, 10.0, 1.0],
                    [0.3, 10.0, 2.0],
                    [0.4, 20.0, 4.0],
                    [0.5, 20.0, 1.0],
                    [0.6, 20.0, 2.0],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(metadata["continuous_theta_names"], ["alpha"])
        self.assertEqual(
            metadata["discrete_theta_names"],
            ["n_agents", "visibility_distance"],
        )
        self.assertEqual(
            metadata["theta_original_order"],
            ["n_agents", "alpha", "visibility_distance"],
        )
        self.assertEqual(metadata["discrete_theta_values"], [[10, 20], [1, 2, 4]])
        self.assertEqual(metadata["num_categories_per_discrete_theta"], [2, 3])

    def test_move_discrete_theta_to_end_keeps_unrequested_agent_count_continuous(
        self,
    ) -> None:
        raw = self.formatted_results().assign(
            theta=(
                ("sample", "theta_parameter"),
                np.asarray(
                    [
                        [10.0, 0.1, 1.0],
                        [10.0, 0.3, 2.0],
                        [20.0, 0.4, 4.0],
                        [20.0, 0.5, 1.0],
                        [20.0, 0.6, 2.0],
                    ],
                    dtype=np.float32,
                ),
            ),
            prior_low=("theta_parameter", np.asarray([10.0, 0.0, 1.0])),
            prior_high=("theta_parameter", np.asarray([20.0, 1.0, 4.0])),
        ).assign_coords(
            theta_parameter=["n_agents", "alpha", "visibility_distance"],
        )
        dataset = filter_inversion_dataset(
            raw,
            observers=("density.mean",),
            theta_parameters=("n_agents", "alpha", "visibility_distance"),
            drop_constant_theta=False,
        )

        mixed_dataset = move_discrete_theta_to_end(
            dataset,
            discrete_theta_parameters=("visibility_distance",),
        )
        metadata = inversion_dataset_metadata(mixed_dataset)

        self.assertEqual(
            list(mixed_dataset.coords["theta_parameter"].values),
            ["n_agents", "alpha", "visibility_distance"],
        )
        self.assertEqual(metadata["continuous_theta_names"], ["n_agents", "alpha"])
        self.assertEqual(metadata["discrete_theta_names"], ["visibility_distance"])
        self.assertEqual(metadata["discrete_theta_values"], [[1, 2, 4]])

    def test_move_discrete_theta_to_end_defaults_to_empty_discrete_theta(
        self,
    ) -> None:
        dataset = self.filter_results(
            observers=("density.mean",),
            theta_parameters=("n_agents", "alpha"),
            drop_constant_theta=False,
        )

        mixed_dataset = move_discrete_theta_to_end(
            dataset,
            discrete_theta_parameters=None,
        )
        metadata = inversion_dataset_metadata(mixed_dataset)

        self.assertEqual(metadata["continuous_theta_names"], ["n_agents", "alpha"])
        self.assertEqual(metadata["discrete_theta_names"], [])
        self.assertEqual(metadata["discrete_theta_values"], [])
        self.assertEqual(metadata["num_categories_per_discrete_theta"], [])

    def test_move_discrete_theta_to_end_rejects_unknown_name(self) -> None:
        dataset = self.filter_results(
            observers=("density.mean",),
            theta_parameters=("alpha", "n_agents"),
            drop_constant_theta=False,
        )

        with self.assertRaisesRegex(ValueError, "unknown discrete theta parameter"):
            move_discrete_theta_to_end(
                dataset,
                discrete_theta_parameters=("agent_count",),
            )


if __name__ == "__main__":
    unittest.main()
