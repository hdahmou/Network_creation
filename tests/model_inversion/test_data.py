from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from network_creation.model_inversion.dataset import (
    build_row_index,
    make_zarr_inversion_dataset,
)
from network_creation.model_inversion.schema import resolve_inversion_schema
from network_creation.model_inversion.source import load_inversion_source

from .support import write_result_store, write_training_config


class ZarrBackedDatasetTests(unittest.TestCase):
    def test_source_schema_and_final_dataset_use_new_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            zarr_path = write_result_store(workdir / "run.zarr")
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=zarr_path,
                normalize_time=False,
            )

            source = load_inversion_source(config_path)
            schema = resolve_inversion_schema(source)
            row_index = build_row_index(source, schema)
            dataset = make_zarr_inversion_dataset(source, row_index, schema)
            theta, x = dataset.item_arrays(0)

        self.assertEqual(schema.observed_outputs, ("density.mean",))
        self.assertIsInstance(source.metadata_arrays["parameters"], np.ndarray)
        self.assertEqual(schema.continuous_theta_names, ("outside_optimism", "exploration"))
        self.assertEqual(schema.discrete_theta_names, ("visibility_distance",))
        self.assertEqual(schema.constant_theta, {"constant_beta": 5.0})
        self.assertEqual(schema.discrete_input_values, {"n_agents": (10, 20)})
        self.assertEqual(schema.discrete_theta_values, {"visibility_distance": (1, 2)})
        self.assertEqual(schema.x_components, ("density.mean", "n_agents=10", "n_agents=20"))
        self.assertEqual(schema.x_event_shape, (3,))
        self.assertEqual(row_index.row_count, 12)
        np.testing.assert_allclose(theta, [0.1, 0.01, 1.0])
        np.testing.assert_allclose(x, [2.0, 1.0, 0.0])

    def test_row_index_filters_non_finite_observed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            zarr_path = write_result_store(workdir / "run.zarr", include_nan=True)
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=zarr_path,
                normalize_time=False,
            )

            source = load_inversion_source(config_path)
            schema = resolve_inversion_schema(source)
            row_index = build_row_index(source, schema)

        self.assertEqual(row_index.row_count, 11)
        self.assertEqual(int(row_index.source_sample_ids[0]), 1)

    def test_timeseries_normalization_is_applied_per_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            zarr_path = write_result_store(workdir / "run.zarr")
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=zarr_path,
                x_representation="timeseries",
                normalize_time=True,
            )

            source = load_inversion_source(config_path)
            schema = resolve_inversion_schema(source)
            row_index = build_row_index(source, schema)
            dataset = make_zarr_inversion_dataset(source, row_index, schema)
            _, x = dataset.item_arrays(0)

        self.assertEqual(schema.x_event_shape, (2, 3))
        np.testing.assert_allclose(schema.step_labels, [0.0, 0.1])
        np.testing.assert_allclose(
            x,
            np.asarray([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
