from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from network_creation.model_inversion.dataset import (
    ZarrInversionDataset,
    build_row_index,
)
from network_creation.model_inversion.schema import resolve_inversion_schema
from network_creation.model_inversion.source import load_inversion_source
from network_creation.model_inversion.splits import split_inversion_rows

from .support import write_result_store, write_training_config


class ZarrBackedDatasetTests(unittest.TestCase):
    def test_source_rejects_sampled_composition_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            zarr_path = write_result_store(workdir / "run.zarr")
            root = zarr.open_group(zarr_path, mode="a")
            root.attrs["sweep_layout"] = "sampled_composition"
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=zarr_path,
            )

            with self.assertRaisesRegex(ValueError, "does not support"):
                load_inversion_source(config_path)

    def test_source_schema_and_final_dataset_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            zarr_path = write_result_store(workdir / "run.zarr")
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=zarr_path,
            )

            source = load_inversion_source(config_path)
            schema = resolve_inversion_schema(source)
            row_index = build_row_index(source, schema)
            dataset = ZarrInversionDataset(
                source=source,
                row_index=row_index,
                schema=schema,
            )
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
        self.assertEqual(schema.time_axis, "activation_opportunities_per_agent")
        np.testing.assert_array_equal(schema.n_agents, [10, 20])
        np.testing.assert_array_equal(schema.raw_step_horizons, [2, 4])
        np.testing.assert_allclose(schema.step_labels, [0.2])
        self.assertEqual(row_index.row_count, 12)
        np.testing.assert_array_equal(
            source.parameter_row_indices,
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
        )
        np.testing.assert_array_equal(
            row_index.parameter_row_indices,
            source.parameter_row_indices,
        )
        np.testing.assert_allclose(theta, [0.1, 0.01, 1.0])
        np.testing.assert_allclose(x, [2.0, 1.0, 0.0])

    def test_row_index_ignores_structural_tails_but_filters_interior_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            finite_zarr_path = write_result_store(workdir / "finite.zarr")
            finite_config_path = write_training_config(
                workdir / "finite.json",
                zarr_path=finite_zarr_path,
            )
            finite_source = load_inversion_source(finite_config_path)
            finite_schema = resolve_inversion_schema(finite_source)
            finite_rows = build_row_index(finite_source, finite_schema)

            zarr_path = write_result_store(
                workdir / "interior-nan.zarr",
                include_nan=True,
            )
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=zarr_path,
            )

            source = load_inversion_source(config_path)
            schema = resolve_inversion_schema(source)
            row_index = build_row_index(source, schema)
            dataset = ZarrInversionDataset(
                source=source,
                row_index=row_index,
                schema=schema,
            )
            split = split_inversion_rows(row_index, dataset, source.config)

        self.assertEqual(finite_rows.row_count, 12)
        self.assertEqual(row_index.row_count, 11)
        self.assertEqual(int(row_index.source_sample_ids[0]), 1)
        np.testing.assert_array_equal(
            row_index.parameter_row_indices,
            [1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
        )
        split_parameter_rows = [
            set(int(value) for value in row_index.parameter_row_indices[indices])
            for indices in (
                split.train_indices,
                split.validation_indices,
                split.diagnostic_indices,
            )
        ]
        self.assertTrue(split_parameter_rows[0].isdisjoint(split_parameter_rows[1]))
        self.assertTrue(split_parameter_rows[0].isdisjoint(split_parameter_rows[2]))
        self.assertTrue(split_parameter_rows[1].isdisjoint(split_parameter_rows[2]))
        self.assertEqual(set.union(*split_parameter_rows), set(range(6)))

    def test_timeseries_normalization_uses_population_specific_valid_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            zarr_path = write_result_store(workdir / "run.zarr")
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=zarr_path,
                x_representation="timeseries",
            )

            source = load_inversion_source(config_path)
            schema = resolve_inversion_schema(source)
            row_index = build_row_index(source, schema)
            dataset = ZarrInversionDataset(
                source=source,
                row_index=row_index,
                schema=schema,
            )
            _, x = dataset.item_arrays(0)

        self.assertEqual(schema.x_event_shape, (3, 3))
        np.testing.assert_allclose(schema.step_labels, [0.0, 0.1, 0.2])
        np.testing.assert_allclose(
            x,
            np.asarray(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_timeseries_ignores_unselected_terminal_agent_payoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            zarr_path = write_result_store(
                workdir / "run.zarr",
                include_terminal_agent_payoff=True,
            )
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=zarr_path,
                x_representation="timeseries",
            )

            source = load_inversion_source(config_path)
            schema = resolve_inversion_schema(source)
            row_index = build_row_index(source, schema)
            dataset = ZarrInversionDataset(
                source=source,
                row_index=row_index,
                schema=schema,
            )
            _, x = dataset.item_arrays(0)

        self.assertIn("agent_payoff.runs", source.observed_output_arrays)
        self.assertEqual(schema.observed_outputs, ("density.mean",))
        self.assertEqual(row_index.row_count, 12)
        self.assertEqual(schema.x_event_shape, (3, 3))
        np.testing.assert_allclose(
            x,
            np.asarray(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_each_selected_output_keeps_its_sampling_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            zarr_path = write_result_store(workdir / "run.zarr")
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=zarr_path,
            )
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            payload["data"]["observed_outputs"] = [
                "density.mean",
                "distance.mean",
            ]
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            source = load_inversion_source(config_path)
            schema = resolve_inversion_schema(source)
            row_index = build_row_index(source, schema)
            dataset = ZarrInversionDataset(
                source=source,
                row_index=row_index,
                schema=schema,
            )
            _, x = dataset.item_arrays(0)

        self.assertEqual(len(schema.output_samplings), 2)
        self.assertEqual(schema.output_samplings[0], schema.output_samplings[1])
        np.testing.assert_allclose(x, [2.0, 12.0, 1.0, 0.0])

    def test_out_of_order_physical_rows_keep_logical_population_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            physical_order = (6, 0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5)
            zarr_path = write_result_store(
                workdir / "run.zarr",
                physical_input_order=physical_order,
            )
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=zarr_path,
            )

            source = load_inversion_source(config_path)
            schema = resolve_inversion_schema(source)
            row_index = build_row_index(source, schema)
            dataset = ZarrInversionDataset(
                source=source,
                row_index=row_index,
                schema=schema,
            )
            _, first_population_x = dataset.item_arrays_from_source_position(0)
            _, second_population_x = dataset.item_arrays_from_source_position(6)

        np.testing.assert_array_equal(source.input_index, np.arange(12))
        np.testing.assert_array_equal(
            source.result_rows,
            [1, 3, 5, 7, 9, 11, 0, 2, 4, 6, 8, 10],
        )
        np.testing.assert_array_equal(
            source.population_indices,
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        )
        np.testing.assert_array_equal(
            source.parameter_row_indices,
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
        )
        np.testing.assert_array_equal(
            row_index.parameter_row_indices,
            source.parameter_row_indices,
        )
        np.testing.assert_allclose(first_population_x, [2.0, 1.0, 0.0])
        np.testing.assert_allclose(second_population_x, [10.0, 0.0, 1.0])

    def test_source_requires_raw_step_horizons(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            zarr_path = write_result_store(workdir / "run.zarr")
            root = zarr.open_group(zarr_path, mode="a")
            del root["metadata"]["raw_step_horizons"]
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=zarr_path,
            )

            with self.assertRaises(KeyError):
                load_inversion_source(config_path)


if __name__ == "__main__":
    unittest.main()
