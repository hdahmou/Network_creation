from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from network_creation.model_inversion.artifacts import write_inversion_artifact
from network_creation.model_inversion.config import load_inversion_config
from network_creation.model_inversion.dataset import (
    ZarrInversionDataset,
    build_row_index,
)
from network_creation.model_inversion.posterior import load_posterior_artifact
from network_creation.model_inversion.schema import resolve_inversion_schema
from network_creation.model_inversion.source import load_inversion_source
from network_creation.model_inversion.splits import (
    select_prior_samples,
    split_inversion_rows,
)
from network_creation.model_inversion.training import EstimatorTrainingResult

from .support import write_result_store, write_training_config


class InversionConfigAndArtifactTests(unittest.TestCase):
    def test_checked_training_configs_use_the_strict_schema(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        config_names = (
            "inversion.best_response.timeseries.visibility_distance_continuous.json",
            "inversion.best_response.timeseries.visibility_distance_discrete.json",
            "inversion.best_response.final.visibility_distance_discrete.json",
        )

        for config_name in config_names:
            with self.subTest(config=config_name):
                config = load_inversion_config(project_root / "configs" / config_name)
                self.assertEqual(config.output.training_data_sample_count, 4096)

    def test_load_inversion_config_resolves_new_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            write_result_store(workdir / "run.zarr")
            config_path = write_training_config(
                workdir / "configs" / "train.json",
                zarr_path=Path("../run.zarr"),
                output_root=Path("../out"),
                artifact_name="manual",
                x_representation="timeseries",
            )

            config = load_inversion_config(config_path)

            self.assertEqual(config.data.zarr_path, (config_path.parent / "../run.zarr").resolve())
            self.assertEqual(config.data.observed_outputs, ("density.mean",))
            self.assertEqual(
                config.data.continuous_theta_names,
                ("outside_optimism", "exploration", "constant_beta"),
            )
            self.assertEqual(config.data.discrete_theta_names, ("visibility_distance",))
            self.assertEqual(config.data.discrete_input_names, ("n_agents",))
            self.assertEqual(config.data.x_representation, "timeseries")
            self.assertEqual(config.model.density_estimator.flow_model, "nsf")
            self.assertEqual(config.training.batch_size, 4)
            self.assertEqual(config.output.artifact_name, "manual")

    def test_load_inversion_config_rejects_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            zarr_path = write_result_store(workdir / "run.zarr")
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=zarr_path,
            )
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            payload["training"]["batch_szie"] = 4
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "batch_szie"):
                load_inversion_config(config_path)

    def test_load_inversion_config_rejects_removed_normalize_time_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            zarr_path = write_result_store(workdir / "run.zarr")
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=zarr_path,
            )
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            payload["data"]["normalize_time"] = True
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "normalize_time"):
                load_inversion_config(config_path)

    def test_load_inversion_config_rejects_incompatible_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=workdir / "run.zarr",
                x_representation="timeseries",
            )
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            payload["model"]["embedding"]["kind"] = "identity"
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "timeseries x requires"):
                load_inversion_config(config_path)

    def test_split_is_disjoint_by_parameter_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            zarr_path = write_result_store(workdir / "run.zarr")
            config_path = write_training_config(workdir / "train.json", zarr_path=zarr_path)
            source = load_inversion_source(config_path)
            schema = resolve_inversion_schema(source)
            row_index = build_row_index(source, schema)
            dataset = ZarrInversionDataset(
                source=source,
                row_index=row_index,
                schema=schema,
            )

            split = split_inversion_rows(row_index, dataset, source.config)
            prior = select_prior_samples(dataset, split, source.config)
            calibration_theta = np.stack(
                [dataset.theta(int(index)) for index in split.calibration_indices]
            )

            def parameter_rows(indices: np.ndarray) -> set[int]:
                return {
                    int(value) for value in row_index.parameter_row_indices[indices]
                }

            def theta_rows(indices: np.ndarray) -> set[tuple[float, ...]]:
                return {
                    tuple(float(value) for value in dataset.theta(int(index)))
                    for index in indices
                }

            def n_agents_values(indices: np.ndarray) -> set[int]:
                source_positions = row_index.kept_indices[indices]
                populations = source.population_indices[source_positions]
                return {int(schema.n_agents[int(value)]) for value in populations}

            train_rows = set(int(value) for value in split.train_indices)
            validation_rows = set(int(value) for value in split.validation_indices)
            diagnostic_rows = set(int(value) for value in split.diagnostic_indices)
            train_parameters = parameter_rows(split.train_indices)
            validation_parameters = parameter_rows(split.validation_indices)
            diagnostic_parameters = parameter_rows(split.diagnostic_indices)
            train_theta = theta_rows(split.train_indices)
            validation_theta = theta_rows(split.validation_indices)
            diagnostic_theta = theta_rows(split.diagnostic_indices)

            self.assertEqual(len(split.diagnostic_indices), 4)
            self.assertEqual(train_parameters, {0, 4})
            self.assertEqual(validation_parameters, {1, 5})
            self.assertEqual(diagnostic_parameters, {2, 3})
            self.assertTrue(train_rows.isdisjoint(validation_rows))
            self.assertTrue(train_rows.isdisjoint(diagnostic_rows))
            self.assertTrue(validation_rows.isdisjoint(diagnostic_rows))
            self.assertEqual(
                train_rows | validation_rows | diagnostic_rows,
                set(range(len(dataset))),
            )
            self.assertTrue(train_parameters.isdisjoint(validation_parameters))
            self.assertTrue(train_parameters.isdisjoint(diagnostic_parameters))
            self.assertTrue(validation_parameters.isdisjoint(diagnostic_parameters))
            self.assertTrue(train_theta.isdisjoint(validation_theta))
            self.assertTrue(train_theta.isdisjoint(diagnostic_theta))
            self.assertTrue(validation_theta.isdisjoint(diagnostic_theta))
            self.assertEqual(n_agents_values(split.train_indices), {10, 20})
            self.assertEqual(n_agents_values(split.validation_indices), {10, 20})
            self.assertEqual(n_agents_values(split.diagnostic_indices), {10, 20})
            self.assertTrue(
                parameter_rows(split.calibration_indices).issubset(train_parameters)
            )
            prior_parameter_rows = {
                int(value)
                for value in source.metadata_arrays["input_coordinates"][
                    prior.source_indices,
                    2,
                ]
            }
            self.assertTrue(prior_parameter_rows.issubset(train_parameters))
            self.assertEqual(set(calibration_theta[:, -1].astype(int).tolist()), {1, 2})
            self.assertEqual(prior.theta.shape[0], 3)
            self.assertEqual(prior.theta.shape[1], len(schema.theta_order))

    def test_write_inversion_artifact_uses_new_metadata_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            zarr_path = write_result_store(workdir / "run.zarr")
            config_path = write_training_config(workdir / "train.json", zarr_path=zarr_path)
            source = load_inversion_source(config_path)
            schema = resolve_inversion_schema(source)
            row_index = build_row_index(source, schema)
            dataset = ZarrInversionDataset(
                source=source,
                row_index=row_index,
                schema=schema,
            )
            split = split_inversion_rows(row_index, dataset, source.config)
            prior = select_prior_samples(dataset, split, source.config)
            training = EstimatorTrainingResult(
                estimator=object(),
                epochs_trained=1,
                train_loss=[1.0],
                validation_loss=[1.5],
                best_validation_loss=1.5,
                final_train_loss=1.0,
                final_validation_loss=1.5,
                best_validation_log_prob=-1.5,
                final_validation_log_prob=-1.5,
            )

            summary = write_inversion_artifact(
                posterior={"posterior": True},
                source=source,
                schema=schema,
                row_index=row_index,
                split=split,
                prior_samples=prior,
                training=training,
                config=source.config,
                dataset=dataset,
            )
            artifact_dir = Path(summary.artifact_dir)
            metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

            self.assertTrue((artifact_dir / "posterior.pkl").exists())
            self.assertTrue((artifact_dir / "diagnostic_data.npz").exists())
            self.assertTrue((artifact_dir / "prior_samples.npz").exists())
            self.assertTrue((artifact_dir / "training_data.npz").exists())
            self.assertEqual(metadata["dataset"]["observed_outputs"], ["density.mean"])
            self.assertEqual(
                metadata["dataset"]["x_components"],
                ["density.mean", "n_agents=10", "n_agents=20"],
            )
            self.assertEqual(
                metadata["dataset"]["continuous_theta_names"],
                ["outside_optimism", "exploration"],
            )
            self.assertEqual(
                metadata["dataset"]["discrete_theta_names"],
                ["visibility_distance"],
            )
            self.assertEqual(
                metadata["dataset"]["time_axis"],
                "activation_opportunities_per_agent",
            )
            self.assertEqual(metadata["dataset"]["n_agents"], [10, 20])
            self.assertEqual(metadata["dataset"]["raw_step_horizons"], [2, 4])
            self.assertEqual(metadata["dataset"]["n_actors"], 1)
            self.assertEqual(
                metadata["dataset"]["output_samplings"],
                [{"kind": "periodic", "every": 1}],
            )
            self.assertEqual(metadata["dataset"]["step_labels"], [0.2])
            self.assertNotIn("normalize_time", metadata["dataset"])
            self.assertNotIn("raw_step_labels", metadata["dataset"])
            self.assertNotIn("final_step_label", metadata["dataset"])
            self.assertEqual(metadata["sbi"]["method"], "MNPE")
            self.assertEqual(metadata["sbi"]["prior_samples"]["count"], 3)
            self.assertEqual(
                Path(metadata["dataset"]["training_config_path"]),
                config_path.resolve(),
            )
            self.assertEqual(
                Path(metadata["dataset"]["simulation_config_path"]),
                zarr_path / "run_config.json",
            )
            artifact = load_posterior_artifact(artifact_dir)
            np.testing.assert_allclose(artifact.prior_samples, prior.theta)
            np.testing.assert_array_equal(
                artifact.prior_sample_indices,
                prior.source_indices,
            )
            with np.load(artifact_dir / "diagnostic_data.npz") as data:
                self.assertIn("source_sample_ids", data)
                self.assertIn("x_components", data)


if __name__ == "__main__":
    unittest.main()
