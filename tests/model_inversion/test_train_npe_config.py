from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_training_script() -> ModuleType:
    script_path = PROJECT_ROOT / "scripts" / "train_npe_from_zarr.py"
    spec = importlib.util.spec_from_file_location("train_npe_from_zarr", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainNPEConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.script = load_training_script()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.config_dir = Path(self.temp_dir.name) / "configs"
        self.config_dir.mkdir(parents=True)

    def write_config(self, payload: dict[str, object]) -> Path:
        config_path = self.config_dir / "train_config.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        return config_path

    def assert_project_relative_metadata_path(
        self,
        metadata_value: object,
        expected_path: Path,
    ) -> None:
        self.assertIsInstance(metadata_value, str)
        self.assertFalse(Path(str(metadata_value)).is_absolute())
        self.assertEqual(
            metadata_value,
            os.path.relpath(expected_path, PROJECT_ROOT).replace(os.sep, "/"),
        )

    def fake_training_result(self) -> SimpleNamespace:
        return SimpleNamespace(
            posterior={"posterior": True},
            training_summary={
                "epochs_trained": 3,
                "training_loss": [3.0, 2.0],
                "validation_loss": [2.5, 2.25],
                "best_validation_loss": 2.25,
                "final_training_loss": 2.0,
                "final_validation_loss": 2.25,
                "best_validation_log_prob": -2.25,
                "final_validation_log_prob": -2.25,
            },
        )

    def test_load_train_config_resolves_paths_relative_to_config_file(self) -> None:
        config_path = self.write_config(
            {
                "dataset": {
                    "zarr_path": "../artifacts/run.zarr",
                    "observers": ["density.mean", "distance.mean"],
                    "theta_parameters": ["n_agents", "exploration"],
                    "input_parameter_names": [],
                    "drop_constant_theta": False,
                    "x_representation": "timeseries",
                },
                "artifact": {
                    "output_root": "../artifacts/model_inversion",
                    "run_name": "manual-run",
                    "save_training_data": True,
                },
                "diagnostics": {
                    "reserve_count": 7,
                },
                "density_estimator": {
                    "model": "maf",
                    "hidden_features": 12,
                    "num_transforms": 3,
                    "num_bins": 8,
                    "z_score_theta": "none",
                    "z_score_x": "structured",
                    "sbi_kwargs": {
                        "dropout_probability": 0.05,
                        "tail_bound": 5.0,
                    },
                },
                "embedding": {
                    "kind": "rnn",
                    "output_dim": 16,
                    "rnn_hidden_size": 24,
                    "rnn_num_layers": 2,
                    "rnn_type": "lstm",
                    "rnn_bidirectional": True,
                },
                "training": {
                    "seed": 7,
                    "device": "cpu",
                    "training_batch_size": 32,
                    "learning_rate": 0.01,
                    "validation_fraction": 0.2,
                    "stop_after_epochs": 4,
                    "max_num_epochs": 9,
                    "clip_max_norm": None,
                    "show_progress_bars": False,
                    "dataloader_kwargs": {
                        "num_workers": 0,
                        "pin_memory": False,
                    },
                },
            }
        )

        config = self.script.load_train_config(config_path)

        self.assertEqual(
            config.zarr_path,
            (self.config_dir / "../artifacts/run.zarr").resolve(),
        )
        self.assertEqual(
            config.output_root,
            (self.config_dir / "../artifacts/model_inversion").resolve(),
        )
        self.assertEqual(config.observers, ("density.mean", "distance.mean"))
        self.assertEqual(config.theta_parameters, ("n_agents", "exploration"))
        self.assertEqual(config.input_parameter_names, ())
        self.assertFalse(config.drop_constant_theta)
        self.assertEqual(config.x_representation, "timeseries")
        self.assertTrue(config.normalize_time)
        self.assertEqual(config.run_name, "manual-run")
        self.assertTrue(config.save_training_data)
        self.assertEqual(config.diagnostics.reserve_count, 7)
        self.assertIsNone(config.discrete_theta_parameters)
        self.assertEqual(config.prior_sample_count, 4096)
        self.assertEqual(config.density.model, "maf")
        self.assertEqual(config.density.flow_model, "nsf")
        self.assertEqual(config.density.hidden_features, 12)
        self.assertEqual(
            config.density.sbi_kwargs,
            {"dropout_probability": 0.05, "tail_bound": 5.0},
        )
        self.assertEqual(config.embedding.kind, "rnn")
        self.assertEqual(config.embedding.rnn_type, "lstm")
        self.assertTrue(config.embedding.rnn_bidirectional)
        self.assertEqual(config.training.seed, 7)
        self.assertIsNone(config.training.clip_max_norm)
        self.assertFalse(config.training.show_progress_bars)
        self.assertEqual(
            config.training.dataloader_kwargs,
            {"num_workers": 0, "pin_memory": False},
        )

    def test_load_train_config_allows_explicit_raw_time(self) -> None:
        config_path = self.write_config(
            {
                "dataset": {
                    "zarr_path": "../artifacts/run.zarr",
                    "observers": ["density.mean"],
                    "normalize_time": False,
                },
                "artifact": {
                    "output_root": "../artifacts/model_inversion",
                },
                "diagnostics": {
                    "reserve_count": 1,
                },
                "density_estimator": {},
                "embedding": {"kind": "identity"},
                "training": {},
            }
        )

        config = self.script.load_train_config(config_path)

        self.assertFalse(config.normalize_time)
        self.assertEqual(config.input_parameter_names, ())

    def test_baseline_config_loads(self) -> None:
        baseline_path = (
            PROJECT_ROOT
            / "configs"
            / "model_inversion.baseline.train_npe_config.json"
        )
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))

        config = self.script.load_train_config(baseline_path)

        self.assertEqual(
            config.zarr_path,
            (baseline_path.parent / payload["dataset"]["zarr_path"]).resolve(),
        )
        self.assertEqual(
            config.output_root,
            (PROJECT_ROOT / "artifacts" / "model_inversion").resolve(),
        )
        self.assertEqual(
            config.observers,
            (
                "central_point_of_dominance.mean",
                "indegree.mean",
                "distance.mean",
            ),
        )
        self.assertEqual(
            config.theta_parameters,
            (
                "n_agents",
                "visibility_distance",
                "outside_optimism",
                "exploration",
            ),
        )
        self.assertEqual(config.x_representation, "final")
        self.assertTrue(config.normalize_time)
        self.assertEqual(config.input_parameter_names, ())
        self.assertEqual(payload["dataset"]["normalize_time"], True)
        self.assertEqual(
            config.discrete_theta_parameters,
            ("n_agents", "visibility_distance"),
        )
        self.assertEqual(config.prior_sample_count, 4096)
        self.assertEqual(config.diagnostics.reserve_count, 200)
        self.assertEqual(config.density.model, "mnpe")
        self.assertEqual(config.density.flow_model, "nsf")
        self.assertEqual(config.embedding.kind, "identity")
        self.assertEqual(payload["embedding"], {"kind": "identity"})
        self.assertNotIn("rnn_hidden_size", payload["embedding"])
        self.assertNotIn("rnn_num_layers", payload["embedding"])
        self.assertNotIn("rnn_type", payload["embedding"])
        self.assertNotIn("rnn_bidirectional", payload["embedding"])
        self.assertEqual(config.training.training_batch_size, 200)

    def test_centered_mean_welfare_rnn_config_loads_mixed_parameters(self) -> None:
        config_path = self.write_config(
            {
                "dataset": {
                    "zarr_path": "../artifacts/run.zarr",
                    "observers": ["centered_mean_welfare.mean"],
                    "theta_parameters": [
                        "visibility_distance",
                        "outside_optimism",
                        "exploration",
                    ],
                    "input_parameter_names": ["n_agents"],
                    "discrete_theta_parameters": ["visibility_distance"],
                    "x_representation": "timeseries",
                    "normalize_time": True,
                },
                "artifact": {
                    "output_root": "../artifacts/model_inversion",
                },
                "diagnostics": {
                    "reserve_count": 1,
                },
                "density_estimator": {
                    "model": "mnpe",
                },
                "embedding": {
                    "kind": "rnn",
                },
                "training": {},
            }
        )

        config = self.script.load_train_config(config_path)

        self.assertEqual(
            config.theta_parameters,
            (
                "visibility_distance",
                "outside_optimism",
                "exploration",
            ),
        )
        self.assertEqual(
            config.discrete_theta_parameters,
            ("visibility_distance",),
        )
        self.assertEqual(config.input_parameter_names, ("n_agents",))
        self.assertEqual(config.x_representation, "timeseries")
        self.assertTrue(config.normalize_time)
        self.assertEqual(config.embedding.kind, "rnn")
        self.assertEqual(config.density.model, "mnpe")

    def test_split_diagnostic_holdout_rejects_reserve_count_too_large(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "diagnostics.reserve_count must be < n_simulations",
        ):
            self.script.split_diagnostic_holdout(
                n_simulations=3,
                reserve_count=3,
                seed=1,
            )

    def test_write_run_artifacts_writes_expected_files(self) -> None:
        artifact_dir = Path(self.temp_dir.name) / "artifact-writer"
        artifact_dir.mkdir()
        config = self.script.TrainNPEConfig(
            zarr_path=(self.config_dir / "../artifacts/run.zarr").resolve(),
            observers=("density.mean",),
            theta_parameters=("alpha",),
            drop_constant_theta=False,
            x_representation="final",
            output_root=Path(self.temp_dir.name),
            run_name="artifact-writer",
            save_training_data=True,
            density=self.script.DensityEstimatorConfig(model="nsf"),
            embedding=self.script.EmbeddingConfig(kind="identity"),
            training=self.script.NPETrainingConfig(seed=1),
            diagnostics=self.script.DiagnosticsConfig(reserve_count=1),
        )
        run = {
            "config": config,
            "dataset_metadata": {
                "result_store_path": str(config.zarr_path),
                "config_path": str(self.config_dir / "run_config.json"),
                "n_simulations": 3,
                "theta_names": ["alpha"],
                "x_names": ["density.mean"],
                "x_event_shape": [1],
                "x_representation": "final",
            },
            "theta": np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32),
            "x": np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
            "train_theta": np.asarray([[0.1], [0.3]], dtype=np.float32),
            "train_x": np.asarray([[1.0], [3.0]], dtype=np.float32),
            "diagnostic_theta": np.asarray([[0.2]], dtype=np.float32),
            "diagnostic_x": np.asarray([[2.0]], dtype=np.float32),
            "diagnostic_source_sample": np.asarray([1], dtype=np.int64),
            "source_sample": np.asarray([0, 1, 2], dtype=np.int64),
            "kept_mask": np.asarray([True, True, True], dtype=np.bool_),
            "train_indices": np.asarray([0, 2], dtype=np.int64),
            "diagnostic_indices": np.asarray([1], dtype=np.int64),
            "category_counts": (),
            "sbi_method": "NPE",
            "prior_samples": np.asarray([[0.1], [0.3]], dtype=np.float32),
            "prior_sample_indices": np.asarray([0, 2], dtype=np.int64),
            "posterior": {"posterior": True},
            "training_summary": {
                "epochs_trained": 2,
                "best_validation_loss": 1.5,
                "best_validation_log_prob": -1.5,
            },
            "artifact_dir": artifact_dir,
            "posterior_path": artifact_dir / "posterior.pkl",
            "metadata_path": artifact_dir / "metadata.json",
            "tensorboard_log_dir": artifact_dir / "sbi_logs",
            "training_metrics_path": artifact_dir / "training_metrics.json",
            "diagnostic_data_path": artifact_dir / "diagnostic_data.npz",
        }

        self.script._write_run_artifacts(run)

        self.assertEqual(run["training_data_path"], artifact_dir / "training_data.npz")
        self.assertTrue(run["metadata_path"].exists())
        self.assertTrue(run["training_metrics_path"].exists())
        self.assertTrue(run["posterior_path"].exists())
        self.assertTrue(run["diagnostic_data_path"].exists())
        self.assertTrue(run["training_data_path"].exists())
        self.assertEqual(
            json.loads(run["training_metrics_path"].read_text(encoding="utf-8")),
            run["training_summary"],
        )
        metadata = json.loads(run["metadata_path"].read_text(encoding="utf-8"))
        self.assertEqual(
            metadata["sbi"]["training_metrics_file"],
            "training_metrics.json",
        )
        self.assertEqual(metadata["diagnostics"]["file"], "diagnostic_data.npz")
        with np.load(run["diagnostic_data_path"]) as diagnostic_data:
            np.testing.assert_allclose(diagnostic_data["theta"], [[0.2]])
            np.testing.assert_array_equal(diagnostic_data["source_sample"], [1])
        with np.load(run["training_data_path"]) as training_data:
            np.testing.assert_allclose(training_data["theta"], [[0.1], [0.3]])
            np.testing.assert_array_equal(training_data["source_sample"], [0, 2])
        with run["posterior_path"].open("rb") as handle:
            self.assertEqual(pickle.load(handle), {"posterior": True})

    def test_train_from_config_logs_progress(self) -> None:
        zarr_path = self.config_dir / "../artifacts/run.zarr"
        output_root = Path(self.temp_dir.name) / "model_inversion"
        config = self.script.TrainNPEConfig(
            zarr_path=zarr_path.resolve(),
            observers=("density.mean",),
            theta_parameters=("alpha",),
            drop_constant_theta=False,
            x_representation="final",
            output_root=output_root,
            run_name="logging-run",
            save_training_data=False,
            density=self.script.DensityEstimatorConfig(
                model="nsf",
                sbi_kwargs={"dropout_probability": 0.05},
            ),
            embedding=self.script.EmbeddingConfig(kind="identity"),
            training=self.script.NPETrainingConfig(
                seed=1,
                device="cpu",
                training_batch_size=2,
                max_num_epochs=3,
                dataloader_kwargs={"num_workers": 0},
            ),
            diagnostics=self.script.DiagnosticsConfig(reserve_count=1),
        )

        raw_results = xr.Dataset(
            data_vars={
                "theta": (
                    ("sample", "theta_parameter"),
                    np.asarray([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32),
                ),
                "prior_low": ("theta_parameter", np.asarray([0.0], dtype=np.float32)),
                "prior_high": ("theta_parameter", np.asarray([1.0], dtype=np.float32)),
                "x": (
                    ("sample", "step", "observable"),
                    np.asarray(
                        [
                            [[1.0], [2.0]],
                            [[3.0], [4.0]],
                            [[5.0], [6.0]],
                            [[7.0], [8.0]],
                        ],
                        dtype=np.float32,
                    ),
                ),
            },
            coords={
                "sample": np.asarray([0, 1, 2, 3], dtype=np.int64),
                "theta_parameter": ("theta_parameter", ["alpha"]),
                "observable": ("observable", ["density.mean"]),
                "step": np.asarray([0, 1], dtype=np.int64),
            },
            attrs={
                "result_store_path": str(config.zarr_path),
                "config_path": str(self.config_dir / "run_config.json"),
                "has_failures": False,
            },
        )

        with (
            patch.object(
                self.script,
                "load_policy_results",
                return_value=raw_results,
            ) as load_policy_results,
            patch.object(
                self.script,
                "train_amortized_npe",
                return_value=self.fake_training_result(),
            ) as train_amortized_npe,
            self.assertLogs(self.script.LOGGER.name, level="INFO") as captured,
        ):
            result = self.script.train_from_config(config)

        artifact_dir = output_root / "logging-run"
        tensorboard_log_dir = artifact_dir / "sbi_logs"
        logs = "\n".join(captured.output)
        load_policy_results.assert_called_once_with(
            config.zarr_path,
            stats=config.observers,
            normalize_time=config.normalize_time,
        )
        self.assertIn("[data] Loading zarr dataset", logs)
        self.assertIn("normalize_time=True", logs)
        self.assertIn("[data] Filtering inversion dataset", logs)
        self.assertIn("[artifacts] Creating artifact directory", logs)
        self.assertIn("[training] Starting NPE training", logs)
        self.assertIn("[training] Finished NPE training", logs)
        self.assertIn("[artifacts] Writing metadata", logs)
        self.assertIn("[artifacts] Writing training metrics", logs)
        self.assertIn("[artifacts] Writing posterior", logs)
        self.assertIn("[artifacts] Skipping training data write", logs)
        self.assertIn("[complete] Training run complete", logs)
        self.assertIn(str(config.zarr_path), logs)
        self.assertIn("theta_shape=(4, 1)", logs)
        self.assertIn("x_shape=(4, 1)", logs)
        self.assertIn("train_count=3", logs)
        self.assertIn("diagnostic_count=1", logs)
        self.assertIn("device=cpu", logs)
        self.assertIn("batch_size=2", logs)
        self.assertIn("max_num_epochs=3", logs)
        self.assertIn(f"tensorboard_log_dir={tensorboard_log_dir}", logs)
        train_amortized_npe.assert_called_once()
        self.assertEqual(
            train_amortized_npe.call_args.kwargs["tensorboard_log_dir"],
            tensorboard_log_dir,
        )
        self.assertEqual(
            train_amortized_npe.call_args.kwargs["num_categories_per_discrete_theta"],
            (),
        )
        train_indices, diagnostic_indices = self.script.split_diagnostic_holdout(
            n_simulations=4,
            reserve_count=1,
            seed=1,
        )
        expected_theta = np.asarray([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
        expected_x = np.asarray([[2.0], [4.0], [6.0], [8.0]], dtype=np.float32)
        np.testing.assert_allclose(
            train_amortized_npe.call_args.kwargs["theta"],
            expected_theta[train_indices],
        )
        np.testing.assert_allclose(
            train_amortized_npe.call_args.kwargs["x"],
            expected_x[train_indices],
        )
        self.assertEqual(result["artifact_dir"], str(artifact_dir))
        self.assertEqual(result["tensorboard_log_dir"], str(tensorboard_log_dir))
        self.assertEqual(result["training_metrics_file"], "training_metrics.json")
        self.assertEqual(result["best_validation_loss"], 2.25)
        self.assertEqual(result["final_validation_loss"], 2.25)
        self.assertEqual(result["best_validation_log_prob"], -2.25)
        self.assertEqual(result["final_validation_log_prob"], -2.25)
        self.assertEqual(result["train_count"], 3)
        self.assertEqual(result["diagnostic_count"], 1)
        self.assertEqual(result["diagnostic_data_file"], "diagnostic_data.npz")
        self.assertEqual(result["normalize_time"], True)
        metadata = json.loads(
            (artifact_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["sbi"]["tensorboard_log_dir"], "sbi_logs")
        self.assertEqual(
            metadata["sbi"]["training_metrics_file"],
            "training_metrics.json",
        )
        self.assertEqual(metadata["sbi"]["method"], "NPE")
        self.assertEqual(metadata["sbi"]["prior_samples"]["count"], 3)
        self.assertEqual(
            metadata["density_estimator"]["sbi_kwargs"],
            {"dropout_probability": 0.05},
        )
        self.assertEqual(
            metadata["training"]["dataloader_kwargs"],
            {"num_workers": 0},
        )
        self.assertEqual(metadata["diagnostics"]["file"], "diagnostic_data.npz")
        self.assertEqual(metadata["diagnostics"]["reserve_count"], 1)
        self.assertEqual(metadata["diagnostics"]["train_count"], 3)
        self.assertEqual(metadata["diagnostics"]["diagnostic_count"], 1)
        self.assertEqual(metadata["dataset"]["train_count"], 3)
        self.assertEqual(metadata["dataset"]["diagnostic_count"], 1)
        self.assertEqual(metadata["dataset"]["path_base"], "project_root")
        self.assertEqual(metadata["dataset"]["normalize_time"], True)
        self.assertEqual(
            metadata["dataset"]["time_axis"],
            "activation_opportunities_per_agent",
        )
        self.assert_project_relative_metadata_path(
            metadata["dataset"]["result_store_path"],
            config.zarr_path,
        )
        self.assert_project_relative_metadata_path(
            metadata["dataset"]["config_path"],
            self.config_dir / "run_config.json",
        )
        training_metrics = json.loads(
            (artifact_dir / "training_metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(training_metrics["epochs_trained"], 3)
        self.assertEqual(training_metrics["validation_loss"], [2.5, 2.25])
        self.assertEqual(training_metrics["best_validation_log_prob"], -2.25)
        with np.load(artifact_dir / "diagnostic_data.npz") as diagnostic_data:
            np.testing.assert_allclose(
                diagnostic_data["theta"],
                expected_theta[diagnostic_indices],
            )
            np.testing.assert_allclose(
                diagnostic_data["x"],
                expected_x[diagnostic_indices],
            )
            np.testing.assert_array_equal(
                diagnostic_data["source_sample"],
                np.asarray([0, 1, 2, 3], dtype=np.int64)[diagnostic_indices],
            )
        with (artifact_dir / "posterior.pkl").open("rb") as handle:
            saved_posterior = pickle.load(handle)
        np.testing.assert_allclose(
            saved_posterior["prior_samples"],
            expected_theta[train_indices],
        )

    def test_train_from_config_uses_mnpe_for_agent_count_and_visibility_distance(
        self,
    ) -> None:
        output_root = Path(self.temp_dir.name) / "model_inversion_mixed"
        config = self.script.TrainNPEConfig(
            zarr_path=(self.config_dir / "../artifacts/run.zarr").resolve(),
            observers=("density.mean",),
            theta_parameters=("n_agents", "alpha", "visibility_distance"),
            drop_constant_theta=False,
            x_representation="final",
            output_root=output_root,
            run_name="mixed-run",
            save_training_data=True,
            density=self.script.DensityEstimatorConfig(model="mnpe", flow_model="nsf"),
            embedding=self.script.EmbeddingConfig(kind="identity"),
            training=self.script.NPETrainingConfig(
                seed=1,
                device="cpu",
                training_batch_size=2,
                max_num_epochs=3,
            ),
            diagnostics=self.script.DiagnosticsConfig(reserve_count=1),
            discrete_theta_parameters=("n_agents", "visibility_distance"),
        )

        raw_results = xr.Dataset(
            data_vars={
                "theta": (
                    ("sample", "theta_parameter"),
                    np.asarray(
                        [
                            [10.0, 0.1, 1.0],
                            [20.0, 0.2, 2.0],
                            [10.0, 0.3, 3.0],
                            [20.0, 0.4, 4.0],
                        ],
                        dtype=np.float32,
                    ),
                ),
                "prior_low": (
                    "theta_parameter",
                    np.asarray([10.0, 0.0, 1.0], dtype=np.float32),
                ),
                "prior_high": (
                    "theta_parameter",
                    np.asarray([20.0, 1.0, 4.0], dtype=np.float32),
                ),
                "x": (
                    ("sample", "step", "observable"),
                    np.asarray(
                        [
                            [[1.0], [2.0]],
                            [[3.0], [4.0]],
                            [[5.0], [6.0]],
                            [[7.0], [8.0]],
                        ],
                        dtype=np.float32,
                    ),
                ),
            },
            coords={
                "sample": np.asarray([0, 1, 2, 3], dtype=np.int64),
                "theta_parameter": (
                    "theta_parameter",
                    ["n_agents", "alpha", "visibility_distance"],
                ),
                "observable": ("observable", ["density.mean"]),
                "step": np.asarray([0, 1], dtype=np.int64),
            },
            attrs={
                "result_store_path": str(config.zarr_path),
                "config_path": str(self.config_dir / "run_config.json"),
                "has_failures": False,
            },
        )

        with (
            patch.object(
                self.script,
                "load_policy_results",
                return_value=raw_results,
            ) as load_policy_results,
            patch.object(
                self.script,
                "train_amortized_npe",
                return_value=self.fake_training_result(),
            ) as train_amortized_npe,
            self.assertLogs(self.script.LOGGER.name, level="INFO") as captured,
        ):
            result = self.script.train_from_config(config)

        logs = "\n".join(captured.output)
        load_policy_results.assert_called_once_with(
            config.zarr_path,
            stats=config.observers,
            normalize_time=config.normalize_time,
        )
        self.assertIn("[training] Starting MNPE training", logs)
        train_indices, diagnostic_indices = self.script.split_diagnostic_holdout(
            n_simulations=4,
            reserve_count=1,
            seed=1,
        )
        expected_reordered_theta = np.asarray(
            [
                [0.1, 10.0, 1.0],
                [0.2, 20.0, 2.0],
                [0.3, 10.0, 3.0],
                [0.4, 20.0, 4.0],
            ],
            dtype=np.float32,
        )
        expected_x = np.asarray([[2.0], [4.0], [6.0], [8.0]], dtype=np.float32)
        np.testing.assert_allclose(
            train_amortized_npe.call_args.kwargs["theta"],
            expected_reordered_theta[train_indices],
        )
        np.testing.assert_allclose(
            train_amortized_npe.call_args.kwargs["x"],
            expected_x[train_indices],
        )
        self.assertEqual(
            train_amortized_npe.call_args.kwargs["num_categories_per_discrete_theta"],
            (2, 4),
        )
        self.assertEqual(result["sbi_method"], "MNPE")
        self.assertEqual(result["normalize_time"], True)
        self.assertEqual(
            result["theta_names"],
            ["alpha", "n_agents", "visibility_distance"],
        )
        self.assertEqual(
            result["discrete_theta_names"],
            ["n_agents", "visibility_distance"],
        )
        self.assertEqual(result["train_count"], 3)
        self.assertEqual(result["diagnostic_count"], 1)

        metadata = json.loads(
            (output_root / "mixed-run" / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["sbi"]["method"], "MNPE")
        self.assertEqual(metadata["sbi"]["prior_samples"]["count"], 3)
        self.assertEqual(metadata["dataset"]["path_base"], "project_root")
        self.assert_project_relative_metadata_path(
            metadata["dataset"]["result_store_path"],
            config.zarr_path,
        )
        self.assert_project_relative_metadata_path(
            metadata["dataset"]["config_path"],
            self.config_dir / "run_config.json",
        )
        with np.load(output_root / "mixed-run" / "diagnostic_data.npz") as diagnostic_data:
            np.testing.assert_allclose(
                diagnostic_data["theta"],
                expected_reordered_theta[diagnostic_indices],
            )
            np.testing.assert_allclose(
                diagnostic_data["x"],
                expected_x[diagnostic_indices],
            )
        with (output_root / "mixed-run" / "posterior.pkl").open("rb") as handle:
            saved_posterior = pickle.load(handle)
        np.testing.assert_allclose(
            saved_posterior["prior_samples"],
            expected_reordered_theta[train_indices],
        )
        self.assertEqual(
            metadata["dataset"]["theta_names"],
            ["alpha", "n_agents", "visibility_distance"],
        )
        self.assertEqual(metadata["dataset"]["continuous_theta_names"], ["alpha"])
        self.assertEqual(
            metadata["dataset"]["discrete_theta_names"],
            ["n_agents", "visibility_distance"],
        )
        self.assertEqual(
            metadata["dataset"]["discrete_theta_values"],
            [[10, 20], [1, 2, 3, 4]],
        )
        self.assertEqual(
            metadata["dataset"]["num_categories_per_discrete_theta"],
            [2, 4],
        )

    def test_train_from_config_appends_agent_count_input_parameter(self) -> None:
        output_root = Path(self.temp_dir.name) / "model_inversion_input"
        config = self.script.TrainNPEConfig(
            zarr_path=(self.config_dir / "../artifacts/run.zarr").resolve(),
            observers=("density.mean",),
            theta_parameters=("alpha", "visibility_distance"),
            input_parameter_names=("n_agents",),
            drop_constant_theta=False,
            x_representation="timeseries",
            output_root=output_root,
            run_name="input-run",
            save_training_data=True,
            density=self.script.DensityEstimatorConfig(model="mnpe", flow_model="nsf"),
            embedding=self.script.EmbeddingConfig(kind="rnn"),
            training=self.script.NPETrainingConfig(
                seed=1,
                device="cpu",
                training_batch_size=2,
                max_num_epochs=3,
            ),
            diagnostics=self.script.DiagnosticsConfig(reserve_count=1),
            discrete_theta_parameters=("visibility_distance",),
        )

        raw_results = xr.Dataset(
            data_vars={
                "theta": (
                    ("sample", "theta_parameter"),
                    np.asarray(
                        [
                            [10.0, 0.1, 1.0],
                            [20.0, 0.2, 2.0],
                            [10.0, 0.3, 3.0],
                            [20.0, 0.4, 4.0],
                        ],
                        dtype=np.float32,
                    ),
                ),
                "prior_low": (
                    "theta_parameter",
                    np.asarray([10.0, 0.0, 1.0], dtype=np.float32),
                ),
                "prior_high": (
                    "theta_parameter",
                    np.asarray([20.0, 1.0, 4.0], dtype=np.float32),
                ),
                "x": (
                    ("sample", "step", "observable"),
                    np.asarray(
                        [
                            [[1.0], [2.0]],
                            [[3.0], [4.0]],
                            [[5.0], [6.0]],
                            [[7.0], [8.0]],
                        ],
                        dtype=np.float32,
                    ),
                ),
            },
            coords={
                "sample": np.asarray([0, 1, 2, 3], dtype=np.int64),
                "theta_parameter": (
                    "theta_parameter",
                    ["n_agents", "alpha", "visibility_distance"],
                ),
                "observable": ("observable", ["density.mean"]),
                "step": np.asarray([0, 1], dtype=np.int64),
            },
            attrs={
                "result_store_path": str(config.zarr_path),
                "config_path": str(self.config_dir / "run_config.json"),
                "has_failures": False,
            },
        )

        with (
            patch.object(
                self.script,
                "load_policy_results",
                return_value=raw_results,
            ),
            patch.object(
                self.script,
                "train_amortized_npe",
                return_value=self.fake_training_result(),
            ) as train_amortized_npe,
        ):
            result = self.script.train_from_config(config)

        train_indices, diagnostic_indices = self.script.split_diagnostic_holdout(
            n_simulations=4,
            reserve_count=1,
            seed=1,
        )
        expected_theta = np.asarray(
            [
                [0.1, 1.0],
                [0.2, 2.0],
                [0.3, 3.0],
                [0.4, 4.0],
            ],
            dtype=np.float32,
        )
        expected_x = np.asarray(
            [
                [[1.0, 10.0], [2.0, 10.0]],
                [[3.0, 20.0], [4.0, 20.0]],
                [[5.0, 10.0], [6.0, 10.0]],
                [[7.0, 20.0], [8.0, 20.0]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            train_amortized_npe.call_args.kwargs["theta"],
            expected_theta[train_indices],
        )
        np.testing.assert_allclose(
            train_amortized_npe.call_args.kwargs["x"],
            expected_x[train_indices],
        )
        self.assertEqual(
            train_amortized_npe.call_args.kwargs["x_event_shape"],
            (2, 2),
        )
        self.assertEqual(
            train_amortized_npe.call_args.kwargs["num_categories_per_discrete_theta"],
            (4,),
        )
        self.assertEqual(result["sbi_method"], "MNPE")
        self.assertEqual(result["theta_names"], ["alpha", "visibility_distance"])
        self.assertEqual(result["discrete_theta_names"], ["visibility_distance"])
        self.assertEqual(result["input_parameter_names"], ["n_agents"])

        metadata = json.loads(
            (output_root / "input-run" / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            metadata["dataset"]["theta_names"],
            ["alpha", "visibility_distance"],
        )
        self.assertEqual(metadata["dataset"]["x_names"], ["density.mean", "n_agents"])
        self.assertEqual(metadata["dataset"]["observer_names"], ["density.mean"])
        self.assertEqual(metadata["dataset"]["input_parameter_names"], ["n_agents"])
        self.assertEqual(metadata["dataset"]["x_event_shape"], [2, 2])
        self.assertEqual(
            metadata["dataset"]["discrete_theta_names"],
            ["visibility_distance"],
        )
        self.assertEqual(metadata["dataset"]["num_categories_per_discrete_theta"], [4])
        with np.load(output_root / "input-run" / "diagnostic_data.npz") as diagnostic_data:
            np.testing.assert_allclose(
                diagnostic_data["theta"],
                expected_theta[diagnostic_indices],
            )
            np.testing.assert_allclose(
                diagnostic_data["x"],
                expected_x[diagnostic_indices],
            )


if __name__ == "__main__":
    unittest.main()
