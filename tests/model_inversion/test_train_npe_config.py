from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
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

    def test_load_train_config_resolves_paths_relative_to_config_file(self) -> None:
        config_path = self.write_config(
            {
                "dataset": {
                    "zarr_path": "../artifacts/run.zarr",
                    "observers": ["density.mean", "distance.mean"],
                    "theta_parameters": ["n_agents", "exploration"],
                    "drop_constant_theta": False,
                    "x_representation": "timeseries",
                },
                "artifact": {
                    "output_root": "../artifacts/model_inversion",
                    "run_name": "manual-run",
                    "save_training_data": True,
                },
                "density_estimator": {
                    "model": "maf",
                    "hidden_features": 12,
                    "num_transforms": 3,
                    "num_bins": 8,
                    "z_score_theta": "none",
                    "z_score_x": "structured",
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
        self.assertFalse(config.drop_constant_theta)
        self.assertEqual(config.x_representation, "timeseries")
        self.assertEqual(config.run_name, "manual-run")
        self.assertTrue(config.save_training_data)
        self.assertEqual(config.density.model, "maf")
        self.assertEqual(config.density.hidden_features, 12)
        self.assertEqual(config.embedding.kind, "rnn")
        self.assertEqual(config.embedding.rnn_type, "lstm")
        self.assertTrue(config.embedding.rnn_bidirectional)
        self.assertEqual(config.training.seed, 7)
        self.assertIsNone(config.training.clip_max_norm)
        self.assertFalse(config.training.show_progress_bars)

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
            ("n_agents", "outside_optimism", "exploration"),
        )
        self.assertEqual(config.x_representation, "final")
        self.assertEqual(config.density.model, "nsf")
        self.assertEqual(config.embedding.kind, "identity")
        self.assertEqual(payload["embedding"], {"kind": "identity"})
        self.assertNotIn("rnn_hidden_size", payload["embedding"])
        self.assertNotIn("rnn_num_layers", payload["embedding"])
        self.assertNotIn("rnn_type", payload["embedding"])
        self.assertNotIn("rnn_bidirectional", payload["embedding"])
        self.assertEqual(config.training.training_batch_size, 200)

    def test_train_from_config_logs_progress_stages(self) -> None:
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
            density=self.script.DensityEstimatorConfig(model="nsf"),
            embedding=self.script.EmbeddingConfig(kind="identity"),
            training=self.script.NPETrainingConfig(
                seed=1,
                device="cpu",
                training_batch_size=2,
                max_num_epochs=3,
            ),
        )

        raw_results = xr.Dataset(
            data_vars={
                "theta": (
                    ("sample", "theta_parameter"),
                    np.asarray([[0.1], [0.2]], dtype=np.float32),
                ),
                "prior_low": ("theta_parameter", np.asarray([0.0], dtype=np.float32)),
                "prior_high": ("theta_parameter", np.asarray([1.0], dtype=np.float32)),
                "x": (
                    ("sample", "step", "observable"),
                    np.asarray([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32),
                ),
            },
            coords={
                "sample": np.asarray([0, 1], dtype=np.int64),
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
            patch.object(self.script, "load_policy_results", return_value=raw_results),
            patch.object(
                self.script,
                "train_amortized_npe",
                return_value={"posterior": True},
            ) as train_amortized_npe,
            self.assertLogs(self.script.LOGGER.name, level="INFO") as captured,
        ):
            result = self.script.train_from_config(config)

        artifact_dir = output_root / "logging-run"
        tensorboard_log_dir = artifact_dir / "sbi_logs"
        logs = "\n".join(captured.output)
        self.assertIn("[stage 2/10] Loading zarr dataset", logs)
        self.assertIn("[stage 3/10] Filtering inversion dataset", logs)
        self.assertIn("[stage 4/10] Creating artifact directory", logs)
        self.assertIn("[stage 5/10] Starting NPE training", logs)
        self.assertIn("[stage 6/10] Finished NPE training", logs)
        self.assertIn("[stage 7/10] Writing metadata", logs)
        self.assertIn("[stage 8/10] Writing posterior", logs)
        self.assertIn("[stage 9/10] Skipping training data write", logs)
        self.assertIn("[stage 10/10] Training run complete", logs)
        self.assertIn(str(config.zarr_path), logs)
        self.assertIn("theta_shape=(2, 1)", logs)
        self.assertIn("x_shape=(2, 1)", logs)
        self.assertIn("device=cpu", logs)
        self.assertIn("batch_size=2", logs)
        self.assertIn("max_num_epochs=3", logs)
        self.assertIn(f"tensorboard_log_dir={tensorboard_log_dir}", logs)
        train_amortized_npe.assert_called_once()
        self.assertEqual(
            train_amortized_npe.call_args.kwargs["tensorboard_log_dir"],
            tensorboard_log_dir,
        )
        self.assertEqual(result["artifact_dir"], str(artifact_dir))
        self.assertEqual(result["tensorboard_log_dir"], str(tensorboard_log_dir))
        metadata = json.loads(
            (artifact_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["sbi"]["tensorboard_log_dir"], "sbi_logs")


if __name__ == "__main__":
    unittest.main()
