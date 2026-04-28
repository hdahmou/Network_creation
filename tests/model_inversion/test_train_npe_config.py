from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


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

        config = self.script.load_train_config(baseline_path)

        self.assertEqual(
            config.zarr_path,
            (
                PROJECT_ROOT
                / "artifacts"
                / "policy-run-20260424T205304.zarr"
            ).resolve(),
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
        self.assertEqual(config.density.model, "nsf")
        self.assertEqual(config.embedding.kind, "identity")
        self.assertEqual(config.training.training_batch_size, 200)


if __name__ == "__main__":
    unittest.main()
