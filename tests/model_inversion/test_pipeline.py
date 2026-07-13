from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from network_creation.model_inversion import (
    load_posterior_artifact,
    train_inversion_from_config,
)

from .support import write_result_store, write_training_config


class InversionPipelineTests(unittest.TestCase):
    def test_mnpe_train_save_load_and_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            zarr_path = write_result_store(workdir / "run.zarr")
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=zarr_path,
                output_root=workdir / "artifacts",
                artifact_name="mnpe-smoke",
            )

            summary = train_inversion_from_config(config_path)
            artifact = load_posterior_artifact(summary.artifact_dir)
            x = artifact.build_x(
                {"density.mean": [0.0, 1.0, 2.0]},
                {"n_agents": 10},
            )
            samples, theta_names = artifact.sample(x, num_samples=3, seed=7)

            self.assertEqual(summary.sbi_method, "MNPE")
            self.assertEqual(samples.shape, (3, 3))
            self.assertEqual(
                theta_names,
                ["outside_optimism", "exploration", "visibility_distance"],
            )
            self.assertTrue(np.isin(samples[:, -1], [1.0, 2.0]).all())
            self.assertEqual(artifact.prior_samples.shape, (3, 3))

            with self.assertRaises(FileExistsError):
                train_inversion_from_config(config_path)

    def test_npe_training_export_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            zarr_path = write_result_store(workdir / "run.zarr")
            config_path = write_training_config(
                workdir / "train.json",
                zarr_path=zarr_path,
                output_root=workdir / "artifacts",
                artifact_name="npe-smoke",
                discrete_theta=False,
            )
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            payload["output"]["training_data_sample_count"] = 2
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            summary = train_inversion_from_config(config_path)
            artifact = load_posterior_artifact(summary.artifact_dir)

            self.assertEqual(summary.sbi_method, "NPE")
            self.assertEqual(artifact.theta_order, ["outside_optimism", "exploration"])
            with np.load(Path(summary.artifact_dir) / "training_data.npz") as data:
                self.assertEqual(data["theta"].shape[0], 2)
                self.assertEqual(data["x"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
