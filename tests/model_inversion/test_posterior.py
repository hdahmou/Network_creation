from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from network_creation.model_inversion import load_posterior_artifact


class FakePosterior:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def sample(self, sample_shape: tuple[int, ...], *, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(
            {
                "sample_shape": sample_shape,
                "x_shape": tuple(x.shape),
                "x_values": x.detach().cpu().numpy().copy(),
            }
        )
        num_samples = sample_shape[0]
        return torch.arange(num_samples * 2, dtype=torch.float32).reshape(
            num_samples,
            2,
        )


class PosteriorArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.artifact_dir = Path(self.temp_dir.name)
        self.metadata = {
            "sbi": {
                "posterior_file": "posterior.pkl",
                "tensorboard_log_dir": "sbi_logs",
            },
            "dataset": {
                "x_event_shape": [2, 2],
                "x_names": ["density.mean", "distance.mean"],
                "theta_names": ["n_agents", "alpha"],
            },
        }
        (self.artifact_dir / "metadata.json").write_text(
            json.dumps(self.metadata),
            encoding="utf-8",
        )
        with (self.artifact_dir / "posterior.pkl").open("wb") as handle:
            pickle.dump(FakePosterior(), handle)

    def test_loads_metadata_and_samples_with_dataset_shape(self) -> None:
        artifact = load_posterior_artifact(self.artifact_dir)

        self.assertEqual(artifact.metadata, self.metadata)
        self.assertEqual(artifact.observation_order, ["density.mean", "distance.mean"])
        self.assertEqual(artifact.theta_order, ["n_agents", "alpha"])
        self.assertEqual(
            artifact.tensorboard_log_dir,
            self.artifact_dir / "sbi_logs",
        )

        samples, theta_names = artifact.sample([1.0, 2.0, 3.0, 4.0], num_samples=3)

        np.testing.assert_allclose(
            samples,
            np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        )
        self.assertEqual(theta_names, ["n_agents", "alpha"])
        self.assertEqual(len(artifact.posterior.calls), 1)
        call = artifact.posterior.calls[0]
        self.assertEqual(call["sample_shape"], (3,))
        self.assertEqual(call["x_shape"], (2, 2))
        np.testing.assert_allclose(
            call["x_values"],
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
