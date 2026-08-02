from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from network_creation.model_inversion.posterior import PosteriorArtifact
from network_creation.model_inversion.recovery import observation_from_policy_outputs
from tests.model_inversion.support import artifact_metadata


class RecoveryBuildXTests(unittest.TestCase):
    def test_observation_from_policy_outputs_delegates_to_artifact_build_x(self) -> None:
        artifact = PosteriorArtifact(
            artifact_dir=Path("."),
            posterior=object(),
            metadata=artifact_metadata(
                dataset_overrides={
                    "discrete_theta_names": [],
                    "discrete_theta_values": [],
                }
            ),
            prior_samples=np.empty((0, 1), dtype=np.float32),
            prior_sample_indices=np.empty((0,), dtype=np.int64),
        )

        x = observation_from_policy_outputs(
            {"density.mean": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)},
            artifact,
            n_agents=20,
        )

        np.testing.assert_allclose(x, [3.0, 0.0, 1.0])

    def test_build_x_interpolates_normalized_timeseries(self) -> None:
        artifact = PosteriorArtifact(
            artifact_dir=Path("."),
            posterior=object(),
            metadata=artifact_metadata(
                dataset_overrides={
                    "x_event_shape": [2, 3],
                    "x_representation": "timeseries",
                    "step_labels": [0.0, 0.1],
                    "discrete_theta_names": [],
                    "discrete_theta_values": [],
                }
            ),
            prior_samples=np.empty((0, 1), dtype=np.float32),
            prior_sample_indices=np.empty((0,), dtype=np.int64),
        )

        x = artifact.build_x(
            {"density.mean": np.asarray([0.0, 1.0, np.nan], dtype=np.float32)},
            {"n_agents": 10},
        )

        np.testing.assert_allclose(
            x,
            np.asarray([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
