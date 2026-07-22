from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from network_creation.model_inversion.artifact_metadata import ArtifactMetadata
from network_creation.model_inversion.posterior import (
    PosteriorArtifact,
    category_probabilities,
    decode_input_parameter_values,
    sample_posterior_observation,
    validate_discrete_theta_samples,
)
from tests.model_inversion.support import artifact_metadata


def metadata() -> ArtifactMetadata:
    return artifact_metadata()


class PosteriorArtifactTests(unittest.TestCase):
    def test_artifact_metadata_rejects_unknown_and_misaligned_fields(self) -> None:
        payload = metadata().model_dump(mode="json")
        payload["legacy_field"] = True
        with self.assertRaisesRegex(ValueError, "legacy_field"):
            ArtifactMetadata.model_validate_json(json.dumps(payload))

        payload = metadata().model_dump(mode="json")
        payload["dataset"]["discrete_theta_values"] = []
        with self.assertRaisesRegex(ValueError, "discrete_theta_values"):
            ArtifactMetadata.model_validate_json(json.dumps(payload))

    def test_validate_discrete_theta_samples_checks_stored_categories(self) -> None:
        raw = np.asarray([[0.1, 10.0], [0.2, 20.0]], dtype=np.float32)
        validated = validate_discrete_theta_samples(
            raw,
            theta_names=("alpha", "n_agents"),
            discrete_theta_values={"n_agents": (10, 20)},
        )

        np.testing.assert_allclose(validated, [[0.1, 10.0], [0.2, 20.0]])

        with self.assertRaisesRegex(ValueError, "unknown categories"):
            validate_discrete_theta_samples(
                [[0.1, 0.0]],
                theta_names=("alpha", "n_agents"),
                discrete_theta_values={"n_agents": (10, 20)},
            )

    def test_input_values_from_x_decodes_one_hot_inputs(self) -> None:
        x = np.asarray([[0.1, 1.0, 0.0], [0.2, 0.0, 1.0]], dtype=np.float32)
        values = decode_input_parameter_values(x, metadata().dataset)

        np.testing.assert_allclose(values["n_agents"], [10.0, 20.0])

    def test_build_x_appends_one_hot_inputs(self) -> None:
        artifact = PosteriorArtifact(
            artifact_dir=Path("."),
            posterior=object(),
            metadata=metadata(),
            prior_samples=np.empty((0, 2), dtype=np.float32),
            prior_sample_indices=np.empty((0,), dtype=np.int64),
        )

        x = artifact.build_x(
            {"density.mean": np.asarray([1.0, 2.0], dtype=np.float32)},
            {"n_agents": 20},
        )

        np.testing.assert_allclose(x, [2.0, 0.0, 1.0])

    def test_sample_posterior_observation_uses_new_theta_metadata(self) -> None:
        class RecordingPosterior:
            def __init__(self) -> None:
                self.x_shape: tuple[int, ...] | None = None

            def sample(self, sample_shape, x, **kwargs):  # noqa: ANN001, ANN003
                self.x_shape = tuple(x.shape)
                return torch.as_tensor(
                    [[0.25, 10.0], [0.75, 20.0]],
                    dtype=torch.float32,
                )

        posterior = RecordingPosterior()
        samples = sample_posterior_observation(
            posterior,
            [1.0, 2.0, 3.0],
            num_samples=2,
            x_event_shape=(3,),
            theta_names=("alpha", "n_agents"),
            discrete_theta_names=("n_agents",),
            discrete_theta_values=((10, 20),),
        )

        self.assertEqual(posterior.x_shape, (1, 3))
        np.testing.assert_allclose(samples, [[0.25, 10.0], [0.75, 20.0]])

    def test_sample_cases_accepts_case_axis(self) -> None:
        class RecordingPosterior:
            def __init__(self) -> None:
                self.x_shapes: list[tuple[int, ...]] = []

            def sample(self, sample_shape, x, **kwargs):  # noqa: ANN001, ANN003
                self.x_shapes.append(tuple(x.shape))
                return torch.as_tensor([1.0, 10.0]).repeat(int(sample_shape[0]), 1)

        posterior = RecordingPosterior()
        artifact = PosteriorArtifact(
            Path("."),
            posterior,
            metadata(),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

        samples = artifact.sample_cases(
            np.asarray([[1.0, 1.0, 0.0], [2.0, 0.0, 1.0]], dtype=np.float32),
            num_samples=3,
        )

        self.assertEqual(samples.shape, (2, 3, 2))
        self.assertEqual(posterior.x_shapes, [(1, 3), (1, 3)])

    def test_category_probabilities_uses_stored_values(self) -> None:
        np.testing.assert_allclose(
            category_probabilities([10.0, 20.0, 20.0], [10, 20]),
            [1 / 3, 2 / 3],
        )

        with self.assertRaisesRegex(ValueError, "unknown categories"):
            category_probabilities([10.0, 30.0], [10, 20])


if __name__ == "__main__":
    unittest.main()
