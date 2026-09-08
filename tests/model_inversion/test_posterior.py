from __future__ import annotations

import json
import unittest
from functools import partial
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from tqdm.std import tqdm as terminal_tqdm

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

        payload = metadata().model_dump(mode="json")
        payload["dataset"]["normalize_time"] = True
        with self.assertRaisesRegex(ValueError, "normalize_time"):
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
            {"density.mean": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)},
            {"n_agents": 20},
        )

        np.testing.assert_allclose(x, [3.0, 0.0, 1.0])

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

    def test_case_progress_is_opt_in_and_independent_of_sbi_bars(self) -> None:
        class Posterior:
            def sample(self, sample_shape, x, show_progress_bars):  # noqa: ANN001
                self.show_progress_bars = show_progress_bars
                return torch.as_tensor([1.0, 10.0]).repeat(int(sample_shape[0]), 1)

        posterior = Posterior()
        artifact = PosteriorArtifact(
            Path("."), posterior, metadata(),
            np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int64),
        )
        observations = np.asarray([[1.0, 1.0, 0.0], [2.0, 0.0, 1.0]], dtype=np.float32)
        for options in ({}, {"show_case_progress": True}):
            with self.subTest(options=options):
                output = StringIO()
                with patch(
                    "network_creation.model_inversion.posterior.tqdm",
                    side_effect=partial(terminal_tqdm, file=output),
                ) as progress:
                    samples = artifact.sample_cases(
                        observations, num_samples=5, max_samples_per_call=2,
                        show_progress_bars=False, **options,
                    )
                self.assertEqual(samples.shape, (2, 5, 2))
                self.assertFalse(posterior.show_progress_bars)
                self.assertEqual(progress.call_count, 1)
                self.assertEqual(progress.call_args.kwargs["unit"], "case")
                if options:
                    self.assertFalse(progress.call_args.kwargs["disable"])
                    self.assertIn("Sampling posterior cases", output.getvalue())
                    self.assertIn("2/2", output.getvalue())
                else:
                    self.assertTrue(progress.call_args.kwargs["disable"])
                    self.assertEqual(output.getvalue(), "")

    def test_case_progress_closes_when_sampling_fails(self) -> None:
        artifact = PosteriorArtifact(
            Path("."), object(), metadata(),
            np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int64),
        )
        observations = np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32)
        with (
            patch("network_creation.model_inversion.posterior.tqdm") as progress,
            patch(
                "network_creation.model_inversion.posterior.sample_posterior_observation",
                side_effect=RuntimeError("sampling failed"),
            ),
        ):
            progress.return_value.__enter__.return_value.__iter__.return_value = iter(observations)
            with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                artifact.sample_cases(observations, num_samples=2, show_case_progress=True)
            progress.return_value.__exit__.assert_called_once()
            self.assertIs(progress.return_value.__exit__.call_args.args[0], RuntimeError)


if __name__ == "__main__":
    unittest.main()
