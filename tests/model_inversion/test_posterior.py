from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from network_creation.model_inversion.posterior import (
    PosteriorArtifact,
    PosteriorWithPriorSamples,
    attach_prior_samples_to_posterior,
    category_probabilities,
    decode_discrete_theta_samples,
    select_prior_samples,
)


class PosteriorArtifactHelpersTest(unittest.TestCase):
    def test_select_prior_samples_keeps_deterministic_training_theta_subset(
        self,
    ) -> None:
        theta = np.arange(20, dtype=np.float32).reshape(10, 2)

        samples_a, indices_a = select_prior_samples(theta, max_samples=4, seed=3)
        samples_b, indices_b = select_prior_samples(theta, max_samples=4, seed=3)

        self.assertEqual(samples_a.shape, (4, 2))
        np.testing.assert_array_equal(indices_a, indices_b)
        np.testing.assert_array_equal(samples_a, samples_b)
        np.testing.assert_array_equal(samples_a, theta[indices_a])

    def test_attach_prior_samples_supports_mock_posterior_dicts(self) -> None:
        posterior: dict[str, object] = {}
        samples = np.asarray([[0.1, 10.0], [0.2, 20.0]], dtype=np.float32)

        returned = attach_prior_samples_to_posterior(
            posterior,
            prior_samples=samples,
            prior_sample_indices=np.asarray([2, 5]),
            theta_names=("alpha", "n_agents"),
            seed=7,
        )

        self.assertIs(returned, posterior)
        np.testing.assert_array_equal(posterior["prior_samples"], samples)
        np.testing.assert_array_equal(posterior["prior_sample_indices"], [2, 5])
        self.assertEqual(
            posterior["prior_sample_metadata"],
            {
                "source": "training_theta_rows",
                "count": 2,
                "theta_order": ["alpha", "n_agents"],
                "selection_seed": 7,
            },
        )

    def test_attach_prior_samples_wraps_posterior_when_attributes_are_disallowed(
        self,
    ) -> None:
        class SlottedPosterior:
            __slots__ = ()

            def sample(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
                return (args, kwargs)

        samples = np.asarray([[0.1, 10.0]], dtype=np.float32)

        wrapped = attach_prior_samples_to_posterior(
            SlottedPosterior(),
            prior_samples=samples,
            prior_sample_indices=None,
            theta_names=("alpha", "n_agents"),
            seed=11,
        )

        self.assertIsInstance(wrapped, PosteriorWithPriorSamples)
        np.testing.assert_array_equal(wrapped.prior_samples, samples)
        self.assertIsNone(wrapped.prior_sample_indices)

    def test_decode_discrete_theta_samples_maps_indices_back_to_categories(
        self,
    ) -> None:
        metadata = {
            "dataset": {
                "theta_names": ["alpha", "n_agents"],
                "discrete_theta_names": ["n_agents"],
                "discrete_theta_values": [[10, 20]],
            }
        }
        raw_samples = np.asarray([[0.1, 0.0], [0.2, 1.0]], dtype=np.float32)

        decoded = decode_discrete_theta_samples(raw_samples, metadata)

        np.testing.assert_allclose(
            decoded,
            np.asarray([[0.1, 10.0], [0.2, 20.0]], dtype=np.float32),
        )

    def test_category_probabilities_uses_stored_category_values(self) -> None:
        probs = category_probabilities([10.0, 20.0, 20.0], [10, 20])
        np.testing.assert_allclose(probs, [1 / 3, 2 / 3])

    def test_posterior_artifact_sample_adds_single_observation_batch_dim(self) -> None:
        class RecordingPosterior:
            def __init__(self) -> None:
                self.x_shape: tuple[int, ...] | None = None

            def sample(self, sample_shape, x, **kwargs):  # noqa: ANN001, ANN003
                self.x_shape = tuple(x.shape)
                return torch.as_tensor([[0.25, 10.0]], dtype=torch.float32)

        posterior = RecordingPosterior()
        artifact = PosteriorArtifact(
            artifact_dir=Path("."),
            posterior=posterior,
            metadata={
                "dataset": {
                    "x_event_shape": [3],
                    "theta_names": ["alpha", "n_agents"],
                    "discrete_theta_names": ["n_agents"],
                    "discrete_theta_values": [[10, 20]],
                }
            },
        )

        samples, theta_names = artifact.sample([1.0, 2.0, 3.0], num_samples=1)

        self.assertEqual(posterior.x_shape, (1, 3))
        self.assertEqual(theta_names, ["alpha", "n_agents"])
        np.testing.assert_allclose(samples, [[0.25, 10.0]])

    def test_posterior_artifact_sample_retries_known_mnpe_broadcast_bug(self) -> None:
        class BroadcastBugPosterior:
            def __init__(self) -> None:
                self.sample_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
                self.sample_batched_calls: list[
                    tuple[tuple[int, ...], tuple[int, ...]]
                ] = []

            def sample(self, sample_shape, x, **kwargs):  # noqa: ANN001, ANN003
                self.sample_calls.append((tuple(sample_shape), tuple(x.shape)))
                raise RuntimeError(
                    "output with shape [512, 1, 50] doesn't match the broadcast shape "
                    "[512, 512, 50]"
                )

            def sample_batched(self, sample_shape, x, **kwargs):  # noqa: ANN001, ANN003
                self.sample_batched_calls.append((tuple(sample_shape), tuple(x.shape)))
                return torch.as_tensor(
                    [[[0.1, 0.0], [0.2, 1.0]]],
                    dtype=torch.float32,
                )

        posterior = BroadcastBugPosterior()
        artifact = PosteriorArtifact(
            artifact_dir=Path("."),
            posterior=posterior,
            metadata={
                "dataset": {
                    "x_event_shape": [4, 1],
                    "theta_names": ["alpha", "n_agents"],
                    "discrete_theta_names": ["n_agents"],
                    "discrete_theta_values": [[10, 20]],
                }
            },
        )

        samples, theta_names = artifact.sample(
            np.zeros((4, 1), dtype=np.float32), num_samples=2
        )

        self.assertEqual(posterior.sample_calls, [((2,), (1, 4, 1))])
        self.assertEqual(posterior.sample_batched_calls, [((1,), (2, 4, 1))])
        self.assertEqual(theta_names, ["alpha", "n_agents"])
        np.testing.assert_allclose(
            samples,
            np.asarray([[0.1, 10.0], [0.2, 20.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
