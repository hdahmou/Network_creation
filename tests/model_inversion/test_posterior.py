from __future__ import annotations

import unittest

import numpy as np

from network_creation.model_inversion.posterior import (
    PosteriorWithPriorSamples,
    attach_prior_samples_to_posterior,
    category_probabilities,
    decode_discrete_theta_samples,
    select_prior_samples,
)


class PosteriorArtifactHelpersTest(unittest.TestCase):
    def test_select_prior_samples_keeps_deterministic_training_theta_subset(self) -> None:
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


    def test_attach_prior_samples_wraps_posterior_when_attributes_are_disallowed(self) -> None:
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

    def test_decode_discrete_theta_samples_maps_indices_back_to_categories(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
