"""Tests for tie-aware final-hub ranking statistics."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.stats.hub_ranking import (
    simultaneous_cmc_band,
    tie_aware_hit_at_k,
)


class HubRankingTest(unittest.TestCase):
    def test_scores_complete_orderings(self) -> None:
        result = tie_aware_hit_at_k(
            np.asarray([[2, 0, 1], [1, 2, 0]]),
            np.asarray([0, 1]),
        )

        np.testing.assert_allclose(
            result,
            np.asarray([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        )

    def test_averages_over_unobserved_bottom_tie(self) -> None:
        order = np.asarray([[2, -1, -1, -1]] * 4)
        result = tie_aware_hit_at_k(order, np.arange(4))

        np.testing.assert_allclose(result[0], [0.0, 1 / 3, 2 / 3, 1.0])
        np.testing.assert_allclose(
            result.mean(axis=0),
            np.asarray([0.25, 0.5, 0.75, 1.0]),
        )

    def test_builds_simultaneous_band_over_orderings_and_ranks(self) -> None:
        rng = np.random.default_rng(31)
        score_values = rng.integers(0, 2, size=(40, 2, 4)).astype(float)
        score_values[:, :, -1] = 1.0

        band = simultaneous_cmc_band(
            score_values,
            n_resamples=200,
            seed=37,
        )

        self.assertEqual(band.observed.shape, (2, 4))
        self.assertEqual(band.low.shape, (2, 4))
        self.assertEqual(band.high.shape, (2, 4))
        self.assertTrue(np.all(band.low <= band.observed))
        self.assertTrue(np.all(band.observed <= band.high))
        np.testing.assert_allclose(band.low[:, -1], 1.0)
        np.testing.assert_allclose(band.high[:, -1], 1.0)


if __name__ == "__main__":
    unittest.main()
