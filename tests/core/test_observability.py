"""Tests for observer sampling policies."""

from __future__ import annotations

import unittest

from network_creation.core.observability import ObservationSampling


class ObservationSamplingTest(unittest.TestCase):
    def test_periodic_sampling_samples_expected_steps(self) -> None:
        sampling = ObservationSampling.periodic(2)

        self.assertEqual(
            [sampling.samples_step(step, max_steps=5) for step in range(5)],
            [False, True, False, True, False],
        )

    def test_terminal_sampling_only_samples_last_step(self) -> None:
        sampling = ObservationSampling.terminal()

        self.assertEqual(
            [sampling.samples_step(step, max_steps=4) for step in range(4)],
            [False, False, False, True],
        )

    def test_fixed_sample_count_matches_public_series_shapes(self) -> None:
        self.assertEqual(ObservationSampling.periodic(3).fixed_sample_count(8), 3)
        self.assertEqual(ObservationSampling.terminal().fixed_sample_count(8), 1)
        self.assertEqual(ObservationSampling.terminal().fixed_sample_count(0), 1)


if __name__ == "__main__":
    unittest.main()
