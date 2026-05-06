"""Tests for sigmoid fitting helpers."""

from __future__ import annotations

import unittest
from typing import Any, cast

import numpy as np
from scipy.special import expit

from network_creation.stats.sigmoid_fit import fit_zero_start_sigmoids


class SigmoidFitTest(unittest.TestCase):
    def test_fit_requires_explicit_time_values(self) -> None:
        with self.assertRaises(TypeError):
            cast(Any, fit_zero_start_sigmoids)(np.zeros((1, 3), dtype=np.float32))

    def test_fit_returns_t0_in_supplied_time_units(self) -> None:
        t = np.arange(0, 55, 5, dtype=np.float32)
        expected_t0 = 25.0
        s = expit(0.25 * (t - expected_t0))
        s_start = expit(0.25 * (t[0] - expected_t0))
        y = 0.8 * (s - s_start) / (1.0 - s_start + 1e-8)

        params, good = fit_zero_start_sigmoids(
            y[None, :],
            t=t,
            steps=900,
            min_r2=0.5,
            max_nrmse=0.5,
        )

        self.assertTrue(bool(np.asarray(good)[0]))
        self.assertAlmostEqual(float(np.asarray(params)[0, 2]), expected_t0, delta=5.0)


if __name__ == "__main__":
    unittest.main()
