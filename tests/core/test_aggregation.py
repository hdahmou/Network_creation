"""Tests for statistic aggregation helpers."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.stats.aggregation.moments import MomentsAccumulator
from network_creation.stats.aggregation.registry import build_accumulators
from network_creation.stats.aggregation.runs import RunsCollector


class AggregationTest(unittest.TestCase):
    def test_build_accumulators_rejects_unknown_outputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported aggregation outputs"):
            build_accumulators(("median",))

    def test_moments_accumulator_export_requires_observations(self) -> None:
        accumulator = MomentsAccumulator(frozenset({"mean", "std"}))

        with self.assertRaisesRegex(ValueError, "no observations have been aggregated"):
            accumulator.export()

    def test_moments_accumulator_rejects_shape_mismatch(self) -> None:
        accumulator = MomentsAccumulator(frozenset({"mean"}))
        accumulator.update(np.asarray([1.0, 2.0], dtype=np.float32))

        with self.assertRaisesRegex(
            ValueError,
            "all aggregated episode values must share the same shape",
        ):
            accumulator.update(np.asarray([[1.0, 2.0]], dtype=np.float32))

    def test_moments_accumulator_returns_zero_std_for_single_sample(self) -> None:
        accumulator = MomentsAccumulator(frozenset({"mean", "std"}))
        sample = np.asarray([1.0, 3.0], dtype=np.float32)

        accumulator.update(sample)

        outputs = accumulator.export()
        np.testing.assert_allclose(outputs["mean"], sample)
        np.testing.assert_allclose(
            outputs["std"],
            np.asarray([0.0, 0.0], dtype=np.float32),
        )

    def test_runs_collector_keeps_copies_of_input_arrays(self) -> None:
        collector = RunsCollector(frozenset({"runs"}))
        values = np.asarray([1.0, 2.0], dtype=np.float32)

        collector.update(values)
        values[0] = 99.0

        runs = collector.export()["runs"]
        self.assertEqual(len(runs), 1)
        np.testing.assert_allclose(
            runs[0],
            np.asarray([1.0, 2.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
