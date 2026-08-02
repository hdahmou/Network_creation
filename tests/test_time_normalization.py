from __future__ import annotations

import unittest

import numpy as np

from network_creation.time_normalization import (
    common_normalized_time_grid,
    common_normalized_time_grid_from_raw_labels,
    interpolate_time_axis,
    normalize_time_tensor,
    normalized_source_times,
)


class TimeNormalizationTests(unittest.TestCase):
    def test_normalized_source_times_scales_by_effective_actors(self) -> None:
        times = normalized_source_times(
            [0, 1, 2],
            n_agents=[2, 4],
            n_actors=1,
        )

        np.testing.assert_allclose(
            times,
            np.asarray(
                [
                    [0.0, 0.5, 1.0],
                    [0.0, 0.25, 0.5],
                ],
                dtype=np.float64,
            ),
        )

    def test_common_grid_uses_common_horizon_and_coarsest_resolution(self) -> None:
        grid = common_normalized_time_grid(
            [0, 1, 2, 3, 4],
            n_agents_values=[2, 4],
            n_actors=1,
        )

        np.testing.assert_allclose(grid.step_labels, [0.0, 0.5, 1.0])
        self.assertEqual(grid.attrs()["time_axis"], "activation_opportunities_per_agent")
        self.assertEqual(grid.attrs()["raw_step_labels"], [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(grid.common_normalized_time_max, 1.0)
        self.assertEqual(grid.normalized_time_resolution, 0.5)

    def test_common_grid_supports_population_specific_valid_raw_labels(self) -> None:
        grid = common_normalized_time_grid_from_raw_labels(
            ([0, 3], [0, 3, 6]),
            n_agents_values=[3, 5],
            n_actors=2,
        )

        np.testing.assert_allclose(grid.step_labels, [0.0, 2.0])
        np.testing.assert_allclose(grid.source_times[0], [0.0, 2.0])
        np.testing.assert_allclose(grid.source_times[1], [0.0, 1.2, 2.4])
        np.testing.assert_allclose(grid.raw_step_labels, [0.0, 3.0, 6.0])
        self.assertEqual(grid.common_normalized_time_max, 2.0)
        self.assertEqual(grid.normalized_time_resolution, 2.0)

    def test_common_grid_rejects_differing_singleton_endpoints(self) -> None:
        with self.assertRaisesRegex(ValueError, "differing normalized endpoints"):
            common_normalized_time_grid_from_raw_labels(
                ([4], [7]),
                n_agents_values=[3, 5],
                n_actors=2,
            )

    def test_interpolate_time_axis_handles_non_leading_step_axis(self) -> None:
        values = np.asarray(
            [
                [0.0, 10.0, 20.0, 30.0],
                [100.0, 110.0, 120.0, 130.0],
            ],
            dtype=np.float32,
        )

        interpolated = interpolate_time_axis(
            values,
            source_times=[0.0, 0.5, 1.0, 1.5],
            target_times=[0.0, 0.75, 1.5],
            step_axis=1,
        )

        np.testing.assert_allclose(
            interpolated,
            np.asarray(
                [
                    [0.0, 15.0, 30.0],
                    [100.0, 115.0, 130.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_normalize_time_tensor_aligns_multiple_agent_slices(self) -> None:
        values = np.asarray(
            [
                [[0.0], [10.0], [20.0], [30.0], [40.0]],
                [[200.0], [201.0], [202.0], [203.0], [204.0]],
            ],
            dtype=np.float32,
        )

        normalized = normalize_time_tensor(
            values,
            raw_step_labels=[0, 1, 2, 3, 4],
            n_agents=[2, 4],
            n_actors=1,
            n_agents_axis=0,
            step_axis=1,
        )

        np.testing.assert_allclose(normalized.grid.step_labels, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(
            normalized.values,
            np.asarray(
                [
                    [[0.0], [10.0], [20.0]],
                    [[200.0], [202.0], [204.0]],
                ],
                dtype=np.float32,
            ),
        )

    def test_normalize_time_tensor_interpolates_single_series_to_target_grid(self) -> None:
        values = np.asarray(
            [
                [0.0, 10.0],
                [3.0, 13.0],
                [6.0, 16.0],
                [9.0, 19.0],
            ],
            dtype=np.float32,
        )

        normalized = normalize_time_tensor(
            values,
            raw_step_labels=[0, 1, 2, 3],
            n_agents=3,
            n_actors=1,
            step_axis=0,
            target_times=[0.0, 0.5, 1.0],
        )

        np.testing.assert_allclose(
            normalized.values,
            np.asarray(
                [
                    [0.0, 10.0],
                    [4.5, 14.5],
                    [9.0, 19.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_interpolation_rejects_target_outside_source_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_times extend"):
            interpolate_time_axis(
                [0.0, 1.0],
                source_times=[0.0, 0.5],
                target_times=[0.0, 1.0],
                step_axis=0,
            )


if __name__ == "__main__":
    unittest.main()
