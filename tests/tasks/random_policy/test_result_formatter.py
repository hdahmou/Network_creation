"""Tests for random policy result formatting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.policies import load_policy_results
from tasks.policies.outputs import SAMPLED_AGENT_PARAMETERS_FIELD

from tests.tasks.config_support import continuous
from tests.tasks.policy_result_support import build_random_config, write_result_store


class RandomPolicyResultFormatterTest(unittest.TestCase):
    def test_from_zarr_reconstructs_timeseries_outputs(self) -> None:
        config = build_random_config()
        config["simulator"]["max_steps"] = 2
        outputs_by_index = [
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray([10.0, 11.0, 12.0], dtype=np.float32),
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=1,
            )

            ds = load_policy_results(store_path)

        self.assertNotIn("final", ds)
        self.assertNotIn("parameters", ds)
        self.assertNotIn("timeseries", ds)
        self.assertEqual(ds["theta"].shape, (1, 1, 1, 4))
        self.assertEqual(ds["x"].dims, ("n_agents", "random_proportion", "parameter_row", "step", "observable"))
        np.testing.assert_array_equal(ds.coords["n_agents"].values, np.asarray([4]))
        np.testing.assert_allclose(ds.coords["random_proportion"].values, [1.0])
        np.testing.assert_array_equal(ds.coords["parameter_row"].values, np.asarray([0]))
        np.testing.assert_array_equal(ds.coords["step"].values, np.asarray([0, 1, 2]))
        np.testing.assert_array_equal(ds.coords["input_index"].values, np.asarray([[[0]]], dtype=np.int64))
        self.assertFalse(ds.attrs["has_failures"])
        self.assertEqual(
            tuple(ds.coords["theta_parameter"].values),
            (
                "n_agents",
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
            ),
        )
        np.testing.assert_allclose(ds["prior_low"].values, [4.0, 0.5, 0.5, 4.0])
        np.testing.assert_allclose(ds["prior_high"].values, [4.0, 0.5, 0.5, 4.0])
        self.assertEqual(ds["x"].shape, (1, 1, 1, 3, 2))
        np.testing.assert_allclose(
            ds["x"].isel(step=-1).values,
            np.asarray([[[[2.0, 12.0]]]], dtype=np.float32),
        )

    def test_from_zarr_loads_only_requested_stats(self) -> None:
        config = build_random_config()
        config["simulator"]["max_steps"] = 2
        outputs_by_index = [
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray(
                        [10.0, 11.0, 12.0],
                        dtype=np.float32,
                    ),
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=1,
            )

            ds = load_policy_results(store_path, stats=("density.mean",))

        self.assertEqual(tuple(ds.coords["observable"].values), ("density.mean",))
        self.assertEqual(ds["x"].shape, (1, 1, 1, 3, 1))
        np.testing.assert_allclose(
            ds["x"].isel(step=-1).values,
            np.asarray([[[[2.0]]]], dtype=np.float32),
        )

    def test_from_zarr_rejects_prefix_stats(self) -> None:
        config = build_random_config()
        outputs_by_index = [
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 1.0], dtype=np.float32),
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=1,
            )

            with self.assertRaisesRegex(
                ValueError,
                "unknown output.*density.*density.mean",
            ):
                load_policy_results(store_path, stats="density")

    def test_from_zarr_supports_empty_requested_stats(self) -> None:
        config = build_random_config()
        outputs_by_index = [
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 1.0], dtype=np.float32),
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=1,
            )

            ds = load_policy_results(store_path, stats=())

        self.assertEqual(tuple(ds.coords["observable"].values), ())
        self.assertEqual(ds["x"].shape, (1, 1, 1, 1, 0))
        self.assertEqual(ds["theta"].shape, (1, 1, 1, 4))

    def test_from_zarr_preserves_agent_payoff_axis(self) -> None:
        config = build_random_config()
        config["population"]["n_agents"] = [2, 4]
        config["simulator"]["max_steps"] = 2
        config["observers"]["outputs"] = [
            {"name": "density"},
            {"name": "agent_payoff"},
        ]
        outputs_by_index = [
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                    "agent_payoff.mean": np.asarray(
                        [
                            [0.0, 0.0, np.nan, np.nan],
                            [0.0, 1.0, np.nan, np.nan],
                            [0.5, 1.0, np.nan, np.nan],
                        ],
                        dtype=np.float32,
                    ),
                },
            ),
            (
                1,
                {
                    "density.mean": np.asarray([10.0, 11.0, 12.0], dtype=np.float32),
                    "agent_payoff.mean": np.asarray(
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                        ],
                        dtype=np.float32,
                    ),
                },
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=2,
            )

            ds = load_policy_results(store_path)

        self.assertEqual(
            ds["agent_payoff.mean"].dims,
            (
                "n_agents",
                "random_proportion",
                "parameter_row",
                "step",
                "agent_slot",
            ),
        )
        np.testing.assert_array_equal(ds.coords["agent_slot"].values, np.arange(4))
        np.testing.assert_array_equal(ds.coords["step"].values, np.asarray([0, 1, 2]))
        self.assertEqual(ds["agent_payoff.mean"].shape, (2, 1, 1, 3, 4))
        self.assertEqual(tuple(ds.coords["observable"].values), ("density.mean",))
        self.assertTrue(
            np.isnan(ds["agent_payoff.mean"].isel(n_agents=0).values[..., 2:]).all()
        )
        np.testing.assert_allclose(
            ds["agent_payoff.mean"].isel(n_agents=1, step=-1).values,
            np.asarray([[[5.0, 6.0, 7.0, 8.0]]], dtype=np.float32),
        )

    def test_from_zarr_loads_sampled_agent_parameters_only_when_requested(self) -> None:
        config = build_random_config()
        config["population"]["policies"][0]["parameters"]["p_act"] = continuous(
            0.0,
            1.0,
            std=0.1,
        )
        outputs_by_index = [
            (
                0,
                {
                    "density.mean": np.asarray(
                        [0.0, 1.0, 2.0, 3.0],
                        dtype=np.float32,
                    ),
                    SAMPLED_AGENT_PARAMETERS_FIELD: np.asarray(
                        [[0.1], [0.2], [0.3], [0.4]],
                        dtype=np.float32,
                    ),
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=1,
            )

            default_ds = load_policy_results(store_path)
            ds = load_policy_results(
                store_path,
                load_sampled_agent_parameters=True,
            )

        self.assertNotIn(SAMPLED_AGENT_PARAMETERS_FIELD, default_ds)
        self.assertNotIn("agent_index", default_ds.coords)
        self.assertNotIn("agent_parameter", default_ds.coords)
        self.assertEqual(
            ds[SAMPLED_AGENT_PARAMETERS_FIELD].dims,
            (
                "n_agents",
                "random_proportion",
                "parameter_row",
                "agent_index",
                "agent_parameter",
            ),
        )
        self.assertEqual(tuple(ds.coords["agent_parameter"].values), ("p_act",))
        self.assertEqual(tuple(ds.coords["observable"].values), ("density.mean",))
        np.testing.assert_allclose(
            ds[SAMPLED_AGENT_PARAMETERS_FIELD].values,
            np.asarray([[[[[0.1], [0.2], [0.3], [0.4]]]]], dtype=np.float32),
        )

    def test_from_zarr_loads_sampled_agent_parameters_with_requested_stats(self) -> None:
        config = build_random_config()
        config["population"]["policies"][0]["parameters"]["p_act"] = continuous(
            0.0,
            1.0,
            std=0.1,
        )
        outputs_by_index = [
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray(
                        [10.0, 11.0, 12.0, 13.0],
                        dtype=np.float32,
                    ),
                    SAMPLED_AGENT_PARAMETERS_FIELD: np.asarray(
                        [[0.1], [0.2], [0.3], [0.4]],
                        dtype=np.float32,
                    ),
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=1,
            )

            ds = load_policy_results(
                store_path,
                stats=("density.mean",),
                load_sampled_agent_parameters=True,
            )

        self.assertEqual(tuple(ds.coords["observable"].values), ("density.mean",))
        self.assertIn(SAMPLED_AGENT_PARAMETERS_FIELD, ds)
        np.testing.assert_allclose(
            ds[SAMPLED_AGENT_PARAMETERS_FIELD].values,
            np.asarray([[[[[0.1], [0.2], [0.3], [0.4]]]]], dtype=np.float32),
        )

    def test_from_zarr_uses_observer_sampling_for_step_coordinates(self) -> None:
        config = build_random_config()
        config["simulator"]["max_steps"] = 10
        config["observers"]["defaults"]["sampling"]["every"] = 5
        outputs_by_index = [
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 5.0, 10.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray(
                        [20.0, 25.0, 30.0], dtype=np.float32
                    ),
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=1,
            )

            ds = load_policy_results(store_path)

        np.testing.assert_array_equal(ds.coords["step"].values, np.asarray([0, 5, 10]))

    def test_from_zarr_normalize_time_false_matches_default(self) -> None:
        config = build_random_config()
        config["population"]["n_agents"] = [2, 4]
        config["simulator"]["max_steps"] = 2
        config["simulator"]["actor_scheduler"]["n_actors"] = 1
        outputs_by_index = [
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray(
                        [10.0, 11.0, 12.0], dtype=np.float32
                    ),
                },
            ),
            (
                1,
                {
                    "density.mean": np.asarray([20.0, 21.0, 22.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray(
                        [30.0, 31.0, 32.0], dtype=np.float32
                    ),
                },
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=2,
            )

            default_ds = load_policy_results(store_path)
            raw_ds = load_policy_results(store_path, normalize_time=False)

        self.assertTrue(default_ds.identical(raw_ds))

    def test_from_zarr_normalizes_time_to_common_range_and_interpolates(self) -> None:
        config = build_random_config()
        config["population"]["n_agents"] = [2, 4]
        config["simulator"]["max_steps"] = 4
        config["simulator"]["actor_scheduler"]["n_actors"] = 1
        outputs_by_index = [
            (
                0,
                {
                    "density.mean": np.asarray(
                        [0.0, 10.0, 20.0, 30.0, 40.0],
                        dtype=np.float32,
                    ),
                    "centered_mean_welfare.mean": np.asarray(
                        [100.0, 110.0, 120.0, 130.0, 140.0],
                        dtype=np.float32,
                    ),
                },
            ),
            (
                1,
                {
                    "density.mean": np.asarray(
                        [200.0, 201.0, 202.0, 203.0, 204.0],
                        dtype=np.float32,
                    ),
                    "centered_mean_welfare.mean": np.asarray(
                        [300.0, 301.0, 302.0, 303.0, 304.0],
                        dtype=np.float32,
                    ),
                },
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=2,
            )

            ds = load_policy_results(store_path, normalize_time=True)

        np.testing.assert_allclose(
            ds.coords["step"].values,
            np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        )
        self.assertEqual(ds.attrs["normalize_time"], True)
        self.assertEqual(ds.attrs["time_axis"], "activation_opportunities_per_agent")
        self.assertEqual(ds.attrs["raw_step_labels"], [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(ds.attrs["common_normalized_time_max"], 1.0)
        self.assertEqual(ds.attrs["normalized_time_resolution"], 0.5)
        np.testing.assert_allclose(
            ds["x"]
            .sel(n_agents=2, observable="density.mean")
            .isel(random_proportion=0, parameter_row=0)
            .values,
            np.asarray([0.0, 10.0, 20.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            ds["x"]
            .sel(n_agents=4, observable="density.mean")
            .isel(random_proportion=0, parameter_row=0)
            .values,
            np.asarray([200.0, 202.0, 204.0], dtype=np.float32),
        )

    def test_from_zarr_normalizes_agent_payoff_time_axis(self) -> None:
        config = build_random_config()
        config["population"]["n_agents"] = [2, 4]
        config["simulator"]["max_steps"] = 2
        config["simulator"]["actor_scheduler"]["n_actors"] = 1
        config["observers"]["outputs"] = [
            {"name": "density"},
            {"name": "agent_payoff"},
        ]
        outputs_by_index = [
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                    "agent_payoff.mean": np.asarray(
                        [
                            [0.0, 0.0, np.nan, np.nan],
                            [0.0, 1.0, np.nan, np.nan],
                            [0.5, 1.0, np.nan, np.nan],
                        ],
                        dtype=np.float32,
                    ),
                },
            ),
            (
                1,
                {
                    "density.mean": np.asarray([10.0, 11.0, 12.0], dtype=np.float32),
                    "agent_payoff.mean": np.asarray(
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                        ],
                        dtype=np.float32,
                    ),
                },
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=2,
            )

            ds = load_policy_results(store_path, normalize_time=True)

        np.testing.assert_allclose(ds.coords["step"].values, [0.0, 0.5])
        self.assertEqual(ds["agent_payoff.mean"].shape, (2, 1, 1, 2, 4))
        np.testing.assert_allclose(
            ds["agent_payoff.mean"]
            .sel(n_agents=2)
            .isel(random_proportion=0, parameter_row=0)
            .values,
            np.asarray(
                [
                    [0.0, 0.0, np.nan, np.nan],
                    [0.0, 1.0, np.nan, np.nan],
                ],
                dtype=np.float32,
            ),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            ds["agent_payoff.mean"]
            .sel(n_agents=4)
            .isel(random_proportion=0, parameter_row=0)
            .values,
            np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [5.0, 6.0, 7.0, 8.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_from_zarr_supports_scalar_outputs_without_timeseries(self) -> None:
        config = build_random_config()
        outputs_by_index = [
            (
                0,
                {
                    "density.final": np.asarray(0.25, dtype=np.float32),
                    "centered_mean_welfare.final": np.asarray(12.5, dtype=np.float32),
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=1,
            )

            ds = load_policy_results(store_path)

        self.assertNotIn("final", ds)
        self.assertEqual(ds["x"].shape, (1, 1, 1, 1, 2))
        self.assertEqual(
            tuple(ds.coords["theta_parameter"].values),
            (
                "n_agents",
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
            ),
        )
        np.testing.assert_allclose(
            ds["x"].values,
            np.asarray([[[[[0.25, 12.5]]]]], dtype=np.float32),
        )
        self.assertEqual(
            tuple(ds.coords["observable"].values),
            ("density.final", "centered_mean_welfare.final"),
        )
