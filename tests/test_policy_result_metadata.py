from __future__ import annotations

import unittest

import numpy as np

from network_creation.policy_result_metadata import (
    configured_output_step_labels,
    derive_compatible_output_schedule,
    derive_policy_output_schedule,
)


def _policy_config() -> dict[str, object]:
    return {
        "simulator": {"actor_scheduler": {"n_actors": 2}},
        "observers": {
            "defaults": {
                "sampling": {"kind": "periodic", "every": 3},
                "aggregations": ["mean"],
            },
            "outputs": [
                {"name": "density"},
                {
                    "name": "agent_payoff",
                    "sampling": {"kind": "terminal"},
                    "aggregations": ["runs"],
                },
            ],
        },
    }


class PolicyResultMetadataTest(unittest.TestCase):
    def test_periodic_schedule_uses_population_horizons_and_observed_steps(
        self,
    ) -> None:
        schedule = derive_policy_output_schedule(
            "density.mean",
            policy_config=_policy_config(),
            n_agents=[3, 5],
            raw_step_horizons=[4, 7],
        )

        np.testing.assert_array_equal(schedule.valid_sample_count, [2, 3])
        np.testing.assert_array_equal(schedule.valid_raw_step_labels[0], [0, 3])
        np.testing.assert_array_equal(schedule.valid_raw_step_labels[1], [0, 3, 6])
        np.testing.assert_array_equal(schedule.storage_raw_step_labels, [0, 3, 6])
        np.testing.assert_array_equal(schedule.last_valid_raw_step, [3, 6])
        np.testing.assert_allclose(
            schedule.realized_activation_opportunities_per_agent,
            [8.0 / 3.0, 14.0 / 5.0],
        )

    def test_terminal_schedule_retains_population_specific_raw_labels(self) -> None:
        schedule = derive_policy_output_schedule(
            "agent_payoff.runs",
            policy_config=_policy_config(),
            n_agents=[3, 5],
            raw_step_horizons=[4, 7],
        )

        np.testing.assert_array_equal(schedule.valid_sample_count, [1, 1])
        np.testing.assert_array_equal(schedule.valid_raw_step_labels[0], [4])
        np.testing.assert_array_equal(schedule.valid_raw_step_labels[1], [7])
        np.testing.assert_array_equal(schedule.storage_raw_step_labels, [7])
        np.testing.assert_array_equal(schedule.last_valid_raw_step, [4, 7])
        np.testing.assert_allclose(
            schedule.realized_activation_opportunities_per_agent,
            [8.0 / 3.0, 14.0 / 5.0],
        )

    def test_selected_outputs_must_have_compatible_sampling(self) -> None:
        with self.assertRaisesRegex(ValueError, "must share one observation"):
            derive_compatible_output_schedule(
                ["density.mean", "agent_payoff.runs"],
                policy_config=_policy_config(),
                n_agents=[3, 5],
                raw_step_horizons=[4, 7],
            )

    def test_homogeneous_step_label_helper_takes_an_explicit_horizon(self) -> None:
        labels = configured_output_step_labels(_policy_config(), 7)

        np.testing.assert_array_equal(labels["density.mean"], [0, 3, 6])
        np.testing.assert_array_equal(labels["agent_payoff.runs"], [7])


if __name__ == "__main__":
    unittest.main()
