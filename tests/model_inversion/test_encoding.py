from __future__ import annotations

import unittest

import numpy as np

from network_creation.model_inversion.encoding import ObservationEncoder


class ObservationEncoderTests(unittest.TestCase):
    def test_timeseries_broadcasts_scalar_outputs(self) -> None:
        encoder = ObservationEncoder(
            observed_outputs=("series.mean", "terminal.mean"),
            continuous_input_names=(),
            discrete_input_names=("n_agents",),
            discrete_input_values={"n_agents": (10, 20)},
            x_components=(
                "series.mean",
                "terminal.mean",
                "n_agents=10",
                "n_agents=20",
            ),
            x_event_shape=(3, 4),
            x_representation="timeseries",
            normalize_time=False,
            raw_step_labels=np.asarray([0.0, 1.0, 2.0]),
            step_labels=np.asarray([0.0, 1.0, 2.0]),
            final_step_label=2,
            n_actors=1,
        )

        x = encoder.encode(
            {
                "terminal.mean": np.asarray(7.0, dtype=np.float32),
                "series.mean": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            },
            {"n_agents": 20},
        )

        np.testing.assert_allclose(
            x,
            [
                [1.0, 7.0, 0.0, 1.0],
                [2.0, 7.0, 0.0, 1.0],
                [3.0, 7.0, 0.0, 1.0],
            ],
        )

    def test_rejects_incompatible_non_scalar_output_lengths(self) -> None:
        encoder = ObservationEncoder(
            observed_outputs=("left.mean", "right.mean"),
            continuous_input_names=(),
            discrete_input_names=(),
            discrete_input_values={},
            x_components=("left.mean", "right.mean"),
            x_event_shape=(3, 2),
            x_representation="timeseries",
            normalize_time=False,
            raw_step_labels=np.asarray([0.0, 1.0, 2.0]),
            step_labels=np.asarray([0.0, 1.0, 2.0]),
            final_step_label=2,
            n_actors=1,
        )

        with self.assertRaisesRegex(ValueError, "expected 1 or 3"):
            encoder.encode(
                {
                    "left.mean": [1.0, 2.0],
                    "right.mean": [1.0, 2.0, 3.0],
                }
            )


if __name__ == "__main__":
    unittest.main()
