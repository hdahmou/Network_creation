from __future__ import annotations

import unittest

import numpy as np

from network_creation.core.observability import ObservationSampling
from network_creation.model_inversion.encoding import ObservationEncoder


def encoder(representation: str) -> ObservationEncoder:
    return ObservationEncoder(
        observed_outputs=("series.mean",),
        continuous_input_names=(),
        discrete_input_names=(),
        discrete_input_values={},
        x_components=("series.mean",),
        x_event_shape=(1,) if representation == "final" else (3, 1),
        x_representation=representation,
        n_agents_values=np.asarray([10, 20], dtype=np.int64),
        raw_step_horizons=np.asarray([2, 4], dtype=np.int64),
        n_actors=1,
        output_samplings=(ObservationSampling.periodic(1),),
        step_labels=np.asarray(
            [0.2] if representation == "final" else [0.0, 0.1, 0.2],
            dtype=np.float64,
        ),
    )


def terminal_encoder() -> ObservationEncoder:
    return ObservationEncoder(
        observed_outputs=("terminal.mean",),
        continuous_input_names=(),
        discrete_input_names=(),
        discrete_input_values={},
        x_components=("terminal.mean",),
        x_event_shape=(1,),
        x_representation="final",
        n_agents_values=np.asarray([10, 20], dtype=np.int64),
        raw_step_horizons=np.asarray([10, 20], dtype=np.int64),
        n_actors=1,
        output_samplings=(ObservationSampling.terminal(),),
        step_labels=np.asarray([1.0], dtype=np.float64),
    )


class ObservationEncoderTests(unittest.TestCase):
    def test_padded_and_native_timeseries_encode_identically(self) -> None:
        item = encoder("timeseries")
        native = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
        padded = np.asarray([0.0, 1.0, 2.0, np.nan, np.nan], dtype=np.float32)

        native_x = item.encode({"series.mean": native}, n_agents=10)
        padded_x = item.encode({"series.mean": padded}, n_agents=10)

        np.testing.assert_allclose(native_x, [[0.0], [1.0], [2.0]])
        np.testing.assert_allclose(padded_x, native_x)

    def test_padded_and_native_final_encode_identically(self) -> None:
        item = encoder("final")
        native = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
        padded = np.asarray([0.0, 1.0, 2.0, np.nan, np.nan], dtype=np.float32)

        native_x = item.encode({"series.mean": native}, n_agents=10)
        padded_x = item.encode({"series.mean": padded}, n_agents=10)

        np.testing.assert_allclose(native_x, [2.0])
        np.testing.assert_allclose(padded_x, native_x)

    def test_population_specific_prefix_is_normalized_to_common_grid(self) -> None:
        item = encoder("timeseries")

        x = item.encode(
            {"series.mean": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])},
            n_agents=20,
        )

        np.testing.assert_allclose(x, [[0.0], [2.0], [4.0]])

    def test_rejects_too_short_unknown_population_and_nonfinite_prefix(self) -> None:
        item = encoder("timeseries")

        with self.assertRaisesRegex(ValueError, "either 3 native samples"):
            item.encode({"series.mean": [0.0, 1.0]}, n_agents=10)
        with self.assertRaisesRegex(ValueError, "unknown n_agents"):
            item.encode({"series.mean": [0.0, 1.0, 2.0]}, n_agents=30)
        with self.assertRaisesRegex(ValueError, "structurally valid prefix"):
            item.encode(
                {"series.mean": [0.0, np.nan, 2.0, np.nan, np.nan]},
                n_agents=10,
            )

    def test_periodic_rejects_intermediate_and_overlong_input(self) -> None:
        item = encoder("timeseries")

        for sample_count in (4, 6):
            with self.subTest(sample_count=sample_count):
                with self.assertRaisesRegex(
                    ValueError,
                    "either 3 native samples or 5 fixed-storage samples",
                ):
                    item.encode(
                        {"series.mean": np.arange(sample_count, dtype=np.float32)},
                        n_agents=10,
                    )

    def test_terminal_scalar_uses_its_actual_normalized_endpoint(self) -> None:
        item = terminal_encoder()

        np.testing.assert_allclose(
            item.encode({"terminal.mean": np.asarray(7.0)}, n_agents=10),
            [7.0],
        )
        np.testing.assert_allclose(
            item.encode({"terminal.mean": [9.0]}, n_agents=20),
            [9.0],
        )

    def test_terminal_rejects_overlong_input(self) -> None:
        item = terminal_encoder()

        with self.assertRaisesRegex(ValueError, "exactly 1 sample"):
            item.encode({"terminal.mean": [7.0, 8.0]}, n_agents=10)


if __name__ == "__main__":
    unittest.main()
