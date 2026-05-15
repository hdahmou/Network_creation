from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from network_creation.model_inversion.posterior import PosteriorArtifact
from network_creation.model_inversion.recovery import observation_from_policy_outputs


def artifact_with_dataset(dataset: dict[str, object]) -> PosteriorArtifact:
    return PosteriorArtifact(
        artifact_dir=Path("."),
        posterior=object(),
        metadata={"dataset": dataset},
    )


class RecoveryObservationTests(unittest.TestCase):
    def test_observation_from_policy_outputs_interpolates_normalized_timeseries(
        self,
    ) -> None:
        artifact = artifact_with_dataset(
            {
                "x_names": ["density.mean", "welfare.mean"],
                "x_representation": "timeseries",
                "x_event_shape": [3, 2],
                "normalize_time": True,
                "raw_step_labels": [0.0, 1.0, 2.0, 3.0],
                "step_labels": [0.0, 0.5, 1.0],
            }
        )
        outputs = {
            "density.mean": np.asarray([0.0, 3.0, 6.0, 9.0], dtype=np.float32),
            "welfare.mean": np.asarray([10.0, 13.0, 16.0, 19.0], dtype=np.float32),
        }

        observation = observation_from_policy_outputs(
            outputs,
            artifact,
            n_agents=3,
            n_actors=1,
        )

        np.testing.assert_allclose(
            observation,
            np.asarray(
                [
                    [0.0, 10.0],
                    [4.5, 14.5],
                    [9.0, 19.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_observation_from_policy_outputs_appends_normalized_timeseries_input(
        self,
    ) -> None:
        artifact = artifact_with_dataset(
            {
                "x_names": ["density.mean", "welfare.mean", "n_agents"],
                "observer_names": ["density.mean", "welfare.mean"],
                "input_parameter_names": ["n_agents"],
                "x_representation": "timeseries",
                "x_event_shape": [3, 3],
                "normalize_time": True,
                "raw_step_labels": [0.0, 1.0, 2.0, 3.0],
                "step_labels": [0.0, 0.5, 1.0],
            }
        )
        outputs = {
            "density.mean": np.asarray([0.0, 3.0, 6.0, 9.0], dtype=np.float32),
            "welfare.mean": np.asarray([10.0, 13.0, 16.0, 19.0], dtype=np.float32),
        }

        observation = observation_from_policy_outputs(
            outputs,
            artifact,
            n_agents=3,
            n_actors=1,
        )

        np.testing.assert_allclose(
            observation,
            np.asarray(
                [
                    [0.0, 10.0, 3.0],
                    [4.5, 14.5, 3.0],
                    [9.0, 19.0, 3.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_observation_from_policy_outputs_interpolates_normalized_final(
        self,
    ) -> None:
        artifact = artifact_with_dataset(
            {
                "x_names": ["density.mean", "welfare.mean"],
                "x_representation": "final",
                "x_event_shape": [2],
                "normalize_time": True,
                "raw_step_labels": [0.0, 1.0, 2.0, 3.0],
                "final_step_label": 0.5,
            }
        )
        outputs = {
            "density.mean": np.asarray([0.0, 3.0, 6.0, 9.0], dtype=np.float32),
            "welfare.mean": np.asarray([10.0, 13.0, 16.0, 19.0], dtype=np.float32),
        }

        observation = observation_from_policy_outputs(
            outputs,
            artifact,
            n_agents=3,
            n_actors=1,
        )

        np.testing.assert_allclose(
            observation,
            np.asarray([4.5, 14.5], dtype=np.float32),
        )

    def test_observation_from_policy_outputs_preserves_raw_final_behavior(self) -> None:
        artifact = artifact_with_dataset(
            {
                "x_names": ["density.mean", "welfare.mean"],
                "x_representation": "final",
                "x_event_shape": [2],
            }
        )
        outputs = {
            "density.mean": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "welfare.mean": np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
        }

        observation = observation_from_policy_outputs(
            outputs,
            artifact,
            n_agents=3,
            n_actors=1,
        )

        np.testing.assert_allclose(
            observation,
            np.asarray([3.0, 6.0], dtype=np.float32),
        )

    def test_observation_from_policy_outputs_appends_raw_final_input(self) -> None:
        artifact = artifact_with_dataset(
            {
                "x_names": ["density.mean", "welfare.mean", "n_agents"],
                "observer_names": ["density.mean", "welfare.mean"],
                "input_parameter_names": ["n_agents"],
                "x_representation": "final",
                "x_event_shape": [3],
            }
        )
        outputs = {
            "density.mean": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "welfare.mean": np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
        }

        observation = observation_from_policy_outputs(
            outputs,
            artifact,
            n_agents=3,
            n_actors=1,
        )

        np.testing.assert_allclose(
            observation,
            np.asarray([3.0, 6.0, 3.0], dtype=np.float32),
        )

    def test_observation_from_policy_outputs_preserves_raw_timeseries_behavior(
        self,
    ) -> None:
        artifact = artifact_with_dataset(
            {
                "x_names": ["density.mean", "welfare.mean"],
                "x_representation": "timeseries",
                "x_event_shape": [3, 2],
            }
        )
        outputs = {
            "density.mean": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "welfare.mean": np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
        }

        observation = observation_from_policy_outputs(
            outputs,
            artifact,
            n_agents=3,
            n_actors=1,
        )

        np.testing.assert_allclose(
            observation,
            np.asarray(
                [
                    [1.0, 4.0],
                    [2.0, 5.0],
                    [3.0, 6.0],
                ],
                dtype=np.float32,
            ),
        )


if __name__ == "__main__":
    unittest.main()
