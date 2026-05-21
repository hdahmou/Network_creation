from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from network_creation.model_inversion.networks import (
    DensityEstimatorConfig,
    EmbeddingConfig,
)
from network_creation.model_inversion.training import (
    NPETrainingConfig,
    _training_summary_from_sbi_summary,
    train_amortized_npe,
)


class RecordingNPE:
    instances: list["RecordingNPE"] = []

    def __init__(self, **kwargs: object) -> None:
        self.init_kwargs = kwargs
        self.append_kwargs = None
        self.train_kwargs = None
        self.appended_theta = None
        self.appended_x = None
        self._summary = {
            "epochs_trained": [3],
            "training_loss": [4.0, 3.0, 2.0],
            "validation_loss": [5.0, 2.5, 2.75],
            "best_validation_loss": [2.5],
            "epoch_durations_sec": [0.1, 0.2, 0.3],
        }
        RecordingNPE.instances.append(self)

    def append_simulations(
        self,
        theta: Any,
        x: Any,
        **kwargs: object,
    ) -> "RecordingNPE":
        self.appended_theta = theta
        self.appended_x = x
        self.append_kwargs = kwargs
        return self

    def train(self, **kwargs: object) -> str:
        self.train_kwargs = kwargs
        return "density-estimator"

    def build_posterior(self, density_estimator: object) -> dict[str, object]:
        return {"density_estimator": density_estimator}


class TrainAmortizedNPETests(unittest.TestCase):
    def test_append_simulations_uses_configured_data_device(self) -> None:
        theta = np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32)
        x = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("sbi.inference.NPE", RecordingNPE),
                patch(
                    "network_creation.model_inversion.training.build_density_estimator",
                    return_value="density-builder",
                ),
            ):
                result = train_amortized_npe(
                    theta=theta,
                    x=x,
                    prior_low=np.asarray([0.0], dtype=np.float32),
                    prior_high=np.asarray([1.0], dtype=np.float32),
                    x_event_shape=(1,),
                    tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                    training=NPETrainingConfig(
                        device="cpu",
                        data_device="cpu",
                        training_batch_size=2,
                        max_num_epochs=1,
                        show_progress_bars=False,
                        dataloader_kwargs={"num_workers": 0},
                    ),
                    density=DensityEstimatorConfig(model="nsf"),
                    embedding=EmbeddingConfig(kind="identity"),
                )

        inference = RecordingNPE.instances[-1]
        self.assertEqual(inference.init_kwargs["device"], "cpu")
        self.assertEqual(inference.append_kwargs, {"data_device": "cpu"})
        self.assertEqual(str(inference.appended_theta.device), "cpu")
        self.assertEqual(str(inference.appended_x.device), "cpu")
        self.assertEqual(inference.train_kwargs["dataloader_kwargs"], {"num_workers": 0})
        self.assertEqual(result.posterior, {"density_estimator": "density-estimator"})
        self.assertEqual(result.training_summary["epochs_trained"], 3)
        self.assertEqual(result.training_summary["training_loss"], [4.0, 3.0, 2.0])
        self.assertEqual(result.training_summary["validation_loss"], [5.0, 2.5, 2.75])
        self.assertEqual(result.training_summary["best_validation_loss"], 2.5)
        self.assertEqual(result.training_summary["final_training_loss"], 2.0)
        self.assertEqual(result.training_summary["final_validation_loss"], 2.75)
        self.assertEqual(result.training_summary["best_validation_log_prob"], -2.5)
        self.assertEqual(result.training_summary["final_validation_log_prob"], -2.75)

    def test_training_summary_handles_scalar_sbi_values(self) -> None:
        summary = _training_summary_from_sbi_summary(
            {
                "epochs_trained": 4,
                "training_loss": [3.0, 2.0],
                "validation_loss": [2.5, 2.25],
                "best_validation_loss": 2.25,
            }
        )

        self.assertEqual(summary["epochs_trained"], 4)
        self.assertEqual(summary["training_loss"], [3.0, 2.0])
        self.assertEqual(summary["validation_loss"], [2.5, 2.25])
        self.assertEqual(summary["best_validation_loss"], 2.25)
        self.assertEqual(summary["final_training_loss"], 2.0)
        self.assertEqual(summary["final_validation_loss"], 2.25)
        self.assertEqual(summary["best_validation_log_prob"], -2.25)
        self.assertEqual(summary["final_validation_log_prob"], -2.25)


if __name__ == "__main__":
    unittest.main()
