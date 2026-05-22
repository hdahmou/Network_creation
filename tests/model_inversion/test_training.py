from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from network_creation.model_inversion.networks import (
    DensityEstimatorConfig,
    EmbeddingConfig,
)
from network_creation.model_inversion.training import (
    NPETrainingConfig,
    _run_epoch,
    train_amortized_npe,
)


class RecordingDensityBuilder:
    def __init__(self, estimator: "RecordingDensityEstimator | None" = None) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], str, str]] = []
        self.init_theta: torch.Tensor | None = None
        self.init_x: torch.Tensor | None = None
        self.estimator = estimator or RecordingDensityEstimator()

    def __call__(self, theta: Any, x: Any) -> "RecordingDensityEstimator":
        self.init_theta = torch.as_tensor(theta).detach().cpu().clone()
        self.init_x = torch.as_tensor(x).detach().cpu().clone()
        self.calls.append(
            (
                tuple(theta.shape),
                tuple(x.shape),
                str(theta.device),
                str(x.device),
            )
        )
        return self.estimator


class RecordingDensityEstimator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.loaded_state_dict: dict[str, torch.Tensor] | None = None
        self.loss_calls: list[tuple[tuple[int, ...], tuple[int, ...], str, str]] = []

    def loss(self, input: Any, condition: Any) -> torch.Tensor:
        self.loss_calls.append(
            (
                tuple(input.shape),
                tuple(condition.shape),
                str(input.device),
                str(condition.device),
            )
        )
        batch_size = int(input.shape[0])
        return self.weight.expand(batch_size)

    def load_state_dict(self, state_dict: Any, *args: Any, **kwargs: Any) -> Any:
        self.loaded_state_dict = dict(state_dict)
        return super().load_state_dict(state_dict, *args, **kwargs)


class NonFiniteDensityEstimator(RecordingDensityEstimator):
    def __init__(self, loss_values: list[float]) -> None:
        super().__init__()
        self.loss_values = loss_values

    def loss(self, input: Any, condition: Any) -> torch.Tensor:
        self.loss_calls.append(
            (
                tuple(input.shape),
                tuple(condition.shape),
                str(input.device),
                str(condition.device),
            )
        )
        value_index = min(len(self.loss_calls) - 1, len(self.loss_values) - 1)
        batch_size = int(input.shape[0])
        return self.weight.expand(batch_size) + float(self.loss_values[value_index])


class RecordingProgressBar:
    instances: list["RecordingProgressBar"] = []

    def __init__(self, iterable: Any, **kwargs: Any) -> None:
        self.iterable = iterable
        self.kwargs = kwargs
        self.postfixes: list[dict[str, str]] = []
        self.closed = False
        RecordingProgressBar.instances.append(self)

    def __iter__(self) -> Any:
        return iter(self.iterable)

    def set_postfix(
        self,
        values: dict[str, str],
        *,
        refresh: bool = True,
    ) -> None:
        self.postfixes.append(dict(values))

    def close(self) -> None:
        self.closed = True


class RecordingDirectPosterior:
    instances: list["RecordingDirectPosterior"] = []

    def __init__(self, posterior_estimator: object, **kwargs: object) -> None:
        self.posterior_estimator = posterior_estimator
        self.kwargs = kwargs
        RecordingDirectPosterior.instances.append(self)


class TrainAmortizedNPETests(unittest.TestCase):
    def setUp(self) -> None:
        RecordingDirectPosterior.instances.clear()
        RecordingProgressBar.instances.clear()
        progress_bar_patcher = patch("tqdm.auto.tqdm", RecordingProgressBar)
        progress_bar_patcher.start()
        self.addCleanup(progress_bar_patcher.stop)

    def test_identity_embedding_uses_explicit_dataloader_training(self) -> None:
        theta = np.asarray([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)
        x = np.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        density_builder = RecordingDensityBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "network_creation.model_inversion.training."
                    "make_density_estimator_builder",
                    return_value=density_builder,
                ),
                patch(
                    "sbi.inference.posteriors.DirectPosterior",
                    RecordingDirectPosterior,
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
                        seed=4,
                        device="cpu",
                        data_device="cpu",
                        training_batch_size=2,
                        learning_rate=0.0,
                        validation_fraction=0.2,
                        stop_after_epochs=1,
                        max_num_epochs=1,
                    ),
                    density=DensityEstimatorConfig(model="nsf"),
                    embedding=EmbeddingConfig(kind="identity"),
                )

        self.assertEqual(density_builder.calls, [((2, 1), (2, 1), "cpu", "cpu")])
        self.assertIs(
            RecordingDirectPosterior.instances[-1].posterior_estimator,
            density_builder.estimator,
        )
        self.assertIs(result.posterior, RecordingDirectPosterior.instances[-1])
        self.assertTrue(density_builder.estimator.loss_calls)
        self.assertEqual(result.training_summary["epochs_trained"], 1)
        self.assertEqual(result.training_summary["training_loss"], [0.0])
        self.assertEqual(result.training_summary["validation_loss"], [0.0])
        self.assertEqual(result.training_summary["best_validation_loss"], 0.0)
        self.assertEqual(result.training_summary["final_training_loss"], 0.0)
        self.assertEqual(result.training_summary["final_validation_loss"], 0.0)
        self.assertEqual(result.training_summary["best_validation_log_prob"], 0.0)
        self.assertEqual(result.training_summary["final_validation_log_prob"], 0.0)
        self.assertEqual(
            RecordingProgressBar.instances[-1].kwargs["desc"],
            "NPE training",
        )
        self.assertEqual(
            RecordingProgressBar.instances[-1].postfixes,
            [{"train_loss": "0", "val_loss": "0", "best_val": "0"}],
        )
        self.assertTrue(RecordingProgressBar.instances[-1].closed)

    def test_rnn_embedding_uses_explicit_dataloader_training(self) -> None:
        theta = np.asarray(
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]],
            dtype=np.float32,
        )
        x = np.arange(30, dtype=np.float32).reshape(5, 3, 2)
        density_builder = RecordingDensityBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "network_creation.model_inversion.training."
                    "make_density_estimator_builder",
                    return_value=density_builder,
                ),
                patch(
                    "sbi.inference.posteriors.DirectPosterior",
                    RecordingDirectPosterior,
                ),
            ):
                result = train_amortized_npe(
                    theta=theta,
                    x=x,
                    prior_low=np.asarray([0.0, 0.0], dtype=np.float32),
                    prior_high=np.asarray([1.0, 1.0], dtype=np.float32),
                    x_event_shape=(3, 2),
                    tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                    training=NPETrainingConfig(
                        seed=4,
                        device="cpu",
                        data_device="cpu",
                        training_batch_size=2,
                        learning_rate=0.0,
                        validation_fraction=0.2,
                        stop_after_epochs=1,
                        max_num_epochs=1,
                    ),
                    density=DensityEstimatorConfig(model="nsf"),
                    embedding=EmbeddingConfig(kind="rnn"),
                )

        self.assertEqual(density_builder.calls, [((2, 2), (2, 3, 2), "cpu", "cpu")])
        self.assertIs(result.posterior, RecordingDirectPosterior.instances[-1])
        self.assertEqual(result.training_summary["epochs_trained"], 1)
        self.assertEqual(result.training_summary["best_validation_loss"], 0.0)

    def test_discrete_initialization_covers_observed_categories(self) -> None:
        theta = np.asarray(
            [
                [0.1, 0.0],
                [0.2, 1.0],
                [0.3, 2.0],
                [0.4, 0.0],
                [0.5, 1.0],
                [0.6, 2.0],
                [0.7, 0.0],
                [0.8, 1.0],
                [0.9, 2.0],
                [1.0, 0.0],
                [1.1, 1.0],
                [1.2, 2.0],
            ],
            dtype=np.float32,
        )
        x = np.arange(72, dtype=np.float32).reshape(12, 3, 2)
        density_builder = RecordingDensityBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "network_creation.model_inversion.training."
                    "make_density_estimator_builder",
                    return_value=density_builder,
                ),
                patch(
                    "sbi.inference.posteriors.DirectPosterior",
                    RecordingDirectPosterior,
                ),
            ):
                train_amortized_npe(
                    theta=theta,
                    x=x,
                    prior_low=np.asarray([0.0, 0.0], dtype=np.float32),
                    prior_high=np.asarray([1.0, 2.0], dtype=np.float32),
                    x_event_shape=(3, 2),
                    tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                    training=NPETrainingConfig(
                        seed=4,
                        device="cpu",
                        data_device="cpu",
                        training_batch_size=2,
                        learning_rate=0.0,
                        validation_fraction=0.25,
                        stop_after_epochs=1,
                        max_num_epochs=1,
                    ),
                    density=DensityEstimatorConfig(model="mnpe"),
                    embedding=EmbeddingConfig(kind="rnn"),
                    num_categories_per_discrete_theta=(3,),
                )

        self.assertEqual(density_builder.calls, [((3, 2), (3, 3, 2), "cpu", "cpu")])
        assert density_builder.init_theta is not None
        self.assertEqual(set(density_builder.init_theta[:, -1].tolist()), {0.0, 1.0, 2.0})

    def test_estimator_builder_does_not_receive_validation_rows(self) -> None:
        theta = np.asarray([[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)
        x = np.asarray([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=np.float32)
        seed = 7
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        permutation = torch.randperm(theta.shape[0], generator=generator)
        validation_count = 3
        expected_validation_indices = torch.sort(permutation[:validation_count]).values
        expected_train_indices = torch.sort(permutation[validation_count:]).values
        density_builder = RecordingDensityBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "network_creation.model_inversion.training."
                    "make_density_estimator_builder",
                    return_value=density_builder,
                ),
                patch(
                    "sbi.inference.posteriors.DirectPosterior",
                    RecordingDirectPosterior,
                ),
            ):
                train_amortized_npe(
                    theta=theta,
                    x=x,
                    prior_low=np.asarray([0.0], dtype=np.float32),
                    prior_high=np.asarray([1.0], dtype=np.float32),
                    x_event_shape=(1,),
                    tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                    training=NPETrainingConfig(
                        seed=seed,
                        device="cpu",
                        data_device="cpu",
                        training_batch_size=10,
                        learning_rate=0.0,
                        validation_fraction=0.5,
                        stop_after_epochs=1,
                        max_num_epochs=1,
                    ),
                    density=DensityEstimatorConfig(model="nsf"),
                    embedding=EmbeddingConfig(kind="identity"),
                )

        assert density_builder.init_theta is not None
        self.assertEqual(
            set(density_builder.init_theta[:, 0].tolist()),
            set(theta[expected_train_indices.numpy(), 0].tolist()),
        )
        self.assertTrue(
            set(density_builder.init_theta[:, 0].tolist()).isdisjoint(
                set(theta[expected_validation_indices.numpy(), 0].tolist())
            )
        )

    def test_mnpe_validation_split_missing_category_fails(self) -> None:
        theta = np.asarray(
            [
                [0.1, 0.0],
                [0.2, 1.0],
                [0.3, 2.0],
                [0.4, 0.0],
                [0.5, 1.0],
                [0.6, 2.0],
            ],
            dtype=np.float32,
        )
        x = np.ones((6, 1), dtype=np.float32)
        density_builder = RecordingDensityBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "network_creation.model_inversion.training.make_density_estimator_builder",
                return_value=density_builder,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "validation split must contain all discrete theta categories",
                ):
                    train_amortized_npe(
                        theta=theta,
                        x=x,
                        prior_low=np.asarray([0.0, 0.0], dtype=np.float32),
                        prior_high=np.asarray([1.0, 2.0], dtype=np.float32),
                        x_event_shape=(1,),
                        tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                        training=NPETrainingConfig(
                            seed=4,
                            device="cpu",
                            data_device="cpu",
                            validation_fraction=1.0 / 3.0,
                        ),
                        density=DensityEstimatorConfig(model="mnpe"),
                        embedding=EmbeddingConfig(kind="identity"),
                        num_categories_per_discrete_theta=(3,),
                    )

        self.assertEqual(density_builder.calls, [])

    def test_mnpe_training_split_missing_category_fails(self) -> None:
        theta = np.asarray(
            [
                [0.1, 0.0],
                [0.2, 1.0],
                [0.3, 2.0],
                [0.4, 0.0],
                [0.5, 1.0],
                [0.6, 2.0],
            ],
            dtype=np.float32,
        )
        x = np.ones((6, 1), dtype=np.float32)
        density_builder = RecordingDensityBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "network_creation.model_inversion.training.make_density_estimator_builder",
                return_value=density_builder,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "training split must contain all discrete theta categories",
                ):
                    train_amortized_npe(
                        theta=theta,
                        x=x,
                        prior_low=np.asarray([0.0, 0.0], dtype=np.float32),
                        prior_high=np.asarray([1.0, 2.0], dtype=np.float32),
                        x_event_shape=(1,),
                        tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                        training=NPETrainingConfig(
                            seed=4,
                            device="cpu",
                            data_device="cpu",
                            validation_fraction=2.0 / 3.0,
                        ),
                        density=DensityEstimatorConfig(model="mnpe"),
                        embedding=EmbeddingConfig(kind="identity"),
                        num_categories_per_discrete_theta=(3,),
                    )

        self.assertEqual(density_builder.calls, [])

    def test_train_amortized_npe_restores_torch_rng_state(self) -> None:
        theta = np.asarray([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)
        x = np.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        density_builder = RecordingDensityBuilder()

        torch.manual_seed(1234)
        rng_state_before = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "network_creation.model_inversion.training."
                    "make_density_estimator_builder",
                    return_value=density_builder,
                ),
                patch(
                    "sbi.inference.posteriors.DirectPosterior",
                    RecordingDirectPosterior,
                ),
            ):
                train_amortized_npe(
                    theta=theta,
                    x=x,
                    prior_low=np.asarray([0.0], dtype=np.float32),
                    prior_high=np.asarray([1.0], dtype=np.float32),
                    x_event_shape=(1,),
                    tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                    training=NPETrainingConfig(
                        seed=4,
                        device="cpu",
                        data_device="cpu",
                        training_batch_size=2,
                        learning_rate=0.0,
                        validation_fraction=0.2,
                        stop_after_epochs=1,
                        max_num_epochs=1,
                    ),
                    density=DensityEstimatorConfig(model="nsf"),
                    embedding=EmbeddingConfig(kind="identity"),
                )

        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state_before))

    def test_invalid_shape_fails_before_building_estimator(self) -> None:
        builder = RecordingDensityBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "network_creation.model_inversion.training.make_density_estimator_builder",
                return_value=builder,
            ):
                with self.assertRaisesRegex(ValueError, "x_event_shape"):
                    train_amortized_npe(
                        theta=np.ones((3, 1), dtype=np.float32),
                        x=np.ones((3, 2), dtype=np.float32),
                        prior_low=np.asarray([0.0], dtype=np.float32),
                        prior_high=np.asarray([1.0], dtype=np.float32),
                        x_event_shape=(3,),
                        tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                        training=NPETrainingConfig(),
                        density=DensityEstimatorConfig(model="nsf"),
                        embedding=EmbeddingConfig(kind="identity"),
                    )

        self.assertEqual(builder.calls, [])

    def test_invalid_discrete_category_count_fails_in_preparation(self) -> None:
        builder = RecordingDensityBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "network_creation.model_inversion.training.make_density_estimator_builder",
                return_value=builder,
            ):
                with self.assertRaisesRegex(ValueError, "category count"):
                    train_amortized_npe(
                        theta=np.asarray(
                            [[0.1, 0.0], [0.2, 1.0], [0.3, 2.0]],
                            dtype=np.float32,
                        ),
                        x=np.ones((3, 1), dtype=np.float32),
                        prior_low=np.asarray([0.0, 0.0], dtype=np.float32),
                        prior_high=np.asarray([1.0, 2.0], dtype=np.float32),
                        x_event_shape=(1,),
                        tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                        training=NPETrainingConfig(validation_fraction=0.34),
                        density=DensityEstimatorConfig(model="mnpe"),
                        embedding=EmbeddingConfig(kind="identity"),
                        num_categories_per_discrete_theta=(2,),
                    )

        self.assertEqual(builder.calls, [])

    def test_non_integer_discrete_theta_fails_in_preparation(self) -> None:
        builder = RecordingDensityBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "network_creation.model_inversion.training.make_density_estimator_builder",
                return_value=builder,
            ):
                with self.assertRaisesRegex(ValueError, "integer-coded"):
                    train_amortized_npe(
                        theta=np.asarray(
                            [[0.1, 0.0], [0.2, 1.5], [0.3, 2.0]],
                            dtype=np.float32,
                        ),
                        x=np.ones((3, 1), dtype=np.float32),
                        prior_low=np.asarray([0.0, 0.0], dtype=np.float32),
                        prior_high=np.asarray([1.0, 2.0], dtype=np.float32),
                        x_event_shape=(1,),
                        tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                        training=NPETrainingConfig(validation_fraction=0.34),
                        density=DensityEstimatorConfig(model="mnpe"),
                        embedding=EmbeddingConfig(kind="identity"),
                        num_categories_per_discrete_theta=(3,),
                    )

        self.assertEqual(builder.calls, [])

    def test_persistent_workers_requires_worker_processes(self) -> None:
        builder = RecordingDensityBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "network_creation.model_inversion.training.make_density_estimator_builder",
                return_value=builder,
            ):
                with self.assertRaisesRegex(ValueError, "persistent_workers"):
                    train_amortized_npe(
                        theta=np.ones((3, 1), dtype=np.float32),
                        x=np.ones((3, 1), dtype=np.float32),
                        prior_low=np.asarray([0.0], dtype=np.float32),
                        prior_high=np.asarray([1.0], dtype=np.float32),
                        x_event_shape=(1,),
                        tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                        training=NPETrainingConfig(
                            validation_fraction=0.34,
                            persistent_workers=True,
                        ),
                        density=DensityEstimatorConfig(model="nsf"),
                        embedding=EmbeddingConfig(kind="identity"),
                    )

        self.assertEqual(builder.calls, [])

    def test_cuda_data_device_rejects_worker_processes_before_tensor_allocation(self) -> None:
        builder = RecordingDensityBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "network_creation.model_inversion.training.make_density_estimator_builder",
                return_value=builder,
            ):
                with self.assertRaisesRegex(ValueError, "num_workers > 0"):
                    train_amortized_npe(
                        theta=np.ones((3, 1), dtype=np.float32),
                        x=np.ones((3, 1), dtype=np.float32),
                        prior_low=np.asarray([0.0], dtype=np.float32),
                        prior_high=np.asarray([1.0], dtype=np.float32),
                        x_event_shape=(1,),
                        tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                        training=NPETrainingConfig(
                            data_device="cuda",
                            validation_fraction=0.34,
                            num_workers=1,
                        ),
                        density=DensityEstimatorConfig(model="nsf"),
                        embedding=EmbeddingConfig(kind="identity"),
                    )

        self.assertEqual(builder.calls, [])

    def test_cuda_data_device_rejects_explicit_pin_memory_before_tensor_allocation(
        self,
    ) -> None:
        builder = RecordingDensityBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "network_creation.model_inversion.training.make_density_estimator_builder",
                return_value=builder,
            ):
                with self.assertRaisesRegex(ValueError, "pin_memory=True"):
                    train_amortized_npe(
                        theta=np.ones((3, 1), dtype=np.float32),
                        x=np.ones((3, 1), dtype=np.float32),
                        prior_low=np.asarray([0.0], dtype=np.float32),
                        prior_high=np.asarray([1.0], dtype=np.float32),
                        x_event_shape=(1,),
                        tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                        training=NPETrainingConfig(
                            data_device="cuda",
                            validation_fraction=0.34,
                            pin_memory=True,
                        ),
                        density=DensityEstimatorConfig(model="nsf"),
                        embedding=EmbeddingConfig(kind="identity"),
                    )

        self.assertEqual(builder.calls, [])

    def test_zero_sample_epoch_raises_clear_error(self) -> None:
        dataset = torch.utils.data.TensorDataset(
            torch.empty((0, 1)),
            torch.empty((0, 1)),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        estimator = RecordingDensityEstimator()
        optimizer = torch.optim.Adam(estimator.parameters(), lr=0.0)

        with self.assertRaisesRegex(ValueError, "training loader processed zero samples"):
            _run_epoch(
                density_estimator=estimator,
                loader=loader,
                device=torch.device("cpu"),
                optimizer=optimizer,
                clip_max_norm=None,
                loader_name="training",
            )

    def test_best_checkpoint_uses_detached_cpu_clones(self) -> None:
        theta = np.asarray([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
        x = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        density_builder = RecordingDensityBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "network_creation.model_inversion.training."
                    "make_density_estimator_builder",
                    return_value=density_builder,
                ),
                patch(
                    "sbi.inference.posteriors.DirectPosterior",
                    RecordingDirectPosterior,
                ),
            ):
                train_amortized_npe(
                    theta=theta,
                    x=x,
                    prior_low=np.asarray([0.0], dtype=np.float32),
                    prior_high=np.asarray([1.0], dtype=np.float32),
                    x_event_shape=(1,),
                    tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                    training=NPETrainingConfig(
                        seed=4,
                        device="cpu",
                        data_device="cpu",
                        training_batch_size=2,
                        learning_rate=0.0,
                        validation_fraction=0.25,
                        stop_after_epochs=1,
                        max_num_epochs=1,
                    ),
                    density=DensityEstimatorConfig(model="nsf"),
                    embedding=EmbeddingConfig(kind="identity"),
                )

        loaded_state = density_builder.estimator.loaded_state_dict
        assert loaded_state is not None
        loaded_weight = loaded_state["weight"]
        self.assertEqual(str(loaded_weight.device), "cpu")
        self.assertFalse(loaded_weight.requires_grad)
        self.assertNotEqual(
            loaded_weight.data_ptr(),
            density_builder.estimator.weight.detach().data_ptr(),
        )

    def test_non_finite_training_loss_raises(self) -> None:
        density_builder = RecordingDensityBuilder(
            estimator=NonFiniteDensityEstimator([float("nan")])
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "network_creation.model_inversion.training."
                    "make_density_estimator_builder",
                    return_value=density_builder,
                ),
                patch(
                    "sbi.inference.posteriors.DirectPosterior",
                    RecordingDirectPosterior,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "training loss"):
                    train_amortized_npe(
                        theta=np.asarray(
                            [[0.1], [0.2], [0.3], [0.4], [0.5]],
                            dtype=np.float32,
                        ),
                        x=np.asarray(
                            [[1.0], [2.0], [3.0], [4.0], [5.0]],
                            dtype=np.float32,
                        ),
                        prior_low=np.asarray([0.0], dtype=np.float32),
                        prior_high=np.asarray([1.0], dtype=np.float32),
                        x_event_shape=(1,),
                        tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                        training=NPETrainingConfig(
                            training_batch_size=5,
                            validation_fraction=0.2,
                            max_num_epochs=1,
                        ),
                        density=DensityEstimatorConfig(model="nsf"),
                        embedding=EmbeddingConfig(kind="identity"),
                    )

        self.assertFalse(RecordingDirectPosterior.instances)

    def test_non_finite_validation_loss_raises(self) -> None:
        density_builder = RecordingDensityBuilder(
            estimator=NonFiniteDensityEstimator([0.0, float("nan")])
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "network_creation.model_inversion.training."
                    "make_density_estimator_builder",
                    return_value=density_builder,
                ),
                patch(
                    "sbi.inference.posteriors.DirectPosterior",
                    RecordingDirectPosterior,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "validation loss"):
                    train_amortized_npe(
                        theta=np.asarray(
                            [[0.1], [0.2], [0.3], [0.4], [0.5]],
                            dtype=np.float32,
                        ),
                        x=np.asarray(
                            [[1.0], [2.0], [3.0], [4.0], [5.0]],
                            dtype=np.float32,
                        ),
                        prior_low=np.asarray([0.0], dtype=np.float32),
                        prior_high=np.asarray([1.0], dtype=np.float32),
                        x_event_shape=(1,),
                        tensorboard_log_dir=Path(temp_dir) / "sbi_logs",
                        training=NPETrainingConfig(
                            training_batch_size=5,
                            validation_fraction=0.2,
                            max_num_epochs=1,
                        ),
                        density=DensityEstimatorConfig(model="nsf"),
                        embedding=EmbeddingConfig(kind="identity"),
                    )

        self.assertFalse(RecordingDirectPosterior.instances)

if __name__ == "__main__":
    unittest.main()
