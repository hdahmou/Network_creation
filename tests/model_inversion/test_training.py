from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from network_creation.model_inversion.dataset import (
    build_row_index,
    make_zarr_inversion_dataset,
)
from network_creation.model_inversion.schema import resolve_inversion_schema
from network_creation.model_inversion.source import load_inversion_source
from network_creation.model_inversion.splits import split_inversion_rows
from network_creation.model_inversion.training import (
    _run_epoch,
    build_posterior_estimator,
    fit_posterior_estimator,
    make_training_loaders,
)

from .support import write_result_store, write_training_config


class RecordingDensityBuilder:
    def __init__(self, estimator: "RecordingDensityEstimator | None" = None) -> None:
        self.estimator = estimator or RecordingDensityEstimator()
        self.init_theta: torch.Tensor | None = None
        self.init_x: torch.Tensor | None = None

    def __call__(self, theta, x):  # noqa: ANN001
        self.init_theta = torch.as_tensor(theta).detach().cpu().clone()
        self.init_x = torch.as_tensor(x).detach().cpu().clone()
        return self.estimator


class RecordingDensityEstimator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.loss_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.loaded_state_dict: dict[str, torch.Tensor] | None = None

    def loss(self, input, condition):  # noqa: ANN001
        self.loss_calls.append((tuple(input.shape), tuple(condition.shape)))
        return self.weight.expand(int(input.shape[0]))

    def load_state_dict(self, state_dict, *args, **kwargs):  # noqa: ANN001
        self.loaded_state_dict = dict(state_dict)
        return super().load_state_dict(state_dict, *args, **kwargs)


class NonFiniteDensityEstimator(RecordingDensityEstimator):
    def loss(self, input, condition):  # noqa: ANN001
        self.loss_calls.append((tuple(input.shape), tuple(condition.shape)))
        return self.weight.expand(int(input.shape[0])) + float("nan")


class ZarrTrainingTests(unittest.TestCase):
    def prepared(self, temp_dir: str):
        workdir = Path(temp_dir)
        zarr_path = write_result_store(workdir / "run.zarr")
        config_path = write_training_config(workdir / "train.json", zarr_path=zarr_path)
        source = load_inversion_source(config_path)
        schema = resolve_inversion_schema(source)
        row_index = build_row_index(source, schema)
        dataset = make_zarr_inversion_dataset(source, row_index, schema)
        split = split_inversion_rows(row_index, dataset, source.config)
        return source, schema, dataset, split

    def test_estimator_is_initialized_from_train_only_calibration_rows(self) -> None:
        builder = RecordingDensityBuilder()
        with tempfile.TemporaryDirectory() as temp_dir:
            source, schema, dataset, split = self.prepared(temp_dir)
            with patch(
                "network_creation.model_inversion.training.make_density_estimator_builder",
                return_value=builder,
            ):
                estimator = build_posterior_estimator(dataset, split, schema, source.config)

        self.assertIs(estimator, builder.estimator)
        assert builder.init_theta is not None
        assert builder.init_x is not None
        self.assertEqual(tuple(builder.init_x.shape[1:]), schema.x_event_shape)
        self.assertEqual(set(builder.init_theta[:, -1].int().tolist()), {1, 2})
        diagnostic_rows = {
            (tuple(theta.tolist()), tuple(x.reshape(-1).tolist()))
            for index in split.diagnostic_indices
            for theta, x in [dataset[int(index)]]
        }
        calibration_rows = {
            (tuple(theta.tolist()), tuple(x.reshape(-1).tolist()))
            for theta, x in zip(
                builder.init_theta.detach().cpu(),
                builder.init_x.detach().cpu(),
                strict=True,
            )
        }
        self.assertTrue(calibration_rows.isdisjoint(diagnostic_rows))

    def test_estimator_construction_is_isolated_from_ambient_rng(self) -> None:
        factory_draws: list[torch.Tensor] = []

        def seeded_factory(*args, **kwargs):  # noqa: ANN002, ANN003
            del args, kwargs
            draw = torch.rand(4)
            factory_draws.append(draw.clone())
            return RecordingDensityBuilder()

        with tempfile.TemporaryDirectory() as temp_dir:
            source, schema, dataset, split = self.prepared(temp_dir)
            with patch(
                "network_creation.model_inversion.training.make_density_estimator_builder",
                side_effect=seeded_factory,
            ):
                torch.manual_seed(111)
                build_posterior_estimator(dataset, split, schema, source.config)
                torch.manual_seed(999)
                build_posterior_estimator(dataset, split, schema, source.config)

        torch.testing.assert_close(factory_draws[0], factory_draws[1])

    def test_fit_posterior_estimator_uses_lazy_loaders_and_restores_best_weights(self) -> None:
        builder = RecordingDensityBuilder()
        with tempfile.TemporaryDirectory() as temp_dir:
            source, schema, dataset, split = self.prepared(temp_dir)
            with patch(
                "network_creation.model_inversion.training.make_density_estimator_builder",
                return_value=builder,
            ):
                estimator = build_posterior_estimator(dataset, split, schema, source.config)
            loaders = make_training_loaders(dataset, split, source.config)
            result = fit_posterior_estimator(estimator, loaders, source.config)

        self.assertEqual(result.epochs_trained, 1)
        self.assertEqual(result.train_loss, [0.0])
        self.assertEqual(result.validation_loss, [0.0])
        self.assertEqual(result.best_validation_loss, 0.0)
        self.assertTrue(builder.estimator.loss_calls)
        assert builder.estimator.loaded_state_dict is not None
        loaded_weight = builder.estimator.loaded_state_dict["weight"]
        self.assertEqual(str(loaded_weight.device), "cpu")
        self.assertFalse(loaded_weight.requires_grad)

    def test_materialized_loaders_share_cached_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source, _, dataset, split = self.prepared(temp_dir)
            training = source.config.training.model_copy(
                update={"materialize_dataset": True}
            )
            config = source.config.model_copy(update={"training": training})

            loaders = make_training_loaders(dataset, split, config)

            train_dataset = loaders.train.dataset
            validation_dataset = loaders.validation.dataset
            self.assertIsInstance(train_dataset, torch.utils.data.Subset)
            self.assertIsInstance(validation_dataset, torch.utils.data.Subset)
            self.assertIs(train_dataset.dataset, validation_dataset.dataset)
            self.assertIsInstance(train_dataset.dataset, torch.utils.data.TensorDataset)
            self.assertIsNone(loaders.train.batch_size)
            self.assertIsNone(loaders.validation.batch_size)
            self.assertEqual(loaders.train.num_workers, 0)
            self.assertFalse(loaders.train.pin_memory)

            for index in range(len(dataset)):
                expected_theta, expected_x = dataset[index]
                cached_theta, cached_x = train_dataset.dataset[index]
                torch.testing.assert_close(cached_theta, expected_theta)
                torch.testing.assert_close(cached_x, expected_x)

            train_batches = list(loaders.train)
            self.assertEqual(
                sum(int(theta.shape[0]) for theta, _ in train_batches),
                len(train_dataset),
            )
            for theta, x in train_batches:
                self.assertLessEqual(
                    int(theta.shape[0]),
                    source.config.training.batch_size,
                )
                self.assertEqual(
                    tuple(theta.shape[1:]),
                    (len(dataset.schema.theta_order),),
                )
                self.assertEqual(tuple(x.shape[1:]), dataset.schema.x_event_shape)

            repeated = make_training_loaders(dataset, split, config)
            for first, second in zip(train_batches, repeated.train, strict=True):
                torch.testing.assert_close(first[0], second[0])
                torch.testing.assert_close(first[1], second[1])

    def test_non_finite_loss_raises(self) -> None:
        dataset = torch.utils.data.TensorDataset(
            torch.ones((2, 1)),
            torch.ones((2, 1)),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        estimator = NonFiniteDensityEstimator()
        optimizer = torch.optim.Adam(estimator.parameters(), lr=0.0)

        with self.assertRaisesRegex(ValueError, "training loss"):
            _run_epoch(
                density_estimator=estimator,
                loader=loader,
                device=torch.device("cpu"),
                optimizer=optimizer,
                clip_max_norm=None,
                loader_name="training",
            )


if __name__ == "__main__":
    unittest.main()
