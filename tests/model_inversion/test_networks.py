from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from network_creation.model_inversion.networks import (
    DensityEstimatorConfig,
    EmbeddingConfig,
    build_density_estimator,
)


class DensityEstimatorBuilderTests(unittest.TestCase):
    def test_accepts_zuko_flow_models(self) -> None:
        for flow_model in ("zuko_nsf", "zuko_maf"):
            with self.subTest(flow_model=flow_model):
                config = DensityEstimatorConfig.model_validate(
                    {"flow_model": flow_model}
                )
                self.assertEqual(config.flow_model, flow_model)

    def test_forwards_sbi_kwargs_for_continuous_npe(self) -> None:
        with patch("sbi.neural_nets.posterior_nn", return_value="builder") as posterior_nn:
            builder = build_density_estimator(
                DensityEstimatorConfig(
                    flow_model="zuko_nsf",
                    hidden_features=50,
                    sbi_kwargs={
                        "dropout_probability": 0.05,
                        "tail_bound": 5.0,
                    },
                ),
                EmbeddingConfig(kind="identity"),
                x_event_shape=(3,),
            )

        self.assertEqual(builder, "builder")
        kwargs = posterior_nn.call_args.kwargs
        self.assertEqual(kwargs["model"], "zuko_nsf")
        self.assertEqual(kwargs["hidden_features"], 50)
        self.assertEqual(kwargs["dropout_probability"], 0.05)
        self.assertEqual(kwargs["tail_bound"], 5.0)
        self.assertIsInstance(kwargs["embedding_net"], nn.Identity)
        self.assertNotIn("num_categories_per_variable", kwargs)

    def test_mnpe_keeps_runtime_controlled_category_counts(self) -> None:
        with patch("sbi.neural_nets.posterior_nn", return_value="builder") as posterior_nn:
            builder = build_density_estimator(
                DensityEstimatorConfig(
                    flow_model="nsf",
                    sbi_kwargs={"dropout_probability": 0.1},
                ),
                EmbeddingConfig(kind="identity"),
                x_event_shape=(3,),
                num_categories_per_discrete_theta=(2, 4),
            )

        self.assertEqual(builder, "builder")
        kwargs = posterior_nn.call_args.kwargs
        self.assertEqual(kwargs["model"], "mnpe")
        self.assertEqual(kwargs["flow_model"], "nsf")
        torch.testing.assert_close(
            kwargs["num_categories_per_variable"],
            torch.as_tensor([2, 4], dtype=torch.long),
        )
        self.assertIsInstance(kwargs["embedding_net"], nn.Identity)

    def test_rejects_sbi_kwargs_that_duplicate_managed_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates managed keys"):
            DensityEstimatorConfig(
                sbi_kwargs={
                    "hidden_features": 77,
                    "embedding_net": "invalid",
                }
            )


if __name__ == "__main__":
    unittest.main()
