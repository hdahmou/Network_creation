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
    def test_forwards_sbi_kwargs_for_continuous_npe(self) -> None:
        with patch("sbi.neural_nets.posterior_nn", return_value="builder") as posterior_nn:
            builder = build_density_estimator(
                DensityEstimatorConfig(
                    model="nsf",
                    hidden_features=50,
                    sbi_kwargs={
                        "hidden_features": 77,
                        "dropout_probability": 0.05,
                        "tail_bound": 5.0,
                        "embedding_net": "ignored",
                        "num_categories_per_variable": "ignored",
                    },
                ),
                EmbeddingConfig(kind="identity"),
                x_event_shape=(3,),
            )

        self.assertEqual(builder, "builder")
        kwargs = posterior_nn.call_args.kwargs
        self.assertEqual(posterior_nn.call_args.args, ())
        self.assertEqual(kwargs["model"], "nsf")
        self.assertEqual(kwargs["hidden_features"], 77)
        self.assertEqual(kwargs["dropout_probability"], 0.05)
        self.assertEqual(kwargs["tail_bound"], 5.0)
        self.assertIsInstance(kwargs["embedding_net"], nn.Identity)
        self.assertNotEqual(kwargs["embedding_net"], "ignored")
        self.assertNotIn("num_categories_per_variable", kwargs)

    def test_mnpe_keeps_runtime_controlled_category_counts(self) -> None:
        with patch("sbi.neural_nets.posterior_nn", return_value="builder") as posterior_nn:
            builder = build_density_estimator(
                DensityEstimatorConfig(
                    model="mnpe",
                    flow_model="nsf",
                    sbi_kwargs={
                        "flow_model": "maf",
                        "num_categories_per_variable": torch.as_tensor([99]),
                    },
                ),
                EmbeddingConfig(kind="identity"),
                x_event_shape=(3,),
                num_categories_per_discrete_theta=(2, 4),
            )

        self.assertEqual(builder, "builder")
        kwargs = posterior_nn.call_args.kwargs
        self.assertEqual(kwargs["model"], "mnpe")
        self.assertEqual(kwargs["flow_model"], "maf")
        torch.testing.assert_close(
            kwargs["num_categories_per_variable"],
            torch.as_tensor([2, 4], dtype=torch.long),
        )
        self.assertIsInstance(kwargs["embedding_net"], nn.Identity)


if __name__ == "__main__":
    unittest.main()
