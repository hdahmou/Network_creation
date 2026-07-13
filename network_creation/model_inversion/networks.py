"""Neural-network choices for SBI posterior estimators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
from pydantic import Field, model_validator
from torch import nn

from .contracts import PositiveInt, StrictModel

EmbeddingKind = Literal["identity", "rnn"]
DensityEstimatorName = Literal["nsf", "maf", "maf_rqs", "mdn", "made"]
RNNKind = Literal["gru", "lstm"]


class EmbeddingConfig(StrictModel):
    kind: EmbeddingKind = "identity"
    output_dim: PositiveInt = 32
    rnn_hidden_size: PositiveInt = 64
    rnn_num_layers: PositiveInt = 1
    rnn_type: RNNKind = "gru"
    rnn_bidirectional: bool = False


ZScoreMode = Literal[
    "independent",
    "structured",
    "transform_to_unconstrained",
    "none",
]


class DensityEstimatorConfig(StrictModel):
    flow_model: DensityEstimatorName = "nsf"
    hidden_features: PositiveInt = 50
    num_transforms: PositiveInt = 5
    num_bins: PositiveInt = 10
    discrete_hidden_features: PositiveInt | None = None
    discrete_hidden_layers: PositiveInt = 2
    continuous_hidden_features: PositiveInt | None = None
    z_score_theta: ZScoreMode = "independent"
    z_score_x: ZScoreMode = "independent"
    sbi_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_managed_sbi_kwargs(self) -> "DensityEstimatorConfig":
        reserved = {
            "model",
            "flow_model",
            "hidden_features",
            "num_transforms",
            "num_bins",
            "z_score_theta",
            "z_score_x",
            "embedding_net",
            "num_categories_per_variable",
            "discrete_hidden_features",
            "discrete_hidden_layers",
            "continuous_hidden_features",
        }
        collisions = sorted(set(self.sbi_kwargs) & reserved)
        if collisions:
            raise ValueError(
                "density_estimator.sbi_kwargs duplicates managed keys: "
                f"{collisions}"
            )
        return self


class RNNEmbedding(nn.Module):
    """Small GRU/LSTM encoder for observer trajectories.

    Input shape is ``(batch, step, feature)``.  ``feature`` is usually the number
    of selected observers at each stored simulation step.
    """

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int = 64,
        output_dim: int = 32,
        num_layers: int = 1,
        rnn_type: RNNKind = "gru",
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        rnn_class: type[nn.GRU] | type[nn.LSTM] = (
            nn.GRU if rnn_type == "gru" else nn.LSTM
        )
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.projection = nn.Linear(hidden_size * directions, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, state = self.rnn(x.float())
        hidden = state[0] if isinstance(state, tuple) else state
        if self.rnn.bidirectional:
            summary = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            summary = hidden[-1]
        return self.projection(summary)


def build_embedding_net(
    config: EmbeddingConfig,
    *,
    x_event_shape: tuple[int, ...],
) -> nn.Module:
    if config.kind == "identity":
        return nn.Identity()

    if config.kind == "rnn":
        if len(x_event_shape) != 2:
            raise ValueError("RNN embedding expects x_event_shape=(steps, features)")
        return RNNEmbedding(
            input_size=x_event_shape[1],
            hidden_size=config.rnn_hidden_size,
            output_dim=config.output_dim,
            num_layers=config.rnn_num_layers,
            rnn_type=config.rnn_type,
            bidirectional=config.rnn_bidirectional,
        )

    raise ValueError(f"unsupported embedding kind: {config.kind}")


def build_density_estimator(
    density: DensityEstimatorConfig,
    embedding: EmbeddingConfig,
    *,
    x_event_shape: tuple[int, ...],
    num_categories_per_discrete_theta: Sequence[int] | None = None,
):
    from sbi.neural_nets import posterior_nn  # type: ignore[import-untyped]

    embedding_net = build_embedding_net(embedding, x_event_shape=x_event_shape)
    category_counts = tuple(
        int(value) for value in (num_categories_per_discrete_theta or ())
    )
    builder_kwargs: dict[str, Any] = {
        "hidden_features": density.hidden_features,
        "num_transforms": density.num_transforms,
        "num_bins": density.num_bins,
        "z_score_theta": density.z_score_theta,
        "z_score_x": density.z_score_x,
    }
    builder_kwargs.update(density.sbi_kwargs)

    if category_counts:
        builder_kwargs = {
            "flow_model": density.flow_model,
            "discrete_hidden_features": density.discrete_hidden_features,
            "discrete_hidden_layers": density.discrete_hidden_layers,
            "continuous_hidden_features": density.continuous_hidden_features,
            **builder_kwargs,
        }
        builder_kwargs["embedding_net"] = embedding_net
        builder_kwargs["num_categories_per_variable"] = torch.as_tensor(
            category_counts,
            dtype=torch.long,
        )
        return posterior_nn(model="mnpe", **builder_kwargs)

    builder_kwargs["embedding_net"] = embedding_net
    return posterior_nn(model=density.flow_model, **builder_kwargs)
