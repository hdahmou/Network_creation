"""Neural-network choices for SBI posterior estimators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from sbi.neural_nets import posterior_nn
from torch import nn

EmbeddingKind = Literal["identity", "rnn"]
DensityEstimatorName = Literal["nsf", "maf", "maf_rqs", "mdn", "made"]
RNNKind = Literal["gru", "lstm"]


@dataclass(frozen=True)
class EmbeddingConfig:
    kind: EmbeddingKind = "identity"
    output_dim: int = 32
    rnn_hidden_size: int = 64
    rnn_num_layers: int = 1
    rnn_type: RNNKind = "gru"
    rnn_bidirectional: bool = False

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DensityEstimatorConfig:
    model: DensityEstimatorName = "nsf"
    hidden_features: int = 50
    num_transforms: int = 5
    num_bins: int = 10
    z_score_theta: str = "independent"
    z_score_x: str = "independent"

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


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
        rnn_class: type[nn.GRU] | type[nn.LSTM] = nn.GRU if rnn_type == "gru" else nn.LSTM
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
):
    embedding_net = build_embedding_net(embedding, x_event_shape=x_event_shape)
    return posterior_nn(
        model=density.model,
        embedding_net=embedding_net,
        hidden_features=density.hidden_features,
        num_transforms=density.num_transforms,
        num_bins=density.num_bins,
        z_score_theta=density.z_score_theta,
        z_score_x=density.z_score_x,
    )
