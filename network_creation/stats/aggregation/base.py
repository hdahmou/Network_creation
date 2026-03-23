"""Shared interfaces for batch aggregation state."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

AggregationOutput = NDArray[np.float32] | tuple[NDArray[np.float32], ...]
AggregationOutputs = dict[str, AggregationOutput]


class StatisticAccumulator(ABC):
    """Stateful online aggregator for one or more named outputs."""

    output_names: frozenset[str]

    def __init__(self, requested_outputs: frozenset[str]) -> None:
        self.requested_outputs = requested_outputs
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated episode state."""

    @abstractmethod
    def update(
        self,
        values: NDArray[np.float32],
    ) -> None:
        """Fold one episode result into the accumulator state."""

    @abstractmethod
    def export(self) -> AggregationOutputs:
        """Return the outputs currently produced by this accumulator."""
