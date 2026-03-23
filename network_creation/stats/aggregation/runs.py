"""Accumulator for keeping one value series per episode."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import AggregationOutputs, StatisticAccumulator


class RunsCollector(StatisticAccumulator):
    """Store one copy of each episode result."""

    output_names = frozenset({"runs"})

    def reset(self) -> None:
        self._runs: list[NDArray[np.float32]] = []

    def update(
        self,
        values: NDArray[np.float32],
    ) -> None:
        self._runs.append(np.asarray(values, dtype=np.float32).copy())

    def export(self) -> AggregationOutputs:
        return {"runs": tuple(self._runs)}
