"""Accumulator for mean and standard deviation outputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import AggregationOutputs, StatisticAccumulator


class MomentsAccumulator(StatisticAccumulator):
    """Track running moments and export any requested subset."""

    output_names = frozenset({"mean", "std"})

    def reset(self) -> None:
        self.count = 0
        self._mean: NDArray[np.float32] | None = None
        self._m2: NDArray[np.float32] | None = None

    def update(
        self,
        values: NDArray[np.float32],
    ) -> None:
        values = np.asarray(values, dtype=np.float32)
        if self.count == 0:
            self._mean = values.copy()
            self._m2 = np.zeros_like(values, dtype=np.float32)
            self.count = 1
            return

        mean = self._mean
        m2 = self._m2
        assert mean is not None
        assert m2 is not None
        if mean.shape != values.shape:
            raise ValueError("all aggregated episode values must share the same shape")

        self.count += 1
        delta = values - mean
        new_mean = (mean + delta / np.float32(self.count)).astype(
            np.float32,
            copy=False,
        )
        delta2 = values - new_mean
        self._mean = new_mean
        self._m2 = (m2 + delta * delta2).astype(np.float32, copy=False)

    def _std(self) -> NDArray[np.float32]:
        assert self._mean is not None
        assert self._m2 is not None
        if self.count == 1:
            return np.zeros_like(self._mean, dtype=np.float32)
        return np.sqrt(self._m2 / self.count).astype(np.float32, copy=False)

    def export(self) -> AggregationOutputs:
        if self._mean is None or self._m2 is None:
            raise ValueError("no observations have been aggregated")

        outputs: AggregationOutputs = {}
        if "mean" in self.requested_outputs:
            outputs["mean"] = self._mean
        if "std" in self.requested_outputs:
            outputs["std"] = self._std()
        return outputs
