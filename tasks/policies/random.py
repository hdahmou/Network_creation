"""Random policy task."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from network_creation.policies.random import RandomPolicy

from .base import PolicyRun
from .results import PolicyOutput


class RandomPolicyRun(PolicyRun):
    name: ClassVar[str] = "random-policy"

    def build_policy(
        self,
        node_id: int,
        _configuration: NDArray[np.float32],
        _simulation_settings,
    ) -> RandomPolicy:
        return RandomPolicy(seed=node_id)

    @classmethod
    def build_output_group(
        cls,
        outputs: dict[str, NDArray[np.generic]],
    ) -> PolicyOutput:
        timeseries, observable_names = cls._build_output_tensors(outputs)
        return PolicyOutput(
            timeseries=timeseries,
            observable_names=observable_names,
            step_labels=np.arange(timeseries.shape[1], dtype=np.int64),
        )

    @staticmethod
    def _build_output_tensors(
        outputs: dict[str, NDArray[np.generic]],
    ) -> tuple[NDArray[np.float32], tuple[str, ...]]:
        series_names: list[str] = []
        series_arrays: list[NDArray[np.float32]] = []
        scalar_names: list[str] = []
        scalar_arrays: list[NDArray[np.float32]] = []
        expected_steps: int | None = None

        for name, values in outputs.items():
            array = np.asarray(values)
            if array.ndim == 2:
                if expected_steps is None:
                    expected_steps = array.shape[1]
                elif array.shape[1] != expected_steps:
                    raise ValueError(
                        "random-policy time-series outputs must share the same step axis length"
                    )
                series_names.append(name)
                series_arrays.append(np.asarray(array, dtype=np.float32))
                continue

            if array.ndim == 1:
                scalar_names.append(name)
                scalar_arrays.append(np.asarray(array, dtype=np.float32))

        if series_arrays:
            timeseries = np.stack(series_arrays, axis=2).astype(np.float32, copy=False)
            return timeseries, tuple(series_names)

        if scalar_arrays:
            scalar_values = np.stack(scalar_arrays, axis=1).astype(
                np.float32,
                copy=False,
            )
            return scalar_values[:, None, :], tuple(scalar_names)

        raise ValueError(
            "random-policy results must contain scalar or time-series outputs"
        )
