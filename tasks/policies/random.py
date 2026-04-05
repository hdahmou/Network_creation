"""Random policy task."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from network_creation.policies.random import RandomPolicy

from .base import PolicyRun
from .results import DimLabels, PolicyTensorGroup


class RandomPolicyRun(PolicyRun):
    name: ClassVar[str] = "random-policy"

    def validate_input_spec(self, input_spec: Mapping[str, object]) -> None:
        super().validate_input_spec(input_spec)

    def build_policy(self, node_id: int, sample: NDArray[np.float32]) -> RandomPolicy:
        del sample
        return RandomPolicy(seed=node_id)

    @classmethod
    def build_output_group(
        cls,
        outputs: dict[str, NDArray[np.generic]],
        input_index: NDArray[np.int64],
    ) -> PolicyTensorGroup:
        final, observable_names, timeseries = cls._build_output_tensors(outputs)

        output_tensors: dict[str, NDArray[np.float32]] = {"final": final}
        output_dims = {"final": ("sample", "observable")}
        output_labels: dict[str, DimLabels] = {
            "sample": input_index,
            "observable": observable_names,
        }
        if timeseries is not None:
            output_tensors["timeseries"] = timeseries
            output_dims["timeseries"] = ("sample", "observable", "step")
            output_labels["step"] = np.arange(timeseries.shape[2], dtype=np.int64)

        return {"tensors": output_tensors, "dims": output_dims, "labels": output_labels}

    @staticmethod
    def _build_output_tensors(
        outputs: dict[str, NDArray[np.generic]],
    ) -> tuple[NDArray[np.float32], tuple[str, ...], NDArray[np.float32] | None]:
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
            timeseries = np.stack(series_arrays, axis=1).astype(np.float32, copy=False)
            final = np.asarray(timeseries[:, :, -1], dtype=np.float32)
            return final, tuple(series_names), timeseries

        if scalar_arrays:
            final = np.stack(scalar_arrays, axis=1).astype(np.float32, copy=False)
            return final, tuple(scalar_names), None

        raise ValueError(
            "random-policy results must contain scalar or time-series outputs"
        )
