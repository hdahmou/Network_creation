"""Best-response policy task."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from network_creation.policies.best_response import BestResponsePolicy

from .base import PolicyRun
from .results import PolicyTensorGroup


class BestResponsePolicyRun(PolicyRun):
    name: ClassVar[str] = "best-response-policy"

    def build_policy(
        self,
        node_id: int,
        sample: NDArray[np.float32],
    ) -> BestResponsePolicy:
        parameter_columns = self.parameter_table.column_index_by_name
        return BestResponsePolicy(
            n_nodes=self.simulation_settings.n_agents,
            seed=node_id,
            benefit_to_cost_ratio=float(
                sample[parameter_columns["benefit_to_cost_ratio"]]
            ),
            distance_discount=float(sample[parameter_columns["distance_discount"]]),
            bridge_value=float(sample[parameter_columns["bridge_value"]]),
            inertia_term=float(sample[parameter_columns["inertia_term"]]),
            tau=float(sample[parameter_columns["tau"]]),
        )

    @classmethod
    def build_output_group(
        cls,
        outputs: dict[str, NDArray[np.generic]],
        input_index: NDArray[np.int64],
    ) -> PolicyTensorGroup:
        observable_names, series_arrays = cls._collect_timeseries_outputs(outputs)
        if not series_arrays:
            raise ValueError(
                "best-response-policy results do not contain any time-series outputs"
            )

        timeseries = np.stack(series_arrays, axis=1).astype(np.float32, copy=False)
        final = np.asarray(timeseries[:, :, -1], dtype=np.float32)

        return {
            "tensors": {
                "timeseries": timeseries,
                "final": final,
            },
            "dims": {
                "timeseries": ("sample", "observable", "step"),
                "final": ("sample", "observable"),
            },
            "labels": {
                "sample": input_index,
                "observable": observable_names,
                "step": np.arange(timeseries.shape[2], dtype=np.int64),
            },
        }

    @staticmethod
    def _collect_timeseries_outputs(
        outputs: dict[str, NDArray[np.generic]],
    ) -> tuple[tuple[str, ...], list[NDArray[np.float32]]]:
        series_names: list[str] = []
        series_arrays: list[NDArray[np.float32]] = []
        expected_steps: int | None = None

        for name, values in outputs.items():
            array = np.asarray(values)
            if array.ndim != 2:
                continue
            if expected_steps is None:
                expected_steps = array.shape[1]
            elif array.shape[1] != expected_steps:
                raise ValueError(
                    "best-response-policy time-series outputs must share "
                    "the same step axis length"
                )
            series_names.append(name)
            series_arrays.append(np.asarray(array, dtype=np.float32))

        return tuple(series_names), series_arrays
