"""Concrete task for myopic-policy simulation sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from ..dispatch.registry import register_task
from network_creation.policies.myopic_payoff import MyopicPayoffPolicy

from .formatted_results import PolicyFormattedResults, load_policy_result_data
from .policy_run import PolicyRun


@register_task
class MyopicPolicyRun(PolicyRun):
    name: ClassVar[str] = "myopic-policy"

    def build_policy(
        self,
        node_id: int,
        sample: NDArray[np.float32],
    ) -> MyopicPayoffPolicy:
        parameter_columns = self.parameter_table.column_index_by_name
        return MyopicPayoffPolicy(
            seed=node_id,
            link_cost=float(sample[parameter_columns["link_cost"]]),
            decay=float(sample[parameter_columns["decay"]]),
            per_link_benefit=float(sample[parameter_columns["per_link_benefit"]]),
            bridge_probability=float(sample[parameter_columns["bridge_probability"]]),
        )
    @classmethod
    def from_zarr(cls, zarr_path: str | Path) -> PolicyFormattedResults:
        result_data, config_path, input_index, parameter_table, parameter_values = (
            load_policy_result_data(zarr_path)
        )
        observable_names, series_arrays = cls._collect_timeseries_outputs(result_data["outputs"])
        if not series_arrays:
            raise ValueError("myopic-policy results do not contain any time-series outputs")

        timeseries = np.stack(series_arrays, axis=1).astype(np.float32, copy=False)
        final = np.asarray(timeseries[:, :, -1], dtype=np.float32)

        formatted = PolicyFormattedResults(
            inputs={
                "tensors": {"parameters": parameter_values},
                "dims": {"parameters": ("sample", "parameter")},
                "labels": {
                    "sample": input_index,
                    "parameter": parameter_table.parameter_names,
                },
            },
            outputs={
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
            },
            metadata={
                "task_name": str(result_data["task_name"]),
                "task_class": str(result_data["task_class"]),
                "result_store_path": Path(result_data["path"]),
                "config_path": config_path,
                "input_index": input_index,
                "output_schema": list(result_data["output_schema"]),
            },
            failures=result_data["failures"],
        )
        return formatted

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
                    "myopic-policy time-series outputs must share the same step axis length"
                )
            series_names.append(name)
            series_arrays.append(np.asarray(array, dtype=np.float32))

        return tuple(series_names), series_arrays
