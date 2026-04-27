"""Myopic policy task."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from network_creation.policies.myopic_payoff import MyopicPayoffPolicy

from .base import PolicyRun
from .results import PolicyOutput
from .support import collect_timeseries_outputs


class MyopicPolicyRun(PolicyRun):
    name: ClassVar[str] = "myopic-policy"

    def build_policy(
        self,
        node_id: int,
        configuration: NDArray[np.float32],
        _simulation_settings,
    ) -> MyopicPayoffPolicy:
        parameter_columns = self.simulation_sweep.column_index_by_name
        return MyopicPayoffPolicy(
            seed=node_id,
            benefit_to_cost_ratio=float(
                configuration[parameter_columns["benefit_to_cost_ratio"]]
            ),
            distance_discount=float(
                configuration[parameter_columns["distance_discount"]]
            ),
            bridge_probability=float(configuration[parameter_columns["bridge_probability"]]),
        )

    @classmethod
    def build_output_group(
        cls,
        outputs: dict[str, NDArray[np.generic]],
    ) -> PolicyOutput:
        observable_names, series_arrays = collect_timeseries_outputs(
            outputs,
            task_name=cls.name,
        )
        if not series_arrays:
            raise ValueError("myopic-policy results do not contain any time-series outputs")

        timeseries = np.stack(series_arrays, axis=2).astype(np.float32, copy=False)
        return PolicyOutput(
            timeseries=timeseries,
            observable_names=observable_names,
            step_labels=np.arange(timeseries.shape[1], dtype=np.int64),
        )
