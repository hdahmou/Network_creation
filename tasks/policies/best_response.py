"""Best-response policy task."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from network_creation.policies.best_response import BestResponsePolicy

from .base import PolicyRun
from .results import PolicyOutput
from .support import collect_timeseries_outputs


class BestResponsePolicyRun(PolicyRun):
    name: ClassVar[str] = "best-response-policy"

    def validate_simulation_sweep(self) -> None:
        parameter_columns = self.simulation_sweep.column_index_by_name
        configuration_values = self.simulation_sweep.configuration_values

        benefit_to_cost_ratio = configuration_values[
            :, :, parameter_columns["benefit_to_cost_ratio"]
        ]
        if np.any((benefit_to_cost_ratio <= 0.0) | (benefit_to_cost_ratio > 1.0)):
            raise ValueError(
                "input_spec.parameters.benefit_to_cost_ratio must be in (0, 1]"
            )

        distance_discount = configuration_values[
            :, :, parameter_columns["distance_discount"]
        ]
        if np.any((distance_discount <= 0.0) | (distance_discount >= 1.0)):
            raise ValueError(
                "input_spec.parameters.distance_discount must be in (0, 1)"
            )

        outside_optimism = configuration_values[
            :, :, parameter_columns["outside_optimism"]
        ]
        if np.any((outside_optimism < 0.0) | (outside_optimism > 1.0)):
            raise ValueError(
                "input_spec.parameters.outside_optimism must be in [0, 1]"
            )

        exploration = configuration_values[:, :, parameter_columns["exploration"]]
        if np.any(exploration <= 0.0):
            raise ValueError(
                "input_spec.parameters.exploration must be greater than 0"
            )

    def build_policy(
        self,
        node_id: int,
        configuration: NDArray[np.float32],
        simulation_settings,
    ) -> BestResponsePolicy:
        parameter_columns = self.simulation_sweep.column_index_by_name
        return BestResponsePolicy(
            n_nodes=simulation_settings.n_agents,
            seed=node_id,
            benefit_to_cost_ratio=float(
                configuration[parameter_columns["benefit_to_cost_ratio"]]
            ),
            distance_discount=float(configuration[parameter_columns["distance_discount"]]),
            exploration=float(configuration[parameter_columns["exploration"]]),
            outside_optimism=float(
                configuration[parameter_columns["outside_optimism"]]
            ),
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
            raise ValueError(
                "best-response-policy results do not contain any time-series outputs"
            )

        timeseries = np.stack(series_arrays, axis=2).astype(np.float32, copy=False)
        return PolicyOutput(
            timeseries=timeseries,
            observable_names=observable_names,
            step_labels=np.arange(timeseries.shape[1], dtype=np.int64),
        )
