"""Concrete task for myopic-policy simulation sweeps."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..dispatch.registry import register_task
from network_creation.policies.myopic_payoff import MyopicPayoffPolicy

from .policy_run import PolicyRun


@register_task
class myopicPolicyRun(PolicyRun):
    name = "myopic-policy"

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
