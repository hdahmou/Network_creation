"""Concrete dasklite task for random-policy baseline runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from numpy.typing import NDArray
from network_creation.policies.random import RandomPolicy
from tasks.policy_run import PolicyRun


class randomPolicyRun(PolicyRun):
    name = "random-policy"

    def validate_input_spec(self, input_spec: Mapping[str, object]) -> None:
        parameters = cast(Mapping[str, object], input_spec["parameters"])
        if parameters:
            raise ValueError(
                "randomPolicyRun requires input_spec.parameters to be empty"
            )
        if int(input_spec["log2_budget"]) != 0:
            raise ValueError(
                "randomPolicyRun requires input_spec.log2_budget to equal 0"
            )

    def validate_sample_idx(self, sample_idx: int) -> None:
        if sample_idx != 0:
            raise ValueError("randomPolicyRun accepts only sample_idx=0")

    def build_policy(self, node_id: int, sample: NDArray[np.float32]) -> RandomPolicy:
        del sample
        return RandomPolicy(seed=node_id)
