"""Concrete policy task implementations."""

from pathlib import Path

import zarr

from .formatted_results import PolicyFormattedResults
from .myopic_policy import MyopicPolicyRun
from .policy_run import PolicyRun
from .random_policy import RandomPolicyRun


def load_policy_results(zarr_path: str | Path) -> PolicyFormattedResults:
    store_path = Path(zarr_path).expanduser()
    if not store_path.exists():
        raise FileNotFoundError(f"Result store not found: {store_path}")

    root = zarr.open_group(store_path, mode="r")
    task_name = root.attrs.get("task_name")
    if task_name is None:
        raise ValueError(f"Result store {store_path} is missing task_name metadata")

    formatter_map = {
        MyopicPolicyRun.name: MyopicPolicyRun,
        RandomPolicyRun.name: RandomPolicyRun,
    }
    formatter_class = formatter_map.get(str(task_name))
    if formatter_class is None:
        raise ValueError(f"Unsupported policy task: {task_name}")
    return formatter_class.from_zarr(zarr_path)
