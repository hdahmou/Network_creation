"""Tests for policy result dispatch."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import zarr

from tasks.policies import load_policy_results


class PolicyDispatchTest(unittest.TestCase):
    def test_load_policy_results_rejects_unknown_task_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "unknown-policy.zarr"
            root = zarr.open_group(store_path, mode="w", zarr_format=2)
            root.attrs["task_name"] = "unsupported-policy"

            with self.assertRaisesRegex(ValueError, "Unsupported policy task"):
                load_policy_results(store_path)
