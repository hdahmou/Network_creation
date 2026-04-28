"""Tests for Zarr-backed task result stores."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from tasks.execution.zarr_store import ZarrResultStore, load_zarr_results
from tasks.execution.schema import OutputArraySpec, OutputSpec


class ZarrResultStoreTest(unittest.TestCase):
    @staticmethod
    def _output_schema() -> OutputSpec:
        return OutputSpec(
            fields=(OutputArraySpec(name="vector", shape=(2,), dtype=np.int64),)
        )

    def test_load_zarr_results_raises_for_missing_store(self) -> None:
        missing_path = Path(tempfile.gettempdir()) / "missing-network-creation-store.zarr"

        with self.assertRaisesRegex(FileNotFoundError, "Result store not found"):
            load_zarr_results(missing_path)

    def test_result_store_rejects_invalid_constructor_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "results.zarr"

            with self.assertRaisesRegex(ValueError, "batch_size must be > 0"):
                ZarrResultStore(
                    str(store_path),
                    output_schema=self._output_schema(),
                    batch_size=0,
                    total_inputs=1,
                )

            with self.assertRaisesRegex(ValueError, "total_inputs must be >= 0"):
                ZarrResultStore(
                    str(store_path),
                    output_schema=self._output_schema(),
                    batch_size=1,
                    total_inputs=-1,
                )

    def test_result_store_rejects_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "results.zarr"
            writer = ZarrResultStore(
                str(store_path),
                output_schema=self._output_schema(),
                batch_size=2,
                total_inputs=1,
            )

            with self.assertRaisesRegex(ValueError, "exported shape"):
                writer.add_success(
                    input_index=0,
                    output={"vector": np.asarray([1, 2, 3], dtype=np.int64)},
                    duration_ms=1.0,
                )

    def test_result_store_records_failure_flag_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "results.zarr"
            writer = ZarrResultStore(
                str(store_path),
                output_schema=self._output_schema(),
                batch_size=2,
                total_inputs=3,
                metadata_attrs={"custom_attr": "present"},
                metadata_arrays={"grid": np.asarray([[1, 2], [3, 4]], dtype=np.int64)},
            )

            writer.add_failure()
            writer.add_failure()
            writer.add_failure()
            writer.flush()

            root = zarr.open_group(store_path, mode="r")
            self.assertEqual(root.attrs["custom_attr"], "present")
            self.assertTrue(root.attrs["has_failures"])
            np.testing.assert_array_equal(
                np.asarray(root["metadata"]["grid"]),
                np.asarray([[1, 2], [3, 4]], dtype=np.int64),
            )
            self.assertNotIn("failures", root)
            self.assertTrue(writer.has_failures)


if __name__ == "__main__":
    unittest.main()
