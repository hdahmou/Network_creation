"""Tests for Zarr-backed task result stores."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from tasks.dispatch.result_store import ResultStore, load_result_zarr, task_class_path
from tasks.dispatch.runtime import OutputField, OutputSchema
from tests.tasks import support_tasks as _support_tasks


class ResultStoreTest(unittest.TestCase):
    @staticmethod
    def _output_schema() -> OutputSchema:
        return OutputSchema(
            fields=(OutputField(name="vector", shape=(2,), dtype=np.int64),)
        )

    def test_task_class_path_uses_module_and_qualname(self) -> None:
        self.assertEqual(
            task_class_path(_support_tasks.VectorTask()),
            "tests.tasks.support_tasks:VectorTask",
        )

    def test_load_result_zarr_raises_for_missing_store(self) -> None:
        missing_path = Path(tempfile.gettempdir()) / "missing-network-creation-store.zarr"

        with self.assertRaisesRegex(FileNotFoundError, "Result store not found"):
            load_result_zarr(missing_path)

    def test_result_store_rejects_invalid_constructor_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "results.zarr"

            with self.assertRaisesRegex(ValueError, "batch_size must be > 0"):
                ResultStore(
                    str(store_path),
                    task_name="vector",
                    task_class="tests.tasks.support_tasks:VectorTask",
                    output_schema=self._output_schema(),
                    batch_size=0,
                    total_inputs=1,
                )

            with self.assertRaisesRegex(ValueError, "total_inputs must be >= 0"):
                ResultStore(
                    str(store_path),
                    task_name="vector",
                    task_class="tests.tasks.support_tasks:VectorTask",
                    output_schema=self._output_schema(),
                    batch_size=1,
                    total_inputs=-1,
                )

    def test_result_store_rejects_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "results.zarr"
            writer = ResultStore(
                str(store_path),
                task_name="vector",
                task_class="tests.tasks.support_tasks:VectorTask",
                output_schema=self._output_schema(),
                batch_size=2,
                total_inputs=1,
            )

            with self.assertRaisesRegex(ValueError, "exported shape"):
                writer.add_success(
                    input_index=0,
                    input_label="0",
                    output={"vector": np.asarray([1, 2, 3], dtype=np.int64)},
                    duration_ms=1.0,
                )

    def test_result_store_batches_failures_and_flushes_pending_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "results.zarr"
            writer = ResultStore(
                str(store_path),
                task_name="vector",
                task_class="tests.tasks.support_tasks:VectorTask",
                output_schema=self._output_schema(),
                batch_size=2,
                total_inputs=3,
            )

            self.assertEqual(writer.add_failure("boom-1"), (0, 0))
            self.assertEqual(writer.add_failure("boom-2"), (0, 2))
            self.assertEqual(writer.add_failure("boom-3"), (0, 0))
            self.assertEqual(writer.flush(), (0, 1))

            root = zarr.open_group(store_path, mode="r")
            np.testing.assert_array_equal(
                np.asarray(root["failures"]["error"]),
                np.asarray(["boom-1", "boom-2", "boom-3"]),
            )
            self.assertEqual(writer.failure_count, 3)


if __name__ == "__main__":
    unittest.main()
