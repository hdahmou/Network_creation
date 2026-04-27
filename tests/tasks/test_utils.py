"""Tests for dispatch result loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.dispatch.result_store import ResultStore, load_result_zarr
from tasks.dispatch.runtime import OutputField, OutputSchema


class TaskUtilsTest(unittest.TestCase):
    def test_load_result_zarr_returns_sorted_xarray_content(self) -> None:
        output_schema = OutputSchema(
            fields=(
                OutputField(name="density.mean", shape=(3,), dtype=np.float32),
                OutputField(
                    name="centered_mean_welfare.mean",
                    shape=(3,),
                    dtype=np.float32,
                ),
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "run.zarr"
            writer = ResultStore(
                str(store_path),
                task_name="myopic-policy",
                output_schema=output_schema,
                batch_size=2,
                total_inputs=3,
                metadata_attrs={"custom_attr": "present"},
                metadata_arrays={
                    "grid": np.asarray([[1, 2], [3, 4]], dtype=np.int64),
                },
            )
            for input_index, density, welfare in (
                (2, [0.2, 0.3, 0.4], [20.0, 21.0, 22.0]),
                (0, [0.0, 0.1, 0.2], [0.0, 1.0, 2.0]),
                (1, [0.1, 0.2, 0.3], [10.0, 11.0, 12.0]),
            ):
                writer.add_success(
                    input_index=input_index,
                    output={
                        "density.mean": np.asarray(density, dtype=np.float32),
                        "centered_mean_welfare.mean": np.asarray(
                            welfare,
                            dtype=np.float32,
                        ),
                    },
                    duration_ms=float(input_index + 1),
                )
            writer.add_failure()
            writer.flush()

            result = load_result_zarr(store_path)

        self.assertEqual(result.attrs["path"], store_path.resolve())
        self.assertEqual(result.attrs["task_name"], "myopic-policy")
        self.assertEqual(result.attrs["average_completion_duration_ms"], 2.0)
        self.assertEqual(result.attrs["total_inputs"], 3)
        self.assertEqual(result.attrs["stored_results"], 3)
        self.assertTrue(result.attrs["has_failures"])
        self.assertEqual(result.attrs["custom_attr"], "present")
        np.testing.assert_array_equal(
            result.attrs["metadata"]["grid"],
            np.asarray([[1, 2], [3, 4]], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            result.coords["input_index"].values,
            np.asarray([0, 1, 2]),
        )
        np.testing.assert_allclose(
            result["density.mean"].values,
            np.asarray(
                [[0.0, 0.1, 0.2], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            result["centered_mean_welfare.mean"].values,
            np.asarray(
                [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]],
                dtype=np.float32,
            ),
        )


if __name__ == "__main__":
    unittest.main()
