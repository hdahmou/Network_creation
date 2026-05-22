"""Tests for execution result loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.execution.zarr_store import ZarrResultStore, load_zarr_results
from tasks.execution.schema import OutputArraySpec, OutputSpec


class TaskUtilsTest(unittest.TestCase):
    def test_load_zarr_results_returns_sorted_xarray_content(self) -> None:
        output_schema = OutputSpec(
            fields=(
                OutputArraySpec(name="density.mean", shape=(3,), dtype=np.float32),
                OutputArraySpec(
                    name="centered_mean_welfare.mean",
                    shape=(3,),
                    dtype=np.float32,
                ),
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "run.zarr"
            writer = ZarrResultStore(
                str(store_path),
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
                )
            writer.add_failure()
            writer.flush()

            result = load_zarr_results(store_path)

        self.assertEqual(result.attrs["path"], store_path.resolve())
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

    def test_load_zarr_results_can_load_selected_outputs_only(self) -> None:
        output_schema = OutputSpec(
            fields=(
                OutputArraySpec(name="density.mean", shape=(2,), dtype=np.float32),
                OutputArraySpec(
                    name="centered_mean_welfare.mean",
                    shape=(2,),
                    dtype=np.float32,
                ),
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "run.zarr"
            writer = ZarrResultStore(
                str(store_path),
                output_schema=output_schema,
                batch_size=2,
                total_inputs=1,
            )
            writer.add_success(
                input_index=0,
                output={
                    "density.mean": np.asarray([0.0, 1.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray(
                        [10.0, 11.0],
                        dtype=np.float32,
                    ),
                },
            )
            writer.flush()

            result = load_zarr_results(
                store_path,
                load_outputs=("density.mean",),
            )

        self.assertIn("density.mean", result)
        self.assertNotIn("centered_mean_welfare.mean", result)
        np.testing.assert_allclose(
            result["density.mean"].values,
            np.asarray([[0.0, 1.0]], dtype=np.float32),
        )

    def test_load_zarr_results_rejects_unknown_selected_outputs(self) -> None:
        output_schema = OutputSpec(
            fields=(
                OutputArraySpec(name="density.mean", shape=(2,), dtype=np.float32),
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "run.zarr"
            writer = ZarrResultStore(
                str(store_path),
                output_schema=output_schema,
                batch_size=2,
                total_inputs=1,
            )
            writer.flush()

            with self.assertRaisesRegex(
                ValueError,
                "unknown output.*density.*density.mean",
            ):
                load_zarr_results(store_path, load_outputs=("density",))


if __name__ == "__main__":
    unittest.main()
