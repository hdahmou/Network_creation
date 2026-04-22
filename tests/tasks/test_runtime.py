"""Tests for dispatch runtime value objects."""

from __future__ import annotations

import unittest

import numpy as np

from tasks.dispatch.runtime import OutputField, OutputSchema, PreparedTask


class DispatchRuntimeTypesTest(unittest.TestCase):
    def test_output_field_metadata_uses_resolved_dtype(self) -> None:
        field = OutputField(name="vector", shape=(2,), dtype=np.int64)

        self.assertEqual(field.resolved_dtype, np.dtype(np.int64))
        self.assertEqual(
            field.metadata(),
            {"name": "vector", "shape": [2], "dtype": "int64"},
        )

    def test_output_schema_rejects_duplicate_field_names(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "output schema field names must be unique",
        ):
            OutputSchema(
                fields=(
                    OutputField(name="value", shape=(), dtype=np.float32),
                    OutputField(name="value", shape=(), dtype=np.float32),
                )
            )

    def test_output_schema_field_map_indexes_by_name(self) -> None:
        schema = OutputSchema(
            fields=(
                OutputField(name="vector", shape=(2,), dtype=np.int64),
                OutputField(name="score", shape=(), dtype=np.float32),
            )
        )

        self.assertEqual(set(schema.field_map), {"vector", "score"})
        self.assertEqual(schema.field_map["score"].shape, ())

    def test_prepared_task_uses_default_descriptions(self) -> None:
        prepared = PreparedTask(
            inputs=(1, 2),
            output_schema=OutputSchema(
                fields=(OutputField(name="vector", shape=(2,), dtype=np.int64),)
            ),
            run=lambda value: {"vector": np.asarray([value, value + 1], dtype=np.int64)},
        )

        self.assertEqual(prepared.describe_input(7), "7")
        self.assertEqual(
            prepared.describe_output(
                {
                    "vector": np.asarray([1, 2], dtype=np.int64),
                    "score": np.asarray(0.5, dtype=np.float32),
                }
            ),
            "vector:shape=(2,) dtype=int64; score:shape=() dtype=float32",
        )

    def test_prepared_task_uses_custom_descriptions(self) -> None:
        prepared = PreparedTask(
            inputs=(1,),
            output_schema=OutputSchema(
                fields=(OutputField(name="vector", shape=(2,), dtype=np.int64),)
            ),
            run=lambda value: {"vector": np.asarray([value, value + 1], dtype=np.int64)},
            describe_input_fn=lambda value: f"input={value}",
            describe_output_fn=lambda output: f"keys={sorted(output)}",
        )

        self.assertEqual(prepared.describe_input(3), "input=3")
        self.assertEqual(
            prepared.describe_output({"vector": np.asarray([3, 4], dtype=np.int64)}),
            "keys=['vector']",
        )


if __name__ == "__main__":
    unittest.main()
