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

    def test_prepared_task_carries_inputs_schema_and_callable(self) -> None:
        prepared = PreparedTask(
            inputs=(1, 2),
            output_schema=OutputSchema(
                fields=(OutputField(name="vector", shape=(2,), dtype=np.int64),)
            ),
            run=lambda value: {"vector": np.asarray([value, value + 1], dtype=np.int64)},
        )

        self.assertEqual(prepared.inputs, (1, 2))
        self.assertEqual(
            prepared.run(7)["vector"].tolist(),
            [7, 8],
        )


if __name__ == "__main__":
    unittest.main()
