"""Tests for execution runtime value objects."""

from __future__ import annotations

import unittest

import numpy as np

from tasks.execution.schema import OutputArraySpec, OutputSpec, PreparedRun


class ExecutionRuntimeTypesTest(unittest.TestCase):
    def test_output_field_metadata_uses_resolved_dtype(self) -> None:
        field = OutputArraySpec(name="vector", shape=(2,), dtype=np.int64)

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
            OutputSpec(
                fields=(
                    OutputArraySpec(name="value", shape=(), dtype=np.float32),
                    OutputArraySpec(name="value", shape=(), dtype=np.float32),
                )
            )

    def test_prepared_task_carries_inputs_schema_and_callable(self) -> None:
        prepared = PreparedRun(
            inputs=(1, 2),
            output_schema=OutputSpec(
                fields=(OutputArraySpec(name="vector", shape=(2,), dtype=np.int64),)
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
