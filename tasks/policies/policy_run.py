"""Shared task template for policy-driven simulation runs."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray
from ..dispatch.result_store import load_result_zarr
from ..dispatch.runtime import OutputSchema, PreparedTask

from .utils import (
    ObserverSpec,
    ParameterTable,
    generate_parameter_table,
    load_observers,
    load_simulation_settings,
)

if TYPE_CHECKING:
    from .formatted_results import PolicyFormattedResults, PolicyTensorGroup
    from network_creation.core.runtime import SimulationSettings
    from network_creation.policies.base import Policy


class PolicyRun(ABC):
    """Shared task lifecycle for simulation runs parameterized by a policy builder."""

    name: ClassVar[str]
    simulation_settings: SimulationSettings
    observer_specs: tuple[ObserverSpec, ...]
    parameter_table: ParameterTable

    def prepare(self, config: Mapping[str, Any]) -> PreparedTask[int]:
        self.simulation_settings = load_simulation_settings(config["simulation"])
        self.observer_specs = load_observers(
            raw_observers=config["observers"],
            simulation_settings=self.simulation_settings,
        )

        input_spec = config["input_spec"]
        self.validate_input_spec(input_spec)
        self.parameter_table = generate_parameter_table(input_spec)
        n_rows = int(self.parameter_table.parameter_values.shape[0])
        return PreparedTask(
            inputs=tuple(range(n_rows)),
            output_schema=OutputSchema(
                fields=tuple(spec.field for spec in self.observer_specs)
            ),
            run=self.run,
        )

    def validate_input_spec(self, input_spec: Mapping[str, object]) -> None:
        del input_spec

    def validate_sample_idx(self, sample_idx: int) -> None:
        del sample_idx

    def run(self, sample_idx: int) -> dict[str, np.ndarray[Any, Any]]:
        from network_creation.core.runtime import create_simulation_runtime

        self.validate_sample_idx(sample_idx)
        sample = self.parameter_table.parameter_values[sample_idx]
        observers = tuple(
            spec.observer_class(**spec.kwargs) for spec in self.observer_specs
        )
        _, _, simulator = create_simulation_runtime(
            simulation_settings=self.simulation_settings,
            build_policy=lambda node_id: self.build_policy(node_id, sample),
            observers=observers,
            seed=self.simulation_settings.seed,
        )
        simulator.run(
            n_episodes=self.simulation_settings.n_episodes,
            reset_observers=True,
        )
        return {
            spec.field.name: np.asarray(
                observer.get().outputs[spec.output],
                dtype=spec.observer_class.fixed_output_dtype(),
            )
            for spec, observer in zip(
                self.observer_specs,
                observers,
                strict=True,
            )
        }

    @abstractmethod
    def build_policy(self, node_id: int, sample: NDArray[np.float32]) -> Policy:
        """Construct one policy instance for one agent from the sampled row."""

    @classmethod
    def from_zarr(cls, zarr_path: str | Path) -> PolicyFormattedResults:
        """Load formatted policy results from a result store."""
        from .formatted_results import PolicyFormattedResults

        result_data = load_result_zarr(zarr_path)
        result_store_path = Path(result_data["path"])
        config_path = result_store_path.with_suffix(".dispatch_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(
                f"Dispatch config at {config_path} must contain a JSON object"
            )

        parameter_table = generate_parameter_table(cast(dict[str, object], config["input_spec"]))
        input_index = np.asarray(result_data["input_index"], dtype=np.int64)
        parameter_values = np.asarray(
            parameter_table.parameter_values[input_index],
            dtype=np.float32,
        )
        output_group = cls.build_output_group(
            outputs=cast(dict[str, NDArray[np.generic]], result_data["outputs"]),
            input_index=input_index,
        )

        return PolicyFormattedResults(
            inputs={
                "tensors": {"parameters": parameter_values},
                "dims": {"parameters": ("sample", "parameter")},
                "labels": {
                    "sample": input_index,
                    "parameter": parameter_table.parameter_names,
                },
            },
            outputs=output_group,
            metadata={
                "task_name": str(result_data["task_name"]),
                "task_class": str(result_data["task_class"]),
                "result_store_path": result_store_path,
                "config_path": config_path,
                "average_completion_duration_ms": cast(
                    float | None, result_data["average_completion_duration_ms"]
                ),
                "input_index": input_index,
                "output_schema": list(result_data["output_schema"]),
            },
            failures=cast(dict[str, NDArray[np.int64] | NDArray[np.float64] | NDArray[np.str_]], result_data["failures"]),
        )

    @classmethod
    @abstractmethod
    def build_output_group(
        cls,
        outputs: dict[str, NDArray[np.generic]],
        input_index: NDArray[np.int64],
    ) -> PolicyTensorGroup:
        """Convert raw stored outputs into formatted tensors."""
