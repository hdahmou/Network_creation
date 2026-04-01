"""Shared Dask task template for policy-driven simulation runs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from dasklite.base_task import BaseTask, OutputSchema, PreparedTask
from tasks.utils import (
    ObserverSpec,
    ParameterTable,
    generate_parameter_table,
    load_observers,
    load_simulation_settings,
)

if TYPE_CHECKING:
    from network_creation.core.runtime import SimulationSettings
    from network_creation.policies.base import Policy


class PolicyRun(BaseTask[int], ABC):
    """Shared task lifecycle for simulation runs parameterized by a policy builder."""

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
