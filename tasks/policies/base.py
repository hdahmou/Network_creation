"""Shared runtime behavior for policy tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray

from network_creation.core.agent import Agent

from ..dispatch.runtime import OutputSchema, PreparedTask
from .support import (
    ObserverSpec,
    ParameterTable,
    generate_parameter_table,
    load_observers,
    load_simulation_settings,
)

if TYPE_CHECKING:
    from .results import PolicyTensorGroup
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
        self.validate_parameter_table()
        n_rows = int(self.parameter_table.parameter_values.shape[0])

        return PreparedTask(
            inputs=tuple(range(n_rows)),
            output_schema=OutputSchema(
                fields=tuple(spec.field for spec in self.observer_specs)
            ),
            run=self.run,
        )

    def validate_input_spec(self, input_spec: Mapping[str, object]) -> None:
        parameters = cast(Mapping[str, object], input_spec["parameters"])
        if "visibility_fraction" not in parameters:
            raise ValueError("input_spec.parameters.visibility_fraction is required")

    def validate_sample_idx(self, sample_idx: int) -> None:
        del sample_idx

    def validate_parameter_table(self) -> None:
        visibility_fraction_column = self.parameter_table.column_index_by_name[
            "visibility_fraction"
        ]
        visibility_fraction_values = self.parameter_table.parameter_values[
            :,
            visibility_fraction_column,
        ]
        if np.any((visibility_fraction_values < 0) | (visibility_fraction_values > 1)):
            raise ValueError(
                "input_spec.parameters.visibility_fraction must satisfy "
                "0 <= visibility_fraction <= 1"
            )

    def build_agent(self, node_id: int, sample: NDArray[np.float32]) -> Agent:
        visibility_fraction_column = self.parameter_table.column_index_by_name[
            "visibility_fraction"
        ]
        return Agent(
            id=node_id,
            visibility_fraction=float(sample[visibility_fraction_column]),
            policy=self.build_policy(node_id, sample),
            seed=node_id,
        )

    def run(self, sample_idx: int) -> dict[str, np.ndarray[Any, Any]]:
        from network_creation.core.runtime import create_simulation_runtime

        self.validate_sample_idx(sample_idx)
        sample = self.parameter_table.parameter_values[sample_idx]
        observers = tuple(
            spec.observer_class(**spec.kwargs) for spec in self.observer_specs
        )
        _, _, simulator = create_simulation_runtime(
            simulation_settings=self.simulation_settings,
            build_agent=lambda node_id: self.build_agent(node_id, sample),
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
    @abstractmethod
    def build_output_group(
        cls,
        outputs: dict[str, NDArray[np.generic]],
        input_index: NDArray[np.int64],
    ) -> PolicyTensorGroup:
        """Convert raw stored outputs into formatted tensors."""
