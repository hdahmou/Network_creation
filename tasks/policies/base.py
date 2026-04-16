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
    SimulationSweep,
    generate_parameter_table,
    load_observers,
    load_simulation_sweep,
)

if TYPE_CHECKING:
    from .results import PolicyTensorGroup
    from network_creation.core.runtime import SimulationSettings
    from network_creation.policies.base import Policy


class PolicyRun(ABC):
    """Shared task lifecycle for simulation runs parameterized by a policy builder."""

    name: ClassVar[str]
    simulation_sweep: SimulationSweep
    observer_specs: tuple[ObserverSpec, ...]
    parameter_table: ParameterTable

    def prepare(self, config: Mapping[str, Any]) -> PreparedTask[int]:
        input_spec = config["input_spec"]
        self.validate_input_spec(input_spec)
        self.parameter_table = generate_parameter_table(input_spec)
        self.validate_parameter_table()
        self.simulation_sweep = load_simulation_sweep(
            config["simulation"],
            self.parameter_table,
        )
        self.observer_specs = load_observers(
            raw_observers=config["observers"],
            max_steps=self.simulation_sweep.max_steps,
            n_episodes=self.simulation_sweep.n_episodes,
        )

        return PreparedTask(
            inputs=tuple(range(self.simulation_sweep.total_inputs)),
            output_schema=OutputSchema(
                fields=tuple(spec.field for spec in self.observer_specs)
            ),
            run=self.run,
        )

    def validate_input_spec(self, input_spec: Mapping[str, object]) -> None:
        parameters = cast(Mapping[str, object], input_spec["parameters"])
        if "visibility_distance" not in parameters:
            raise ValueError("input_spec.parameters.visibility_distance is required")

    def validate_sample_idx(self, sample_idx: int) -> None:
        if not 0 <= sample_idx < self.simulation_sweep.total_inputs:
            raise IndexError(f"sample_idx {sample_idx} is out of range")

    def validate_parameter_table(self) -> None:
        visibility_distance_column = self.parameter_table.column_index_by_name[
            "visibility_distance"
        ]
        visibility_distance_values = self.parameter_table.parameter_values[
            :,
            visibility_distance_column,
        ]
        if np.any(visibility_distance_values < 1):
            raise ValueError(
                "input_spec.parameters.visibility_distance must satisfy "
                "visibility_distance >= 1"
            )
        if np.any(visibility_distance_values != np.floor(visibility_distance_values)):
            raise ValueError(
                "input_spec.parameters.visibility_distance must contain only integers"
            )

    def build_agent(
        self,
        node_id: int,
        configuration: NDArray[np.float32],
        simulation_settings: SimulationSettings,
    ) -> Agent:
        visibility_distance_column = self.simulation_sweep.column_index_by_name[
            "visibility_distance"
        ]
        return Agent(
            id=node_id,
            visibility_distance=int(configuration[visibility_distance_column]),
            policy=self.build_policy(node_id, configuration, simulation_settings),
            seed=node_id,
        )

    def _build_observer(
        self,
        spec: ObserverSpec,
        configuration: NDArray[np.float32],
    ) -> Any:
        kwargs = dict(spec.kwargs)
        if spec.name == "normalized_welfare":
            for parameter_name in ("benefit_to_cost_ratio", "distance_discount"):
                if parameter_name in kwargs:
                    continue
                column_index = self.simulation_sweep.column_index_by_name[
                    parameter_name
                ]
                kwargs[parameter_name] = float(configuration[column_index])

        return spec.observer_class(**kwargs)

    def run(self, sample_idx: int) -> dict[str, np.ndarray[Any, Any]]:
        from network_creation.core.runtime import create_simulation_runtime

        self.validate_sample_idx(sample_idx)
        configuration = self.simulation_sweep.configuration_at(sample_idx)
        simulation_settings = self.simulation_sweep.resolve(configuration)
        observers = tuple(
            self._build_observer(spec, configuration) for spec in self.observer_specs
        )
        _, _, simulator = create_simulation_runtime(
            simulation_settings=simulation_settings,
            build_agent=lambda node_id: self.build_agent(
                node_id,
                configuration,
                simulation_settings,
            ),
            observers=observers,
            seed=simulation_settings.seed,
        )
        simulator.run(
            n_episodes=simulation_settings.n_episodes,
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
    def build_policy(
        self,
        node_id: int,
        configuration: NDArray[np.float32],
        simulation_settings: SimulationSettings,
    ) -> Policy:
        """Construct one policy instance for one agent from the sampled row."""

    @classmethod
    @abstractmethod
    def build_output_group(
        cls,
        outputs: dict[str, NDArray[np.generic]],
    ) -> PolicyTensorGroup:
        """Convert raw stored outputs into formatted tensors."""
