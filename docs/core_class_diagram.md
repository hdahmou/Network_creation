This diagram shows the public API and stable extension points. Internal underscore-prefixed methods and internal stats-aggregation plumbing are intentionally omitted.

```mermaid
classDiagram

namespace Core {
    class Environment {
        +seed: int
        +state: EnvironmentState
        +reset() EnvironmentState
        +step(actions: Sequence[Action]) EnvironmentState
        +get_observation(agent: Agent) Observation
        +get_action_space(agent: Agent) ActionSpace
    }

    class EnvironmentState {
        +graph: Graph
        +ownership_matrix: NDArray
    }

    class Observation {
        +node_id: int
        +graph: Graph
        +outside_node_ids: NDArray
        +paid_node_ids: NDArray
    }

    class ActionSpace {
        +actions: Map[ActionType, List[Action]]
        +__getitem__(action_type: ActionType) List[Action]
    }

    class ActionType {
        <<enumeration>>
        ADD
        DELETE
        PASS
        BRIDGE
    }

    class Action {
        +type: ActionType
        +source: int
        +target: int | None
    }
}

namespace Simulation {
    class SchedulingMode {
        <<enumeration>>
        SEQUENTIAL
        SIMULTANEOUS
    }

    class Simulator {
        +rng: Generator
        +agents: Tuple[Agent]
        +n_agents: int
        +n_completed_episodes: int
        +env: Environment
        +mode: SchedulingMode
        +observers: Tuple[SimulationObserver]
        +max_steps: int
        +reset() void
        +get_acting_agents(step: int) Sequence[Agent]
        +run(n_episodes: int) void
    }

    class Agent {
        +id: int
        +radius: int
        +policy: Policy
        +seed: int
        +reset(seed: int) void
        +act(observation: Observation, action_space: ActionSpace) List[Action]
    }

    class Policy {
        <<abstract>>
        +reset(seed: int) void
        +get_actions(observation: Observation, action_space: ActionSpace) List[Action]
    }
}

namespace Observability {
    class StateSnapshot {
        +ownership_matrix: NDArray
        +graph: Graph
        +from_state(state: EnvironmentState) StateSnapshot
    }

    class EpisodeStartEvent {
        +episode_index: int
        +seed: int
        +mode: SchedulingMode
        +max_steps: int
        +n_agents: int
        +initial_state: StateSnapshot
    }

    class StepEvent {
        +step: int
        +mode: SchedulingMode
        +acting_agent_ids: Tuple[int]
        +actions: Tuple[Action]
        +modified: bool
        +pre_state: StateSnapshot
        +post_state: StateSnapshot
    }

    class EpisodeEndEvent {
        +episode_index: int
        +seed: int
        +mode: SchedulingMode
        +current_step: int
        +last_modified_step: int
        +final_state: StateSnapshot
    }

    class StatisticResult {
        +steps: NDArray | None
        +n_episodes: int
        +outputs: Map[str, NDArray | Tuple[NDArray]]
    }

    class SimulationObserver {
        <<interface>>
        +observation_frequency: int
        +reset() void
        +on_episode_start(event: EpisodeStartEvent) void
        +on_step_end(event: StepEvent) void
        +on_episode_end(event: EpisodeEndEvent) void
        +get() Any
    }
}

namespace Observers {
    class EpisodeStatisticObserver {
        <<abstract>>
        +name: str
        +observation_frequency: int
        +aggregations: Tuple[str]
        +n_episodes: int
        +reset() void
        +get() StatisticResult
    }

    class EpisodeSeriesStatisticObserver {
        <<abstract>>
        +label: str
        +x_label: str
        +y_label: str
        +y_min: float | None
        +y_max: float | None
        +steps: NDArray
        +values: NDArray
        +compute_initial_value(event: EpisodeStartEvent) float
        +compute_step_value(event: StepEvent) float
    }

    class EpisodeScalarStatisticObserver {
        <<abstract>>
        +compute_terminal_value(event: EpisodeEndEvent) float
    }

    class DensityObserver {
        +label: str
        +x_label: str
        +y_label: str
        +y_min: float
        +y_max: float
        +steps: NDArray
        +densities: NDArray
    }

    class StateHistoryResult {
        +steps: NDArray
        +ownership_matrices: NDArray
        +n_episodes: int
        +runs_steps: Tuple[NDArray] | None
        +runs_ownership_matrices: Tuple[NDArray] | None
    }

    class StateHistoryObserver {
        +observation_frequency: int
        +keep_runs: bool
        +steps: NDArray
        +ownership_matrices: NDArray
        +graphs: List[Graph]
        +reset() void
        +get() StateHistoryResult
    }

    class ActionPrintObserver {
        +observation_frequency: int
        +print_empty_steps: bool
        +reset() void
        +get() None
    }
}

%% Relationships
Simulator --> Environment : uses
Simulator --> Agent : schedules
Simulator --> SchedulingMode : mode
Simulator --> SimulationObserver : notifies
Environment *-- EnvironmentState : owns
Environment --> Observation : produces
Environment --> ActionSpace : produces
EpisodeStartEvent *-- StateSnapshot : initial_state
StepEvent *-- StateSnapshot : pre/post
EpisodeEndEvent *-- StateSnapshot : final_state
SimulationObserver ..> StateSnapshot : reads via events
EpisodeStatisticObserver ..> StatisticResult : returns
StateHistoryObserver ..> StateHistoryResult : returns
EpisodeStatisticObserver ..|> SimulationObserver : implements
EpisodeSeriesStatisticObserver --|> EpisodeStatisticObserver : extends
EpisodeScalarStatisticObserver --|> EpisodeStatisticObserver : extends
DensityObserver --|> EpisodeSeriesStatisticObserver : extends
StateHistoryObserver ..|> SimulationObserver : implements
ActionPrintObserver ..|> SimulationObserver : implements
ActionSpace --> ActionType : keys
ActionSpace --> Action : groups
Action --> ActionType : type
Agent *-- Policy : has
```
