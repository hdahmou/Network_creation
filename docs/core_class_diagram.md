```mermaid
classDiagram

namespace Core {
    class Environment {
        +seed: int
        +state: EnvironmentState
        +node_ids: list[int]
        +reset() EnvironmentState
        +step(actions: Sequence[Action]) EnvironmentState
        +get_observation(agent: Agent) Observation
        +get_action_space(agent: Agent) ActionSpace
    }

    class EnvironmentState {
        +graph: Graph
    }

    class Observation {
        +node_id: int
        +graph: Graph
        +outside_node_ids: NDArray
    }

    class ActionSpace {
        +actions: dict[ActionType, list[Action]]
        +__getitem__(action_type: ActionType) list[Action]
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

    class PolicyDecision {
        +actions: tuple[Action, ...]
        +value: dict[str, object] | None
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
        +agents: tuple[Agent, ...]
        +n_agents: int
        +n_completed_episodes: int
        +env: Environment
        +mode: SchedulingMode
        +observers: tuple[SimulationObserver, ...]
        +max_steps: int
        +stagnation_patience: int | None
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
        +act(observation: Observation, action_space: ActionSpace) PolicyDecision
    }

    class Policy {
        <<abstract>>
        +reset(seed: int) void
        +get_decision(observation: Observation, action_space: ActionSpace) PolicyDecision
    }
}

namespace Policies {
    class RandomPolicy {
        +reset(seed: int) void
        +get_decision(observation: Observation, action_space: ActionSpace) PolicyDecision
    }

    class ReplayPolicy {
        +actions_by_step: dict[int, tuple[Action, ...]]
        +next_step_id: int
        +reset(seed: int) void
        +get_decision(observation: Observation, action_space: ActionSpace) PolicyDecision
    }

    class MyopicPayoffPolicy {
        +link_cost: float
        +decay: float
        +per_link_benefit: float
        +bridge_probability: float
        +reset(seed: int) void
        +get_decision(observation: Observation, action_space: ActionSpace) PolicyDecision
    }
}

namespace Observability {
    class StateSnapshot {
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
        +acting_agent_ids: tuple[int, ...]
        +decisions: tuple[PolicyDecision, ...]
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
        +outputs: dict[str, NDArray | tuple[NDArray, ...]]
        +sample_shape: tuple[int, ...]
        +dimension_names: tuple[str, ...] | None
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
        +aggregations: tuple[str, ...]
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
        +y_max: float | None
        +steps: NDArray
        +densities: NDArray
    }

    class ActionCountObserver {
        +label: str
        +x_label: str
        +y_label: str
        +y_min: float
        +y_max: float | None
        +steps: NDArray
        +average_action_counts: NDArray
    }

    class StateHistoryResult {
        +steps: NDArray
        +graphs: tuple[Graph, ...]
        +n_episodes: int
        +runs_steps: tuple[NDArray, ...] | None
        +runs_graphs: tuple[tuple[Graph, ...], ...] | None
    }

    class StateHistoryObserver {
        +observation_frequency: int
        +keep_runs: bool
        +steps: NDArray
        +graphs: NDArray
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
RandomPolicy --|> Policy : extends
ReplayPolicy --|> Policy : extends
MyopicPayoffPolicy --|> Policy : extends
Environment *-- EnvironmentState : owns
Environment --> Observation : produces
Environment --> ActionSpace : produces
PolicyDecision --> Action : contains
EpisodeStartEvent *-- StateSnapshot : initial_state
StepEvent *-- StateSnapshot : pre/post
StepEvent --> PolicyDecision : decisions
EpisodeEndEvent *-- StateSnapshot : final_state
SimulationObserver ..> StateSnapshot : reads via events
EpisodeStatisticObserver ..> StatisticResult : returns
StateHistoryObserver ..> StateHistoryResult : returns
EpisodeStatisticObserver ..|> SimulationObserver : implements
EpisodeSeriesStatisticObserver --|> EpisodeStatisticObserver : extends
EpisodeScalarStatisticObserver --|> EpisodeStatisticObserver : extends
DensityObserver --|> EpisodeSeriesStatisticObserver : extends
ActionCountObserver --|> EpisodeSeriesStatisticObserver : extends
StateHistoryObserver ..|> SimulationObserver : implements
ActionPrintObserver ..|> SimulationObserver : implements
ActionSpace --> ActionType : keys
ActionSpace --> Action : groups
Action --> ActionType : type
Agent *-- Policy : has
```
