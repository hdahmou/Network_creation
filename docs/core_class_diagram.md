```mermaid
classDiagram

namespace CoreModel {
    class ActionType {
        <<enumeration>>
        ADD
        DELETE
        PASS
        BRIDGE
    }

    class SchedulingMode {
        <<enumeration>>
        SEQUENTIAL
        SIMULTANEOUS
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

    class ActionSpace {
        +actions: dict[ActionType, list[Action]]
        +__getitem__(action_type: ActionType) list[Action]
    }

    class EnvironmentState {
        +graph: Graph
    }

    class Observation {
        +node_id: int
        +graph: Graph
        +outside_node_ids: NDArray
    }
}

namespace Runtime {
    class Environment {
        +seed: int
        +state: EnvironmentState
        +node_ids: list[int]
        +reset() EnvironmentState
        +step(actions: Sequence[Action]) EnvironmentState
        +get_observation(agent: Agent) Observation
        +get_action_space(agent: Agent) ActionSpace
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
        +run(n_episodes: int, reset_observers: bool) void
    }
}

namespace Policies {
    class RandomPolicy
    class ReplayPolicy {
        +actions_by_step: dict[int, tuple[Action, ...]]
        +next_step_id: int
    }
    class MyopicPayoffPolicy {
        +link_cost: float
        +decay: float
        +per_link_benefit: float
        +bridge_probability: float
    }
}

namespace ObserverAPI {
    class StateView {
        +graph: Graph
        +from_state(state: EnvironmentState) StateView
    }

    class EpisodeStartEvent {
        +episode_index: int
        +seed: int
        +mode: SchedulingMode
        +max_steps: int
        +n_agents: int
        +initial_state: StateView
    }

    class StepEvent {
        +step: int
        +mode: SchedulingMode
        +acting_agent_ids: tuple[int, ...]
        +decisions: tuple[PolicyDecision, ...]
        +modified: bool
        +pre_state: StateView
        +post_state: StateView
    }

    class EpisodeEndEvent {
        +episode_index: int
        +seed: int
        +mode: SchedulingMode
        +current_step: int
        +last_modified_step: int
        +final_state: StateView
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

namespace ObserverStatistics {
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

    class EpisodeSeriesTensorStatisticObserver {
        <<abstract>>
        +label: str
        +x_label: str
        +y_label: str
        +y_min: float | None
        +y_max: float | None
        +dimension_names: tuple[str, ...]
        +sample_shape: tuple[int, ...]
        +steps: NDArray
        +values: NDArray
        +compute_initial_value(event: EpisodeStartEvent) NDArray
        +compute_step_value(event: StepEvent) NDArray
    }

    class EpisodeScalarStatisticObserver {
        <<abstract>>
        +compute_terminal_value(event: EpisodeEndEvent) float
    }
}

namespace Observers {
    class DensityObserver
    class ActionCountObserver
    class DistanceObserver
    class IndegreeObserver
    class GlobalClosenessCentralityObserver
    class CentralPointOfDominanceObserver

    class SocialWelfareObserver {
        +link_cost: float
        +decay: float
        +per_link_benefit: float
        +max_welfare: float | None
        +baseline_welfare: float | None
    }

    class HubPersistenceObserver {
        -_window_size: int
    }

    class TriadCensusObserver {
        +dimension_names: tuple[str, ...]
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
Environment *-- EnvironmentState : owns
Environment --> Observation : produces
Environment --> ActionSpace : produces
ActionSpace --> ActionType : keys
ActionSpace --> Action : groups
PolicyDecision --> Action : contains
Action --> ActionType : type

Agent *-- Policy : has
Simulator --> Environment : uses
Simulator --> Agent : schedules
Simulator --> SchedulingMode : mode
Simulator --> SimulationObserver : notifies

RandomPolicy --|> Policy : extends
ReplayPolicy --|> Policy : extends
MyopicPayoffPolicy --|> Policy : extends

EpisodeStartEvent *-- StateView : initial_state
StepEvent *-- StateView : pre/post
StepEvent --> PolicyDecision : decisions
EpisodeEndEvent *-- StateView : final_state
SimulationObserver ..> StateView : reads via events

EpisodeStatisticObserver ..|> SimulationObserver : implements
EpisodeStatisticObserver ..> StatisticResult : returns
EpisodeSeriesStatisticObserver --|> EpisodeStatisticObserver : extends
EpisodeSeriesTensorStatisticObserver --|> EpisodeStatisticObserver : extends
EpisodeScalarStatisticObserver --|> EpisodeStatisticObserver : extends

DensityObserver --|> EpisodeSeriesStatisticObserver : extends
ActionCountObserver --|> EpisodeSeriesStatisticObserver : extends
DistanceObserver --|> EpisodeSeriesStatisticObserver : extends
IndegreeObserver --|> EpisodeSeriesStatisticObserver : extends
GlobalClosenessCentralityObserver --|> EpisodeSeriesStatisticObserver : extends
CentralPointOfDominanceObserver --|> EpisodeSeriesStatisticObserver : extends
SocialWelfareObserver --|> EpisodeSeriesStatisticObserver : extends
HubPersistenceObserver --|> EpisodeSeriesStatisticObserver : extends
TriadCensusObserver --|> EpisodeSeriesTensorStatisticObserver : extends
StateHistoryObserver ..|> SimulationObserver : implements
StateHistoryObserver ..> StateHistoryResult : returns
ActionPrintObserver ..|> SimulationObserver : implements
```
