This diagram shows the public API and stable extension points. Internal underscore-prefixed methods and internal stats-aggregation plumbing are intentionally omitted.

```mermaid
classDiagram

namespace Core {
    class Environment {
        +seed: int
        +state: EnvironmentState
        +reset() EnvironmentState
        +step(actions: Sequence[Action]) EnvironmentState
        +getObservation(agent: Agent) Observation
        +getActionSpace(agent: Agent) ActionSpace
    }

    class EnvironmentState {
        +graph: Graph
        +ownershipMatrix: NDArray
    }

    class Observation {
        +nodeId: int
        +graph: Graph
        +outsideNodeIds: NDArray
        +paidNodeIds: NDArray
    }

    class ActionSpace {
        +actions: Map[ActionType, List[Action]]
        +__getitem__(actionType: ActionType) List[Action]
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
        +nAgents: int
        +nCompletedEpisodes: int
        +env: Environment
        +mode: SchedulingMode
        +observers: Tuple[SimulationObserver]
        +maxSteps: int
        +stagnationPatience: int | None
        +reset() void
        +getActingAgents(step: int) Sequence[Agent]
        +run(nEpisodes: int) void
    }

    class Agent {
        +id: int
        +radius: int
        +policy: Policy
        +seed: int
        +reset(seed: int) void
        +act(observation: Observation, actionSpace: ActionSpace) List[Action]
    }

    class Policy {
        <<abstract>>
        +reset(seed: int) void
        +getActions(observation: Observation, actionSpace: ActionSpace) List[Action]
    }
}

namespace Policies {
    class RandomPolicy {
        +reset(seed: int) void
        +getActions(observation: Observation, actionSpace: ActionSpace) List[Action]
    }

    class ReplayPolicy {
        +actionsByStep: Map[int, Tuple[Action]]
        +reset(seed: int) void
        +getActions(observation: Observation, actionSpace: ActionSpace) List[Action]
    }
}

namespace Observability {
    class StateSnapshot {
        +ownershipMatrix: NDArray
        +graph: Graph
        +fromState(state: EnvironmentState) StateSnapshot
    }

    class EpisodeStartEvent {
        +episodeIndex: int
        +seed: int
        +mode: SchedulingMode
        +maxSteps: int
        +nAgents: int
        +initialState: StateSnapshot
    }

    class StepEvent {
        +step: int
        +mode: SchedulingMode
        +actingAgentIds: Tuple[int]
        +actions: Tuple[Action]
        +modified: bool
        +preState: StateSnapshot
        +postState: StateSnapshot
    }

    class EpisodeEndEvent {
        +episodeIndex: int
        +seed: int
        +mode: SchedulingMode
        +currentStep: int
        +lastModifiedStep: int
        +finalState: StateSnapshot
    }

    class StatisticResult {
        +steps: NDArray | None
        +nEpisodes: int
        +outputs: Map[str, NDArray | Tuple[NDArray]]
    }

    class SimulationObserver {
        <<interface>>
        +observationFrequency: int
        +reset() void
        +onEpisodeStart(event: EpisodeStartEvent) void
        +onStepEnd(event: StepEvent) void
        +onEpisodeEnd(event: EpisodeEndEvent) void
        +get() Any
    }
}

namespace Observers {
    class EpisodeStatisticObserver {
        <<abstract>>
        +name: str
        +observationFrequency: int
        +aggregations: Tuple[str]
        +nEpisodes: int
        +reset() void
        +get() StatisticResult
    }

    class EpisodeSeriesStatisticObserver {
        <<abstract>>
        +label: str
        +xLabel: str
        +yLabel: str
        +yMin: float | None
        +yMax: float | None
        +steps: NDArray
        +values: NDArray
        +computeInitialValue(event: EpisodeStartEvent) float
        +computeStepValue(event: StepEvent) float
    }

    class EpisodeScalarStatisticObserver {
        <<abstract>>
        +computeTerminalValue(event: EpisodeEndEvent) float
    }

    class DensityObserver {
        +label: str
        +xLabel: str
        +yLabel: str
        +yMin: float
        +yMax: float
        +steps: NDArray
        +densities: NDArray
    }

    class StateHistoryResult {
        +steps: NDArray
        +ownershipMatrices: NDArray
        +nEpisodes: int
        +runsSteps: Tuple[NDArray] | None
        +runsOwnershipMatrices: Tuple[NDArray] | None
    }

    class StateHistoryObserver {
        +observationFrequency: int
        +keepRuns: bool
        +steps: NDArray
        +ownershipMatrices: NDArray
        +graphs: List[Graph]
        +reset() void
        +get() StateHistoryResult
    }

    class ActionPrintObserver {
        +observationFrequency: int
        +printEmptySteps: bool
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
Agent *-- RandomPolicy : for randomized agents
Agent *-- ReplayPolicy : for replay agents
Environment *-- EnvironmentState : owns
Environment --> Observation : produces
Environment --> ActionSpace : produces
EpisodeStartEvent *-- StateSnapshot : initialState
StepEvent *-- StateSnapshot : pre/post
EpisodeEndEvent *-- StateSnapshot : finalState
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
