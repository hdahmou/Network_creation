"""Load and transform Choi 2020 linking-game data into replay trajectories."""

from __future__ import annotations

from ast import literal_eval
from collections.abc import Iterable
from collections import defaultdict
from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.graph import create_empty_graph
from network_creation.core.model import Action, ActionType, ActorScheduler
from network_creation.core.observability import SimulationObserver
from network_creation.core.simulator import Simulator
from network_creation.policies.replay import ReplayPolicy


@dataclass(frozen=True, order=True)
class Choi2020ReplayConfig:
    """One available replay configuration in the Choi 2020 dataset."""

    group_size: int
    session_id: int
    group_id: int
    round_id: int
    source_path: Path


@dataclass(frozen=True)
class Choi2020NormalizedRow:
    """One normalized row from a selected Choi 2020 game slice."""

    node_id: int
    step_id: int
    target_node_ids: tuple[int, ...]


@dataclass(frozen=True)
class Choi2020NormalizedData:
    """Normalized Choi 2020 data for one selected game slice."""

    config: Choi2020ReplayConfig
    n_agents: int
    max_step: int
    rows: tuple[Choi2020NormalizedRow, ...]


@dataclass(frozen=True)
class ReplayTrajectory:
    """Recorded actions for one node indexed by replay step."""

    node_id: int
    actions_by_step: dict[int, tuple[Action, ...]]


@dataclass(frozen=True)
class ReplayDataset:
    """Processed replay data for one experimental CSV file."""

    source_path: Path
    n_agents: int
    max_step: int
    trajectories: dict[int, ReplayTrajectory]


def list_choi_2020_configs(
    data_root: str | Path,
) -> tuple[Choi2020ReplayConfig, ...]:
    root = Path(data_root)
    configs: set[Choi2020ReplayConfig] = set()

    for csv_path in sorted(root.glob("Linking_*_*/data.csv")):
        group_size = _get_group_size(csv_path)
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                round_id = int(row["round"])
                if round_id == 1:
                    continue

                configs.add(
                    Choi2020ReplayConfig(
                        group_size=group_size,
                        session_id=int(row["session"]),
                        group_id=int(row["grp"]),
                        round_id=round_id,
                        source_path=csv_path,
                    )
                )

    return tuple(sorted(configs))


def load_choi_2020_normalized_data(
    config: Choi2020ReplayConfig,
) -> Choi2020NormalizedData:
    """Load and normalize one Choi 2020 game slice selected by configuration."""

    if config.round_id == 1:
        raise ValueError("round_id=1 is skipped for backward compatibility")

    raw_rows = []
    with config.source_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["session"]) != config.session_id:
                continue
            if int(row["grp"]) != config.group_id:
                continue
            if int(row["round"]) != config.round_id:
                continue

            raw_rows.append(
                (
                    int(row["id_grp"]) - 1,
                    int(row["time"]),
                    tuple(
                        int(target_node_id) - 1
                        for target_node_id in literal_eval(row["links"])
                    ),
                )
            )

    if not raw_rows:
        return Choi2020NormalizedData(
            config=config,
            n_agents=config.group_size,
            max_step=0,
            rows=(),
        )

    rows = tuple(
        sorted(
            (
                Choi2020NormalizedRow(
                    node_id=node_id,
                    # Preserve the raw recorded time. Step 0 is the extra empty
                    # initial state introduced by the simulator before replay.
                    step_id=raw_step,
                    target_node_ids=target_node_ids,
                )
                for node_id, raw_step, target_node_ids in raw_rows
            ),
            key=lambda row: (row.node_id, row.step_id),
        )
    )
    max_step = max((row.step_id for row in rows), default=0)
    return Choi2020NormalizedData(
        config=config,
        n_agents=config.group_size,
        max_step=max_step,
        rows=rows,
    )


def load_choi_2020_replay_data(
    config: Choi2020ReplayConfig,
    max_steps: int,
) -> ReplayDataset:
    """Build replay trajectories from one normalized Choi 2020 configuration."""

    normalized_data = load_choi_2020_normalized_data(config)
    if max_steps < normalized_data.max_step:
        raise ValueError(
            f"max_steps must be at least the raw replay horizon ({normalized_data.max_step})"
        )

    rows_by_node_id = _group_rows_by_node_id(normalized_data.rows)

    trajectories = {
        node_id: ReplayTrajectory(
            node_id=node_id,
            actions_by_step=_build_actions_by_step(
                rows=rows_by_node_id.get(node_id, []),
                source_node_id=node_id,
            ),
        )
        for node_id in range(normalized_data.n_agents)
    }
    if max_steps > normalized_data.max_step:
        trajectories = _pad_tail_pass_actions(
            trajectories=trajectories,
            raw_max_step=normalized_data.max_step,
            max_step=max_steps,
        )

    return ReplayDataset(
        source_path=normalized_data.config.source_path,
        n_agents=normalized_data.n_agents,
        max_step=max_steps,
        trajectories=trajectories,
    )


def build_replay_simulator(
    config: Choi2020ReplayConfig,
    observers: tuple[SimulationObserver, ...],
    max_steps: int,
) -> Simulator:
    """Create a simulator that replays one Choi 2020 game slice."""

    replay_dataset = load_choi_2020_replay_data(config, max_steps=max_steps)
    env = Environment(create_empty_graph(replay_dataset.n_agents), seed=0)
    agents = tuple(
        Agent(
            activation_weight=1.0,
            id=node_id,
            visibility_distance=replay_dataset.n_agents,
            policy=ReplayPolicy(replay_dataset.trajectories[node_id].actions_by_step),
            seed=node_id,
        )
        for node_id in range(replay_dataset.n_agents)
    )
    return Simulator(
        env=env,
        agents=agents,
        actor_scheduler=ActorScheduler(
            n_agents=replay_dataset.n_agents,
            n_actors=replay_dataset.n_agents,
            activation_weights=(1.0,) * replay_dataset.n_agents,
            seed=0,
        ),
        max_steps=replay_dataset.max_step,
        rng=np.random.default_rng(0),
        observers=observers,
        stagnation_patience=None,
    )


def run_choi_2020_replay_batch(
    configs: Iterable[Choi2020ReplayConfig],
    observers: tuple[SimulationObserver, ...],
    max_steps: int,
) -> None:
    """Run one replay episode per config through one shared observer tuple."""

    for config in sorted(configs):
        simulator = build_replay_simulator(
            config=config,
            observers=observers,
            max_steps=max_steps,
        )
        simulator.run(n_episodes=1, reset_observers=False)


def _get_group_size(csv_path: Path) -> int:
    parts = csv_path.parent.name.split("_")
    if len(parts) < 3 or parts[0] != "Linking":
        raise ValueError(f"unsupported Choi 2020 path format: {csv_path}")
    return int(parts[1])


def _group_rows_by_node_id(
    rows: tuple[Choi2020NormalizedRow, ...],
) -> dict[int, list[tuple[int, tuple[int, ...]]]]:
    rows_by_node_id: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    for row in rows:
        rows_by_node_id[row.node_id].append((row.step_id, row.target_node_ids))

    for rows in rows_by_node_id.values():
        rows.sort(key=lambda item: item[0])
    return dict(rows_by_node_id)


def _build_actions_by_step(
    rows: list[tuple[int, tuple[int, ...]]],
    source_node_id: int,
) -> dict[int, tuple[Action, ...]]:
    actions_by_step: dict[int, list[Action]] = defaultdict(list)
    previous_targets: set[int] = set()

    for raw_step, link_targets in rows:
        current_targets = set(link_targets)

        for target_node_id in sorted(current_targets - previous_targets):
            actions_by_step[raw_step].append(
                Action(ActionType.ADD, source=source_node_id, target=target_node_id)
            )

        for target_node_id in sorted(previous_targets - current_targets):
            actions_by_step[raw_step].append(
                Action(
                    ActionType.DELETE,
                    source=source_node_id,
                    target=target_node_id,
                )
            )

        previous_targets = current_targets

    return {step: tuple(actions) for step, actions in sorted(actions_by_step.items())}


def _pad_tail_pass_actions(
    trajectories: dict[int, ReplayTrajectory],
    raw_max_step: int,
    max_step: int,
) -> dict[int, ReplayTrajectory]:
    padded_trajectories: dict[int, ReplayTrajectory] = {}
    for node_id, trajectory in trajectories.items():
        padded_actions_by_step = dict(trajectory.actions_by_step)
        for step_id in range(raw_max_step + 1, max_step + 1):
            padded_actions_by_step[step_id] = (Action(ActionType.PASS, source=node_id),)

        padded_trajectories[node_id] = ReplayTrajectory(
            node_id=node_id,
            actions_by_step=padded_actions_by_step,
        )

    return padded_trajectories
