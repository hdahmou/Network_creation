"""Tests for Choi 2020 replay loading and replay policies."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from network_creation.core.environment import Environment
from network_creation.core.model import (
    Action,
    ActionSpace,
    ActionType,
    Observation,
    SchedulingMode,
)
from network_creation.data.choi_2020 import (
    Choi2020NormalizedData,
    Choi2020NormalizedRow,
    Choi2020ReplayConfig,
    build_replay_simulator,
    list_choi_2020_configs,
    load_choi_2020_normalized_data,
    load_choi_2020_replay_data,
)
from network_creation.policies.replay import ReplayPolicy


class ReplayTest(unittest.TestCase):
    @staticmethod
    def _write_csv(
        directory: Path,
        rows: list[dict[str, str]],
        scenario_name: str = "Linking_3_1",
    ) -> Path:
        scenario_dir = directory / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        csv_path = scenario_dir / "data.csv"
        fieldnames = [
            "id",
            "id_subj",
            "time",
            "session",
            "id_grp",
            "grp",
            "round",
            "period",
            "effort",
            "links",
        ]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    @staticmethod
    def _dummy_observation() -> Observation:
        env = Environment(np.zeros((1, 1), dtype=np.int8), seed=7)
        return Observation(
            node_id=0,
            graph=env.state.graph,
            outside_node_ids=np.array([], dtype=np.int_),
            paid_node_ids=np.array([], dtype=np.int_),
        )

    @staticmethod
    def _dummy_action_space() -> ActionSpace:
        return ActionSpace(
            actions={
                ActionType.ADD: [],
                ActionType.BRIDGE: [],
                ActionType.DELETE: [],
                ActionType.PASS: [Action(ActionType.PASS, source=0)],
            }
        )

    def test_list_choi_2020_configs_skips_first_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = self._write_csv(
                Path(tmp_dir),
                rows=[
                    {
                        "id": "1",
                        "id_subj": "1000",
                        "time": "1",
                        "session": "3",
                        "id_grp": "1",
                        "grp": "1",
                        "round": "1",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[]",
                    },
                    {
                        "id": "2",
                        "id_subj": "1001",
                        "time": "2",
                        "session": "3",
                        "id_grp": "2",
                        "grp": "2",
                        "round": "1",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[]",
                    },
                    {
                        "id": "3",
                        "id_subj": "1002",
                        "time": "3",
                        "session": "3",
                        "id_grp": "1",
                        "grp": "1",
                        "round": "2",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[]",
                    },
                ],
            )

            configs = list_choi_2020_configs(Path(tmp_dir))

        self.assertEqual(
            configs,
            (
                Choi2020ReplayConfig(
                    group_size=3,
                    session_id=3,
                    group_id=1,
                    round_id=2,
                    source_path=csv_path,
                ),
            ),
        )

    def test_load_choi_2020_normalized_data_filters_and_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = self._write_csv(
                Path(tmp_dir),
                rows=[
                    {
                        "id": "1",
                        "id_subj": "1000",
                        "time": "1",
                        "session": "3",
                        "id_grp": "1",
                        "grp": "1",
                        "round": "2",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[2]",
                    },
                    {
                        "id": "2",
                        "id_subj": "1001",
                        "time": "2",
                        "session": "3",
                        "id_grp": "1",
                        "grp": "2",
                        "round": "2",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[3]",
                    },
                    {
                        "id": "3",
                        "id_subj": "1002",
                        "time": "3",
                        "session": "3",
                        "id_grp": "1",
                        "grp": "1",
                        "round": "2",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[3]",
                    },
                ],
            )
            config = Choi2020ReplayConfig(
                group_size=3,
                session_id=3,
                group_id=1,
                round_id=2,
                source_path=csv_path,
            )

            normalized_data = load_choi_2020_normalized_data(config)

        self.assertEqual(
            normalized_data,
            Choi2020NormalizedData(
                config=config,
                n_agents=3,
                max_step=3,
                rows=(
                    Choi2020NormalizedRow(
                        node_id=0,
                        step_id=1,
                        target_node_ids=(1,),
                    ),
                    Choi2020NormalizedRow(
                        node_id=0,
                        step_id=3,
                        target_node_ids=(2,),
                    ),
                ),
            ),
        )

    def test_load_choi_2020_replay_data_builds_action_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = self._write_csv(
                Path(tmp_dir),
                rows=[
                    {
                        "id": "1",
                        "id_subj": "1000",
                        "time": "1",
                        "session": "1",
                        "id_grp": "1",
                        "grp": "1",
                        "round": "2",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[2]",
                    },
                    {
                        "id": "2",
                        "id_subj": "1001",
                        "time": "2",
                        "session": "1",
                        "id_grp": "1",
                        "grp": "1",
                        "round": "2",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[2, 3]",
                    },
                    {
                        "id": "3",
                        "id_subj": "1002",
                        "time": "2",
                        "session": "1",
                        "id_grp": "2",
                        "grp": "1",
                        "round": "2",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[1]",
                    },
                    {
                        "id": "4",
                        "id_subj": "1003",
                        "time": "3",
                        "session": "1",
                        "id_grp": "1",
                        "grp": "1",
                        "round": "2",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[3]",
                    },
                ],
            )
            config = Choi2020ReplayConfig(
                group_size=3,
                session_id=1,
                group_id=1,
                round_id=2,
                source_path=csv_path,
            )

            replay_data = load_choi_2020_replay_data(config)

        self.assertEqual(replay_data.n_agents, 3)
        self.assertEqual(replay_data.max_step, 3)
        self.assertEqual(
            replay_data.trajectories[0].actions_by_step,
            {
                1: (Action(ActionType.ADD, source=0, target=1),),
                2: (Action(ActionType.ADD, source=0, target=2),),
                3: (Action(ActionType.DELETE, source=0, target=1),),
            },
        )
        self.assertEqual(
            replay_data.trajectories[1].actions_by_step,
            {
                2: (Action(ActionType.ADD, source=1, target=0),),
            },
        )
        self.assertEqual(replay_data.trajectories[2].actions_by_step, {})

    def test_load_choi_2020_replay_data_keeps_multiple_actions_at_same_step(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = self._write_csv(
                Path(tmp_dir),
                rows=[
                    {
                        "id": "1",
                        "id_subj": "1000",
                        "time": "4",
                        "session": "1",
                        "id_grp": "1",
                        "grp": "1",
                        "round": "2",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[]",
                    },
                    {
                        "id": "2",
                        "id_subj": "1000",
                        "time": "5",
                        "session": "1",
                        "id_grp": "1",
                        "grp": "1",
                        "round": "2",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[2, 3]",
                    },
                ],
            )
            config = Choi2020ReplayConfig(
                group_size=3,
                session_id=1,
                group_id=1,
                round_id=2,
                source_path=csv_path,
            )

            replay_data = load_choi_2020_replay_data(config)

        self.assertEqual(
            replay_data.trajectories[0].actions_by_step[5],
            (
                Action(ActionType.ADD, source=0, target=1),
                Action(ActionType.ADD, source=0, target=2),
            ),
        )

    def test_load_choi_2020_replay_data_preserves_raw_step_ids_with_empty_start(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = self._write_csv(
                Path(tmp_dir),
                rows=[
                    {
                        "id": "1",
                        "id_subj": "1000",
                        "time": "6",
                        "session": "1",
                        "id_grp": "1",
                        "grp": "1",
                        "round": "2",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[2]",
                    },
                    {
                        "id": "2",
                        "id_subj": "1000",
                        "time": "10",
                        "session": "1",
                        "id_grp": "1",
                        "grp": "1",
                        "round": "2",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[2, 3]",
                    },
                ],
            )
            config = Choi2020ReplayConfig(
                group_size=3,
                session_id=1,
                group_id=1,
                round_id=2,
                source_path=csv_path,
            )

            replay_data = load_choi_2020_replay_data(config)

        self.assertEqual(replay_data.max_step, 10)
        self.assertEqual(
            replay_data.trajectories[0].actions_by_step,
            {
                6: (Action(ActionType.ADD, source=0, target=1),),
                10: (Action(ActionType.ADD, source=0, target=2),),
            },
        )

    def test_replay_policy_returns_recorded_actions_step_by_step(self) -> None:
        policy = ReplayPolicy(
            {
                1: (Action(ActionType.ADD, source=0, target=1),),
                3: (
                    Action(ActionType.DELETE, source=0, target=1),
                    Action(ActionType.ADD, source=0, target=2),
                ),
            }
        )
        observation = self._dummy_observation()
        action_space = self._dummy_action_space()

        self.assertEqual(
            policy.get_actions(observation, action_space),
            [Action(ActionType.ADD, source=0, target=1)],
        )
        self.assertEqual(policy.get_actions(observation, action_space), [])
        self.assertEqual(
            policy.get_actions(observation, action_space),
            [
                Action(ActionType.DELETE, source=0, target=1),
                Action(ActionType.ADD, source=0, target=2),
            ],
        )

        policy.reset(seed=11)
        self.assertEqual(
            policy.get_actions(observation, action_space),
            [Action(ActionType.ADD, source=0, target=1)],
        )

    def test_build_replay_simulator_replays_final_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = self._write_csv(
                Path(tmp_dir),
                rows=[
                    {
                        "id": "1",
                        "id_subj": "1000",
                        "time": "1",
                        "session": "1",
                        "id_grp": "1",
                        "grp": "1",
                        "round": "2",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[2, 3]",
                    },
                    {
                        "id": "2",
                        "id_subj": "1001",
                        "time": "1",
                        "session": "1",
                        "id_grp": "2",
                        "grp": "1",
                        "round": "2",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[1]",
                    },
                    {
                        "id": "3",
                        "id_subj": "1002",
                        "time": "2",
                        "session": "1",
                        "id_grp": "1",
                        "grp": "1",
                        "round": "2",
                        "period": "1",
                        "effort": "NULL",
                        "links": "[3]",
                    },
                ],
            )
            config = Choi2020ReplayConfig(
                group_size=3,
                session_id=1,
                group_id=1,
                round_id=2,
                source_path=csv_path,
            )

            simulator = build_replay_simulator(
                config=config,
                observers=(),
            )
            simulator.run(n_episodes=1)

        self.assertEqual(simulator.mode, SchedulingMode.SIMULTANEOUS)
        self.assertEqual(simulator.max_steps, 2)
        np.testing.assert_array_equal(
            simulator.env.state.ownership_matrix,
            np.array(
                [
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 0, 0],
                ],
                dtype=np.int8,
            ),
        )


if __name__ == "__main__":
    unittest.main()
