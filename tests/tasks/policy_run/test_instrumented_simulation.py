"""Tests for preparing selected policy inputs with custom observers."""

from __future__ import annotations

import unittest

from network_creation.core.observability import ObservationSampling
from network_creation.observers.state_history_observer import StateHistoryObserver
from tasks.policies import prepare_instrumented_policy_simulation
from tests.tasks.config_support import task_config


class InstrumentedPolicySimulationTest(unittest.TestCase):
    def test_prepares_nonzero_input_with_caller_observer(self) -> None:
        config = task_config(
            n_agents=[2, 3],
            n_actors=1,
            log2_budget=0,
            n_episodes=2,
        )
        observer = StateHistoryObserver(
            observation_sampling=ObservationSampling.periodic(1),
            keep_runs=True,
        )

        prepared = prepare_instrumented_policy_simulation(
            config,
            input_index=1,
            observers=(observer,),
        )

        self.assertEqual(prepared.settings.n_agents, 3)
        self.assertEqual(prepared.settings.n_episodes, 2)
        self.assertIs(prepared.simulator.observers[0], observer)
        self.assertEqual(len(prepared.simulator.agents), 3)

        prepared.simulator.run(
            n_episodes=prepared.settings.n_episodes,
            reset_observers=True,
        )

        result = observer.get()
        self.assertEqual(result.n_episodes, 2)
        self.assertEqual(len(result.runs_graphs or ()), 2)

    def test_rejects_out_of_range_input(self) -> None:
        config = task_config(n_agents=2, log2_budget=0)

        with self.assertRaisesRegex(IndexError, "out of range"):
            prepare_instrumented_policy_simulation(
                config,
                input_index=1,
                observers=(),
            )


if __name__ == "__main__":
    unittest.main()
