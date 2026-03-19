"""Simulation loop for sequential or simultaneous action scheduling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .agent import Agent
from .environment import Environment
from .graph import get_action_space
from .model import SchedulingMode
from .observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    SimulationObserver,
    StateSnapshot,
    StepEvent,
)


class Simulator:
    """Drive one episode over an environment and a set of agents."""

    def __init__(
        self,
        env: Environment,
        agents: Mapping[int, Agent],
        mode: SchedulingMode,
        max_steps: int,
        seed: int,
        observers: Sequence[SimulationObserver] = (),
    ) -> None:
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")

        self.seed = seed
        self.env = env
        self.mode = mode
        self.max_steps = max_steps
        self.current_step = 0
        self.last_modified_step = -1

        self.agents = dict(agents)
        for agent_id, agent in self.agents.items():
            if agent.id != agent_id:
                raise ValueError("agent mapping keys must match agent ids")
        self.n_agents = len(self.agents)
        self._ordered_agent_ids = sorted(self.agents)
        self.observers = tuple(observers)

        if self._ordered_agent_ids != self.env.node_ids:
            raise ValueError("agent ids must match environment node ids")

    def reset(self) -> None:
        """Reset the environment, all agents, and simulator counters."""

        self.env.reset()
        for agent_id in self._ordered_agent_ids:
            self.agents[agent_id].reset()
        self.current_step = 0
        self.last_modified_step = -1

    def get_acting_agents(self) -> list[Agent]:
        """Return the agents scheduled to act at the current step."""

        if self.n_agents == 0:
            return []

        if self.mode is SchedulingMode.SEQUENTIAL:
            agent_id = self._ordered_agent_ids[self.current_step % self.n_agents]
            return [self.agents[agent_id]]

        return [self.agents[agent_id] for agent_id in self._ordered_agent_ids]

    def run_episode(self) -> None:
        """Run steps until the configured step budget is exhausted."""

        self.reset()

        episode_start_event = EpisodeStartEvent(
            seed=self.seed,
            mode=self.mode,
            max_steps=self.max_steps,
            n_agents=self.n_agents,
            initial_state=StateSnapshot.from_state(self.env.state),
        )
        for observer in self.observers:
            observer.on_episode_start(episode_start_event)

        while self.current_step < self.max_steps and self.n_agents > 0:
            acting_agents = self.get_acting_agents()
            observations = {}
            action_spaces = {}

            for agent in acting_agents:
                observation = self.env.get_observation(agent)
                observations[agent.id] = observation
                action_spaces[agent.id] = get_action_space(observation)

            actions = [
                action
                for agent in acting_agents
                for action in agent.act(observations[agent.id], action_spaces[agent.id])
            ]

            pre_state = StateSnapshot.from_state(self.env.state)
            self.env.step(actions)
            post_state = StateSnapshot.from_state(self.env.state)

            modified = not np.array_equal(
                pre_state.ownership_matrix,
                post_state.ownership_matrix,
            )
            if modified:
                self.last_modified_step = self.current_step

            step_event = StepEvent(
                step=self.current_step,
                mode=self.mode,
                acting_agent_ids=tuple(agent.id for agent in acting_agents),
                actions=tuple(actions),
                modified=modified,
                pre_state=pre_state,
                post_state=post_state,
            )
            for observer in self.observers:
                if (self.current_step + 1) % observer.observation_frequency == 0:
                    observer.on_step_end(step_event)

            self.current_step += 1

        episode_end_event = EpisodeEndEvent(
            seed=self.seed,
            mode=self.mode,
            current_step=self.current_step,
            last_modified_step=self.last_modified_step,
            final_state=StateSnapshot.from_state(self.env.state),
        )
        for observer in self.observers:
            observer.on_episode_end(episode_end_event)
