"""Simulation loop for actor-scheduled action selection."""

from __future__ import annotations

import numpy as np

from .agent import Agent
from .environment import Environment
from .graph import get_action_space
from .model import Action, ActionType, ActorScheduler, Observation, PolicyDecision
from .observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    SimulationObserver,
    StateView,
    StepEvent,
)


class Simulator:
    """Run one or more episodes over an environment and a set of agents."""

    def __init__(
        self,
        env: Environment,
        agents: tuple[Agent, ...],
        actor_scheduler: ActorScheduler,
        max_steps: int,
        rng: np.random.Generator,
        observers: tuple[SimulationObserver, ...],
        stagnation_patience: int | None = None,
    ) -> None:
        if stagnation_patience is not None and stagnation_patience < 1:
            raise ValueError("stagnation_patience must be positive when provided")

        self.rng = rng
        self.env = env
        self.actor_scheduler = actor_scheduler
        self.max_steps = max_steps
        self.agents = agents
        self.n_agents = len(self.agents)
        self.observers = observers
        self._step_observers_by_step = tuple(
            tuple(
                observer
                for observer in observers
                if observer.observes_step(step, max_steps)
            )
            for step in range(max_steps)
        )
        self.stagnation_patience = stagnation_patience
        self.n_completed_episodes = 0

    def reset(self) -> None:
        self.env.reset()
        for agent in self.agents:
            agent.reset(seed=agent.seed)

    def run(self, n_episodes: int, *, reset_observers: bool) -> None:
        """Run a batch of episodes, optionally clearing observer state first."""

        if reset_observers:
            for observer in self.observers:
                observer.reset()

        for _ in range(n_episodes):
            episode_index = self.n_completed_episodes
            self._run_one_episode(episode_index)
            self.n_completed_episodes += 1

    def _run_one_episode(self, episode_index: int) -> None:
        episode_seed = self._reset_for_episode()
        current_step = 0
        last_modified_step = -1
        stagnation_steps = 0

        episode_start_event = EpisodeStartEvent(
            episode_index=episode_index,
            seed=episode_seed,
            max_steps=self.max_steps,
            n_agents=self.n_agents,
            initial_state=StateView.from_state(self.env.state),
        )
        for observer in self.observers:
            observer.on_episode_start(episode_start_event)

        while current_step < self.max_steps and self.n_agents > 0:
            acting_agents = self.actor_scheduler.select_agents(self.agents)
            decisions: tuple[PolicyDecision, ...]
            if self._is_stagnated(stagnation_steps):
                decisions = ()
            else:
                decision_list: list[PolicyDecision] = []
                for agent in acting_agents:
                    observation = self.env.get_observation(agent)
                    action_space = get_action_space(observation)
                    agent_choose_decision = agent.act(observation, action_space)
                    decision_list.append(
                        self._decision_to_global(observation, agent_choose_decision)
                    )
                decisions = tuple(decision_list)

            step_actions: tuple[Action, ...] = tuple(
                action for decision in decisions for action in decision.actions
            )

            step_observers = self._step_observers_by_step[current_step]
            if step_observers:
                pre_state = StateView.from_state(self.env.state)
                self.env.step(step_actions)
                post_state = StateView.from_state(self.env.state)
            else:
                self.env.step(step_actions)

            modified = any(
                action.type is not ActionType.PASS for action in step_actions
            )
            if modified:
                last_modified_step = current_step
                stagnation_steps = 0
            else:
                stagnation_steps += 1

            if step_observers:
                step_event = StepEvent(
                    step=current_step,
                    decisions=decisions,
                    modified=modified,
                    pre_state=pre_state,
                    post_state=post_state,
                )
                for observer in step_observers:
                    observer.on_step_end(step_event)

            current_step += 1

        episode_end_event = EpisodeEndEvent(
            episode_index=episode_index,
            seed=episode_seed,
            current_step=current_step,
            last_modified_step=last_modified_step,
            final_state=StateView.from_state(self.env.state),
        )
        for observer in self.observers:
            observer.on_episode_end(episode_end_event)

    def _is_stagnated(self, stagnation_steps: int) -> bool:
        return (
            self.stagnation_patience is not None
            and stagnation_steps >= self.stagnation_patience
        )

    @staticmethod
    def _decision_to_global(
        observation: Observation,
        decision: PolicyDecision,
    ) -> PolicyDecision:
        return PolicyDecision(
            actions=tuple(
                Action(
                    action.type,
                    observation.global_from_local(action.source),
                    None
                    if action.target is None
                    else observation.global_from_local(action.target),
                )
                for action in decision.actions
            ),
            value=decision.value,
        )

    def _reset_for_episode(self) -> int:
        episode_seed = self._next_seed()
        self.env.reset()
        for agent in self.agents:
            agent_seed = self._next_seed()
            agent.reset(seed=agent_seed)
        return episode_seed

    def _next_seed(self) -> int:
        return int(self.rng.bit_generator.random_raw())
