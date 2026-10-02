import logging

from agents import Agent

import config

from environment import FrozenLakeEnvironment

from .simulation_result import SimulationResult

logger = logging.getLogger(__name__)


class Simulation:
    """
    Executes an agent in an environment and collects the simulation results.
    """
    def __init__(
        self,
        agent: Agent,
        environment: FrozenLakeEnvironment,
        trace_context: str = ""
    ):
        self.agent = agent
        self.environment = environment
        self.trace_context = trace_context

    def run(self) -> SimulationResult:
        """
        Runs the simulation and returns the resulting metrics.
        """
        self.environment.reset()
        self.agent.reset()

        terminated = False
        truncated = False
        total_reward = 0
        steps = 0

        for step_index in range(config.MAX_STEPS):
            observation = self.environment.get_local_observation()
            position = self.environment.get_agent_position()
            if hasattr(self.agent, "trace_context"):
                self.agent.trace_context = (
                    f"{self.trace_context} step={step_index}"
                )
            action = self.agent.act(observation)

            next_observation, reward, terminated, truncated = (
                self.environment.step(action)
            )
            next_position = self.environment.get_agent_position()

            logger.debug(
                "Environment transition | %s | step=%d | position=%s | "
                "action=%d | next_position=%s | reward=%s | terminated=%s | "
                "truncated=%s | next_state=%s",
                self.trace_context,
                step_index,
                position,
                action,
                next_position,
                reward,
                terminated,
                truncated,
                next_observation,
            )

            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        final_position = self.environment.get_agent_position()
        goal_position = self.environment.get_goal_position()

        result = SimulationResult(
            total_reward=total_reward,
            steps=steps,
            terminated=terminated,
            truncated=truncated,
            final_position=final_position,
            goal_position=goal_position,
            success = final_position == goal_position
        )

        logger.debug(
            "Episode completed | %s | result=%s",
            self.trace_context,
            result,
        )

        return result
