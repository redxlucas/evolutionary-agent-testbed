from agents import Agent

import config

from environment import FrozenLakeEnvironment

from .simulation_result import SimulationResult


class Simulation:
    """
    Executes an agent in an environment and collects the simulation results.
    """
    def __init__(
        self,
        agent: Agent,
        environment: FrozenLakeEnvironment
    ):
        self.agent = agent
        self.environment = environment

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

        for _ in range(config.MAX_STEPS):
            observation = self.environment.get_local_observation()
            action = self.agent.act(observation)

            _, reward, terminated, truncated = self.environment.step(action)

            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        final_position = self.environment.get_agent_position()
        goal_position = self.environment.get_goal_position()

        return SimulationResult(
            total_reward=total_reward,
            steps=steps,
            terminated=terminated,
            truncated=truncated,
            final_position=final_position,
            goal_position=goal_position,
            success = final_position == goal_position
        )