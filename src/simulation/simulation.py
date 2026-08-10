from agents import Agent
from environment import FrozenLakeEnvironment
from .simulation_result import SimulationResult

class Simulation:

    def __init__(self, agent: Agent, environment: FrozenLakeEnvironment):
        self.agent = agent
        self.environment = environment

    def run(self):
        state = self.environment.reset()
        self.agent.reset(state)

        done = False
        total_reward = 0
        steps = 0

        for _ in range(len(self.agent.genome)):
            if done:
                break

            action = self.agent.act()
            state, reward, terminated, truncated = self.environment.step(action)

            done = terminated or truncated

            total_reward += reward
            steps += 1

        final_position = self.environment.get_agent_position()
        goal_position = self.environment.get_goal_position()

        return SimulationResult(
            total_reward=total_reward,
            steps=steps,
            terminated=terminated,
            truncated=truncated,
            final_position=final_position,
            goal_position=goal_position # faz sentido a simulação saber a posição do objetivo?? 
        )