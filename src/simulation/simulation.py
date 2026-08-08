from agents import Agent
from environment import FrozenLakeEnvironment

class Simulation:

    def __init__(self, agent: Agent, environment: FrozenLakeEnvironment):
        self.agent = agent
        self.environment = environment

    def run(self):
        state = self.environment.reset()
        done = False

        total_reward = 0

        for _ in range(len(self.agent.genome)):
            if done:
                break

            action = self.agent.act()
            state, reward, done = self.environment.step(action)

            total_reward += reward

        return total_reward