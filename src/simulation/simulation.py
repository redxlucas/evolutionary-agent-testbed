import numpy as np

from agents import Adaline
import config
from environment import FrozenLakeEnvironment
from .simulation_result import SimulationResult

class Simulation:

    def __init__(self, adaline: Adaline, environment: FrozenLakeEnvironment):
        self.adaline = adaline
        self.environment = environment

    def run(self):
        self.environment.reset()

        done = False
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        
        for _ in range(config.MAX_STEPS):
            if done:
                break

            observation = self.environment.get_local_observation()

            inputs = np.delete(observation.flatten(), 4)

            action = self.adaline.act(inputs)

            _, reward, terminated, truncated = self.environment.step(action)

            done = terminated or truncated

            total_reward += reward
            steps += 1

        return SimulationResult(
            total_reward=total_reward,
            steps=steps,
            terminated=terminated,
            truncated=truncated,
            final_position=self.environment.get_agent_position(),
            goal_position=self.environment.get_goal_position()
        )

    def run_debug(self):
        self.environment.reset()

        done = False

        action_names = {
            0: "LEFT",
            1: "DOWN",
            2: "RIGHT",
            3: "UP"
        }

        for step in range(config.MAX_STEPS):
            if done:
                break

            position = self.environment.get_agent_position()

            observation = self.environment.get_local_observation()
            inputs = np.delete(observation.flatten(), 4)

            outputs = self.adaline.predict(inputs)
            action = np.argmax(outputs)

            print(f"\nStep: {step}")
            print(f"Position: {position}")

            print("Observation:")
            print(observation)

            print(f"Inputs: {inputs}")
            print(f"Outputs: {outputs}")
            print(f"Action: {action} ({action_names[action]})")

            _, reward, terminated, truncated = self.environment.step(action)

            new_position = self.environment.get_agent_position()

            print(f"New position: {new_position}")
            print(f"Reward: {reward}")

            done = terminated or truncated