import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import config

class FrozenLakeEnvironment:
    """
    Classe responsável por encapsular a interação com o ambiente
    FrozenLake disponibilizado pelo Gymnasium.
    """

    def __init__(self, desc=None, is_slippery=False):
        self.env = gym.make(
            "FrozenLake-v1",
            render_mode=config.RENDER_MODE,
            is_slippery=is_slippery,
            desc=desc,
            map_name=config.DEFAULT_MAP_SIZE
        )
        self.state = None
        self.size = int(config.DEFAULT_MAP_SIZE.split("x")[0])

    def reset(self):
        observation, _ = self.env.reset()
        
        self.state = observation

        return observation

    def step(self, action):
        observation, reward, terminated, truncated, _ = self.env.step(action)

        self.state = observation

        return observation, reward, terminated, truncated
    
    def get_agent_position(self) -> tuple[int, int]:
        row = self.state // self.size
        col = self.state % self.size

        return row, col
    
    def get_goal_position(self) -> tuple[int, int]:
        positions = np.argwhere(self.env.unwrapped.desc == b"G")

        if len(positions) == 0:
            raise ValueError("Goal position not found.")
        
        return tuple(positions[0])

    def sample_action(self):
        return self.env.action_space.sample()

    def close(self):
        self.env.close()
