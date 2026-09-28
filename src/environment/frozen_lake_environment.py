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

        self._prepare_grid()

    def _prepare_grid(self):
        desc = self.env.unwrapped.desc

        goal_positions = np.argwhere(desc == b"G")

        if len(goal_positions) == 0:
            raise ValueError("Goal position not found.")

        self._goal_position = tuple(goal_positions[0])

        grid = np.zeros(desc.shape, dtype=np.float32)

        grid[desc == b"H"] = -2
        grid[desc == b"G"] = 1

        pad_width = config.BORDER_WIDTH

        self._padded_grid = np.pad(
            grid,
            pad_width=config.BORDER_WIDTH,
            constant_values=-1
        )

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
    
    def get_goal_position(self):
        return self._goal_position

    def get_local_observation(self):
        row, col = self.get_agent_position()

        pad_width = config.BORDER_WIDTH

        padded_row = row + pad_width
        padded_col = col + pad_width

        return self._padded_grid[
            padded_row - 1:padded_row + 2,
            padded_col - 1:padded_col + 2
        ]

    def sample_action(self):
        return self.env.action_space.sample()

    def close(self):
        self.env.close()
