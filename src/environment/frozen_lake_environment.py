import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
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

    def reset(self):
        observation, _ = self.env.reset()
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, _ = self.env.step(action)

        return observation, reward, terminated, truncated
    
    # def render(self):

    def sample_action(self):
        return self.env.action_space.sample()

    def close(self):
        self.env.close()
