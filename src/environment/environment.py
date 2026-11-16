from abc import ABC, abstractmethod


class Environment(ABC):
    """Common interface for environments used by simulations."""

    @abstractmethod
    def reset(self):
        """Reset the environment and return its initial state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """Apply an action and return state, reward, and episode flags."""
        raise NotImplementedError

    @abstractmethod
    def get_agent_position(self) -> tuple[int, int]:
        """Return the agent's current grid position."""
        raise NotImplementedError

    @abstractmethod
    def get_goal_position(self) -> tuple[int, int]:
        """Return the goal's grid position."""
        raise NotImplementedError

    @abstractmethod
    def get_local_observation(self):
        """Return the observation representation consumed by an agent."""
        raise NotImplementedError

    @abstractmethod
    def get_agent_observation_index(self) -> int:
        """Return the flattened cell index occupied by the agent in its view."""
        raise NotImplementedError

    @abstractmethod
    def sample_action(self):
        """Sample a valid action from the environment's action space."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Release resources owned by the environment."""
        raise NotImplementedError
