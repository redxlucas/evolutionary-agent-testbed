import config

from .frozen_lake_environment import FrozenLakeEnvironment
from .minigrid_environment import MiniGridEnvironment


def create_environment(environment_type: str | None = None):
    """Create the environment selected in config or explicitly supplied."""
    environment_type = (
        environment_type or config.ENVIRONMENT_TYPE
    ).strip().lower()

    if environment_type == "frozen_lake":
        return FrozenLakeEnvironment()

    if environment_type == "minigrid":
        return MiniGridEnvironment()

    raise ValueError(
        f"Unknown environment type: {environment_type!r}. "
        "Available environments: frozen_lake, minigrid."
    )


def get_environment_action_count(environment_type: str | None = None) -> int:
    """Return the number of discrete actions for a configured environment."""
    environment_type = (
        environment_type or config.ENVIRONMENT_TYPE
    ).strip().lower()

    action_counts = {
        "frozen_lake": FrozenLakeEnvironment.ACTION_COUNT,
        "minigrid": MiniGridEnvironment.ACTION_COUNT,
    }

    try:
        return action_counts[environment_type]
    except KeyError as error:
        raise ValueError(
            f"Unknown environment type: {environment_type!r}. "
            "Available environments: frozen_lake, minigrid."
        ) from error
