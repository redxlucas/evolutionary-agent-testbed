from __future__ import annotations

import numpy as np

from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv

import config
from .environment import Environment


class _FrozenLakeMapEnv(MiniGridEnv):
    MAP = (
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    )

    def __init__(
        self,
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        agent_view_size: int = config.OBSERVATION_SIZE,
        render_mode: str | None = config.RENDER_MODE,
    ):
        if max_steps is None:
            max_steps = config.MAX_STEPS
        self.agent_start_dir = agent_start_dir
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
        )
        super().__init__(
            mission_space=mission_space,
            grid_size=len(self.MAP) + 2,
            see_through_walls=False,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            render_mode=render_mode,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "reach the goal without falling into a hole"

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        for row, cells in enumerate(self.MAP):
            for column, cell in enumerate(cells):
                position = (column + 1, row + 1)

                if cell == "H":
                    self.grid.set(*position, Lava())
                elif cell == "G":
                    self.grid.set(*position, Goal())

        self.agent_pos = (1, 1)
        self.agent_dir = self.agent_start_dir
        self.mission = self._gen_mission()

class MiniGridEnvironment(Environment):
    """MiniGrid adapter for the project's 4x4 Frozen Lake layout."""

    # The movement genome uses only navigation actions; interaction actions
    # (pickup, drop, toggle, done) are not relevant to this map.
    ACTION_COUNT = 3
    OBJECT_VALUES = {
        "unseen": config.UNSEEN_CELL_VALUE,
        "empty": config.EMPTY_CELL_VALUE,
        "floor": config.EMPTY_CELL_VALUE,
        "wall": config.BORDER_CELL_VALUE,
        "lava": config.HOLE_CELL_VALUE,
        "goal": config.GOAL_CELL_VALUE,
    }

    def __init__(
        self,
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        agent_view_size: int = config.OBSERVATION_SIZE,
        render_mode: str | None = config.RENDER_MODE,
    ):
        self.env = _FrozenLakeMapEnv(
            agent_start_dir=agent_start_dir,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            render_mode=render_mode,
        )
        self._observation = None
        self._goal_position = self._find_goal_position()
        self._object_value_lookup = np.full(
            max(OBJECT_TO_IDX.values()) + 1,
            config.EMPTY_CELL_VALUE,
            dtype=np.float32,
        )
        for object_name, value in self.OBJECT_VALUES.items():
            self._object_value_lookup[OBJECT_TO_IDX[object_name]] = value

    @staticmethod
    def _find_goal_position() -> tuple[int, int]:
        for row, cells in enumerate(_FrozenLakeMapEnv.MAP):
            for column, cell in enumerate(cells):
                if cell == "G":
                    return row, column
        raise ValueError("Goal position not found in MiniGrid map.")

    def reset(self, *, seed: int | None = None, options=None):
        self._observation, info = self.env.reset(seed=seed, options=options)
        return self._observation, info

    def step(self, action: int):
        (
            self._observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        return self._observation, reward, terminated, truncated

    def get_agent_position(self) -> tuple[int, int]:
        x, y = self.env.unwrapped.agent_pos
        return int(y) - 1, int(x) - 1

    def get_goal_position(self) -> tuple[int, int]:
        return self._goal_position

    def get_local_observation(self):
        if self._observation is None:
            raise RuntimeError("Call reset() before requesting an observation.")

        encoded_image = self._observation["image"]
        # MiniGrid stores grid coordinates as (x, y, channels); transpose to
        # the conventional (row=y, column=x) matrix used by the agents.
        object_types = encoded_image[:, :, 0].T
        return self._object_value_lookup[object_types]

    def get_agent_observation_index(self) -> int:
        view_size = self.env.agent_view_size
        agent_row = view_size - 1
        agent_column = view_size // 2
        return agent_row * view_size + agent_column

    def sample_action(self) -> int:
        return int(self.env.action_space.sample())

    def close(self):
        self.env.close()
