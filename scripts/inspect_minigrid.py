
from minigrid.core.actions import Actions
from minigrid.manual_control import ManualControl

from environment import MiniGridEnvironment


def print_observation(env, observation):
    image = observation["image"]
    # MiniGrid arrays use (x, y, channels); print conventional rows first.
    object_types = image[:, :, 0].T
    symbols = {
        0: "?",  # unseen
        1: ".",  # empty
        2: "#",  # wall
        3: ".",  # floor
        8: "G",  # goal
        9: "L",  # lava
        10: "A",  # agent, when encoded by the view
    }
    readable_grid = [
        [symbols.get(int(object_type), "?") for object_type in row]
        for row in object_types
    ]

    # MiniGrid omits the observing agent from the encoded image. Mark its
    # fixed location in the egocentric view for easier visual interpretation.
    agent_row = env.agent_view_size - 1
    agent_column = env.agent_view_size // 2
    readable_grid[agent_row][agent_column] = "A"

    direction_names = ("right", "down", "left", "up")
    direction = int(observation["direction"])
    agent_position = tuple(int(value) for value in env.agent_pos)

    print("\nAgent observation (egocentric view):")
    print(
        f"Position: {agent_position} | "
        f"direction: {direction} ({direction_names[direction]}) | "
        f"mission: {observation['mission']}"
    )
    print("Readable grid:  A=agent  #=wall  L=lava  G=goal  .=empty  ?=unseen")
    for row in readable_grid:
        print(" ".join(row))
    print("Encoded image (object, color, state):")
    print(image.transpose(1, 0, 2))


class ObservationManualControl(ManualControl):
    def reset(self, seed=None):
        observation, _ = self.env.reset(seed=seed)
        self.env.render()
        print_observation(self.env.unwrapped, observation)

    def step(self, action: Actions):
        observation, reward, terminated, truncated, _ = self.env.step(action)
        print(
            f"\nstep={self.env.unwrapped.step_count}, "
            f"action={action.name}, reward={reward:.2f}, "
            f"terminated={terminated}, truncated={truncated}"
        )
        print_observation(self.env.unwrapped, observation)

        if terminated or truncated:
            print(f"Episode finished; resetting. Reward: {reward}")
            self.reset(self.seed)
        else:
            self.env.render()


def main():
    env = MiniGridEnvironment(
        render_mode="human",
        agent_view_size=3,
    )

    try:
        ObservationManualControl(env.env, seed=42).start()
    finally:
        env.close()


if __name__ == "__main__":
    main()
