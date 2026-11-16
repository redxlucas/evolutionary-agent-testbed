from typing import Protocol


class Agent(Protocol):

    def reset(self) -> None:
        ...

    def act(
        self,
        observation,
        agent_observation_index: int | None = None,
    ) -> int:
        ...

