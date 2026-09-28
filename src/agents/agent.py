from typing import Protocol


class Agent(Protocol):

    def reset(self) -> None:
        ...

    def act(self, observation):
        ...

