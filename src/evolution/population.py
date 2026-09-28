from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Population(Generic[T]):

    def __init__(self, individuals: list[T]):
        self.individuals = individuals

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)

    def __getitem__(self, index: int) -> T:
        return self.individuals[index]

    def get_best(self) -> T:
        return max(
            self.individuals,
            key=lambda individual: individual.fitness
        )

    @classmethod
    def random(
        cls,
        size: int,
        genome_factory: Callable[[], T]
    ) -> "Population[T]":
        individuals = [
            genome_factory()
            for _ in range(size)
        ]

        return cls(individuals)