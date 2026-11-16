from __future__ import annotations

import random
from .genome import Genome

class MovementGenome(Genome):
    DEFAULT_ACTION_COUNT = 4

    def __init__(
        self,
        genes,
        action_count: int = DEFAULT_ACTION_COUNT,
    ):
        super().__init__(genes)
        self.genes = self.genes.astype(int)
        self.action_count = action_count

    def get_gene_values(self) -> list[int]:
        return list(range(self.action_count))

    @staticmethod
    def random_gene(action_count: int = DEFAULT_ACTION_COUNT):
        return random.randrange(action_count)

    @classmethod
    def random(
        cls,
        length: int,
        action_count: int = DEFAULT_ACTION_COUNT,
    ):
        genes = [
            cls.random_gene(action_count)
            for _ in range(length)
        ]
        return cls(genes, action_count=action_count)

    def copy(self) -> "MovementGenome":
        return MovementGenome(
            self.genes.copy(),
            action_count=self.action_count,
        )
