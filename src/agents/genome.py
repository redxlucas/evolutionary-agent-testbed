from __future__ import annotations

import random
import numpy as np
import config

class Genome:

    def __init__(self, genes):
        self.genes = np.array(genes, dtype=int)
        self.fitness = None
        self.result = None

    def __len__(self) -> int:
        return len(self.genes)

    def get_gene(self, index: int) -> int:
        return int(self.genes[index])

    def copy(self):
        return Genome(self.genes.copy())

    @classmethod
    def random(cls, length: int) -> Genome:
        genes = [
            random.randrange(config.NUM_MOVEMENTS)
            for _ in range(length)
        ]

        return cls(genes)