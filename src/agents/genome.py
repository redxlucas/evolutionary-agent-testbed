from __future__ import annotations
import random

import config

class Genome:
    """
    Classe que representa o genoma de um agente evolutivo.
    """

    def __init__(self, genes: list[int]):
        self.genes = genes

    def __len__(self) -> int:
        return len(self.genes)

    def get_gene(self, index):
        return self.genes[index]

    def copy(self):
        return Genome(self.genes.copy())

    @classmethod
    def random(cls, length: int) -> Genome:
        genes = [random.randrange(config.NUM_MOVEMENTS) for _ in range(length)]
        return cls(genes)
