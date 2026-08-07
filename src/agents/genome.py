from __future__ import annotations
import random

NUM_MOVEMENTS = 4

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

    @classmethod
    def random(cls, length: int) -> Genome:
        genes = [random.randrange(NUM_MOVEMENTS) for _ in range(length)]
        return cls(genes)
