import numpy as np

from agents import Agent
from agents import Genome


class Population:
    
    def __init__(self, genomes: list[Genome]):
        self.genomes = genomes

    def __len__(self):
        return len(self.genomes)

    def __iter__(self):
        return iter(self.genomes)

    @classmethod
    def random(cls, size: int, genome_length: int):
        genomes = [Genome(np.random.uniform(-1, 1, genome_length)) for _ in range(size)]

        return cls(genomes)