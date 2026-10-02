from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

class Genome(ABC):
    """
    Base class representing a genome used by the genetic algorithm.
    """
    def __init__(self, genes):
        self.genes = np.array(genes)
        self.fitness = None
        self.result = None

    def __len__(self) -> int:
        return len(self.genes)

    def __getitem__(self, index: int):
        return self.genes[index]

    def __iter__(self):
        return iter(self.genes)

    def copy(self) -> Genome:
        return self.__class__(self.genes.copy())

    @abstractmethod
    def get_gene_values(self):
        pass

    @classmethod
    def random(cls, length: int):
        """
        Creates a genome with randomly generated genes.
        """
        genes = [cls.random_gene() for _ in range(length)]
        return cls(genes)

    @staticmethod
    @abstractmethod
    def random_gene():
        pass