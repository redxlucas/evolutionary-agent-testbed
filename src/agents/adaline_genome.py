from __future__ import annotations
import random

import numpy as np

import config

class AdalineGenome:

    def __init__(self, genes):
        self.genes = np.array(genes, dtype=float)
        self.fitness = None
        self.result = None

    def __len__(self) -> int:
        return len(self.genes)

    def get_gene(self, index):
        return self.genes[index]

    def copy(self):
        return AdalineGenome(self.genes.copy())

    def get_weights(self):
        return self.genes[:32].reshape(8, 4)

    def get_bias(self):
        return self.genes[32:36]

    @classmethod
    def random(cls, length: int) -> AdalineGenome:
        genes = [random.randrange(config.NUM_MOVEMENTS) for _ in range(length)]
        return cls(genes)
