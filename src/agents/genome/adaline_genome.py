from __future__ import annotations
import random

import config

from .genome import Genome

class AdalineGenome(Genome):

    NUM_INPUTS = config.OBSERVATION_SIZE ** 2 - 1
    NUM_OUTPUTS = config.NUM_MOVEMENTS
    NUM_WEIGHTS = NUM_INPUTS * NUM_OUTPUTS

    def __init__(self, genes):
        super().__init__(genes)
        self.genes = self.genes.astype(float)

    def get_gene_values(self) -> list[int]:
        return [-1, 0, 1]

    @staticmethod
    def random_gene():
        return random.uniform(-1, 1)

    def get_weights(self):
        return self.genes[:self.NUM_WEIGHTS].reshape(self.NUM_INPUTS, self.NUM_OUTPUTS)

    def get_bias(self):
        return self.genes[self.NUM_WEIGHTS:self.NUM_WEIGHTS + self.NUM_OUTPUTS]
