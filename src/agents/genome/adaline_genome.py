from __future__ import annotations
import random

import config

from .genome import Genome

class AdalineGenome(Genome):

    NUM_INPUTS = config.OBSERVATION_SIZE ** 2 - 1
    NUM_OUTPUTS = config.NUM_MOVEMENTS

    NUM_WEIGHTS = NUM_INPUTS * NUM_OUTPUTS
    NUM_BIASES = NUM_OUTPUTS

    GENOME_LENGTH = NUM_WEIGHTS + NUM_BIASES

    def __init__(self, genes):
        super().__init__(genes)
        self.genes = self.genes.astype(float)

    def get_gene_values(self) -> list[float]:
        return [-1, -0.5, 0, 0.5, 1]

    @staticmethod
    def random_gene():
        return random.choice([-1, -0.5, 0, 0.5, 1])

    def get_weights(self):
        return self.genes[:self.NUM_WEIGHTS].reshape(self.NUM_INPUTS, self.NUM_OUTPUTS)

    def get_bias(self):
        return self.genes[self.NUM_WEIGHTS:self.NUM_WEIGHTS + self.NUM_OUTPUTS]
