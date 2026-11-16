from __future__ import annotations
import random

import config

from .genome import Genome

class AdalineGenome(Genome):

    NUM_INPUTS = config.OBSERVATION_SIZE ** 2 - 1
    DEFAULT_NUM_OUTPUTS = 4

    def __init__(self, genes, num_outputs: int = DEFAULT_NUM_OUTPUTS):
        self.num_inputs = self.NUM_INPUTS
        self.num_outputs = num_outputs
        self.num_weights = self.num_inputs * self.num_outputs
        super().__init__(genes)
        self.genes = self.genes.astype(float)

    @classmethod
    def random(cls, length=None, num_outputs: int = DEFAULT_NUM_OUTPUTS):
        length = length or cls.genome_length_for(num_outputs)
        genes = [cls.random_gene() for _ in range(length)]
        return cls(genes, num_outputs=num_outputs)

    @classmethod
    def genome_length_for(cls, num_outputs: int) -> int:
        return cls.NUM_INPUTS * num_outputs + num_outputs

    def get_gene_values(self) -> list[float]:
        return [-1, -0.5, 0, 0.5, 1]

    @staticmethod
    def random_gene():
        return random.choice([-1, -0.5, 0, 0.5, 1])

    def get_weights(self):
        return self.genes[:self.num_weights].reshape(
            self.num_inputs,
            self.num_outputs,
        )

    def get_bias(self):
        return self.genes[self.num_weights:self.num_weights + self.num_outputs]

    def copy(self):
        return AdalineGenome(
            self.genes.copy(),
            num_outputs=self.num_outputs,
        )
