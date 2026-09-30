import random

import numpy as np

from agents.genome import Genome

class Crossover:
    """
    Performs single-point crossover between two genomes.
    """
    def __init__(self, crossover_rate: float):
        self.crossover_rate = crossover_rate

    def crossover(self, parent_a: Genome, parent_b: Genome) -> tuple[Genome, Genome]:
        """
        Generates two offspring from a pair of parent genomes.
        """

        if len(parent_a.genes) != len(parent_b.genes):
            raise ValueError(
                "Parent genomes must have the same length."
            )

        if random.random() > self.crossover_rate:
            return (parent_a.copy(), parent_b.copy())

        crossover_point = random.randint(1, len(parent_a.genes) - 1)

        child_a_genes = np.concatenate([parent_a.genes[:crossover_point], parent_b.genes[crossover_point:]])
        child_b_genes = np.concatenate([parent_a.genes[crossover_point:], parent_b.genes[:crossover_point]])

        child_a = parent_a.__class__(child_a_genes)
        child_b = parent_b.__class__(child_b_genes)

        return child_a, child_b



