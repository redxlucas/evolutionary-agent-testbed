import random

import numpy as np

from agents.genome import MovementGenome
import config


class Crossover:
    def __init__(self, crossover_rate: float):
        self.crossover_rate = crossover_rate

    def crossover(self, parent_a_genome: MovementGenome, parent_b_genome: MovementGenome) -> tuple[MovementGenome, MovementGenome]:

        if random.random() > self.crossover_rate:
            return parent_a_genome.copy(), parent_b_genome.copy()

        crossover_point = random.randint(1, config.GENOME_LENGTH - 1)

        child_a_genes = np.concatenate([parent_a_genome.genes[:crossover_point], parent_b_genome.genes[crossover_point:]])
        child_b_genes = np.concatenate([parent_a_genome.genes[crossover_point:], parent_b_genome.genes[:crossover_point]])

        child_a = MovementGenome(child_a_genes)
        child_b = MovementGenome(child_b_genes)

        return child_a, child_b



