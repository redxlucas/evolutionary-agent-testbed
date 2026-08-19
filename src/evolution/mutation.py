from agents import Genome
import random
import config


class Mutation:

    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate

    def mutate(self, genome: Genome) -> Genome:
        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                possible_genes = [
                    gene
                    for gene in range(config.NUM_MOVEMENTS)
                    if gene != genome.genes[i]
                ]

                genome.genes[i] = random.choice(possible_genes)

        return genome


