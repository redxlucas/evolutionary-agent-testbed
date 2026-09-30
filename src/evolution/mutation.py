from agents.genome import MovementGenome
import random
import config


class Mutation:

    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate

    def mutate(self, genome: MovementGenome) -> MovementGenome:
        possible_genes = genome.get_gene_values()

        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                current_gene = genome.genes[i]

                alternatives = [
                    gene
                    for gene in possible_genes
                    if gene != current_gene
                ]

                genome.genes[i] = random.choice(alternatives)

        return genome


