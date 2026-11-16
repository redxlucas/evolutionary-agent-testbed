import random

from agents.genome import Genome


class Mutation:

    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate

    def mutate(self, genome: Genome) -> Genome:
        gene_bounds = genome.get_gene_bounds()
        possible_genes = (
            None
            if gene_bounds is not None
            else genome.get_gene_values()
        )

        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                if gene_bounds is not None:
                    genome.genes[i] = random.uniform(*gene_bounds)
                else:
                    current_gene = genome.genes[i]
                    alternatives = [
                        gene
                        for gene in possible_genes
                        if gene != current_gene
                    ]
                    genome.genes[i] = random.choice(alternatives)

        return genome


