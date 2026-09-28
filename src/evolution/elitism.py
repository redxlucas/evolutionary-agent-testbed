from evolution import Population

class Elitism:

    def __init__(self, elite_size: int):
        if elite_size < 0:
            raise ValueError(
                "Elite size cannot be negative."
            )
        
        self.elite_size = elite_size

    def select_elites(self, population: Population) -> Population:
         
        if self.elite_size > len(population):
            raise ValueError(
                "Elite size cannot be greater than population size."
            )
         
        selected_genomes = sorted(
            population.individuals,
            key=lambda genome: genome.fitness,
            reverse=True
        )[:self.elite_size]

        return Population(
            individuals=[
                genome.copy()
                for genome in selected_genomes
            ]
        )
