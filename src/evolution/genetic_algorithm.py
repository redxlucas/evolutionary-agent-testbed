from evolution import FitnessEvaluator
from evolution import Population
from environment import FrozenLakeEnvironment
from agents import *
from evolution import Selection
from .generation_metrics import GenerationMetrics
from simulation import Simulation

class GeneticAlgorithm:

    def __init__(
        self,
        population: Population,
        fitness_evaluator: FitnessEvaluator,
        generations: int,
        tournament_size,
        selection_amount
    ):
        self.metrics: list[GenerationMetrics] = []
        self.population = population
        self.fitness_evaluator = fitness_evaluator
        self.generations = generations
        self.selection = Selection(
            tournament_size=tournament_size,
        )
        self.selection_amount=selection_amount

    def collect_generation_metrics(self, population: Population, generation: int) -> GenerationMetrics:
        if not population.individuals:
            raise ValueError("Population cannot be empty.")

        fitness_values = [
            agent.fitness
            for agent in population.individuals
        ]

        best_fitness = max(fitness_values)
        average_fitness = sum(fitness_values) / len(fitness_values)
        worst_fitness = min(fitness_values)
        success_rate = 0.0 # adicionar futuramente

        return GenerationMetrics(
            generation=generation,
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            worst_fitness=worst_fitness,
            success_rate=success_rate
        )

    def run(self):
        for generation in range(self.generations):

            for agent in self.population.individuals:

                environment = FrozenLakeEnvironment()

                simulation = Simulation(
                    agent=agent, 
                    environment=environment
                )

                result = simulation.run()

                agent.fitness = self.fitness_evaluator.evaluate(result)

            parents = self.selection.select(
                self.population.individuals,
                amount=self.selection_amount
            )

            generation_metrics = self.collect_generation_metrics(
                population=self.population,
                generation=generation
            )

            self.metrics.append(generation_metrics)
