import random

from evolution import *

from environment import FrozenLakeEnvironment
from agents import *
from simulation import Simulation

class GeneticAlgorithm:

    def __init__(
        self,
        population: Population,
        fitness_evaluator: FitnessEvaluator,
        generations: int,
        tournament_size: int,
        selection_amount: int,
        crossover_rate: float,
        mutation_rate: float,
        elite_size: int = 0
    ):
        self.metrics: list[GenerationMetrics] = []
        self.population = population
        self.fitness_evaluator = fitness_evaluator
        self.generations = generations

        self.selection = Selection(
            tournament_size=tournament_size,
        )
        self.selection_amount = selection_amount
        self.elitism = Elitism(
            elite_size=elite_size
        )

        self.crossover = Crossover(
            crossover_rate=crossover_rate
        )

        self.mutation = Mutation(
            mutation_rate=mutation_rate
        )

    def run(self):
        for generation in range(self.generations):

            self._evaluate_population()

            generation_metrics = self._collect_generation_metrics(
                population=self.population,
                generation=generation
            )

            self.metrics.append(generation_metrics)

            elites = self.elitism.select_elites(
                population=self.population,
            )

            parents = self.selection.select(
                population=self.population.individuals,
                amount=self.selection_amount
            )

            offspring_amount = (len(self.population) - len(elites))

            offspring = self._create_offspring(
                parents=parents,
                amount=offspring_amount   
            )

            self.population.individuals = elites.individuals + offspring

    def _evaluate_population(self):
        for agent in self.population.individuals:

            environment = FrozenLakeEnvironment() # corrigir para não instanciar um novo ambiente toda vez
            simulation = Simulation(
                    agent=agent, 
                    environment=environment
            )
            result = simulation.run()

            agent.fitness = self.fitness_evaluator.evaluate(result)
            agent.success = result.final_position == result.goal_position

    def _collect_generation_metrics(self, population: Population, generation: int) -> GenerationMetrics:
        if not population.individuals:
            raise ValueError("Population cannot be empty.")

        fitness_values = [
            agent.fitness
            for agent in population.individuals
        ]

        success_count = sum(
            agent.success
            for agent in population.individuals
        )

        best_fitness = max(fitness_values)
        average_fitness = sum(fitness_values) / len(population)
        worst_fitness = min(fitness_values)
        success_rate = success_count / len(population)

        return GenerationMetrics(
            generation=generation,
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            worst_fitness=worst_fitness,
            success_rate=success_rate
        )

    def _create_offspring(self, parents: Population, amount: int) -> list[Agent]:

        offspring = []

        while len(offspring) < amount:

            parent_a = random.choice(parents.individuals)
            parent_b = random.choice(parents.individuals)

            child_a_genome, child_b_genome = self.crossover.crossover(
                parent_a_genome=parent_a.genome,
                parent_b_genome=parent_b.genome
            )

            self.mutation.mutate(child_a_genome)
            self.mutation.mutate(child_b_genome)

            child_a = Agent(child_a_genome)
            child_b = Agent(child_b_genome)

            offspring.append(child_a)

            if len(offspring) < amount:
                offspring.append(child_b)

        return offspring


