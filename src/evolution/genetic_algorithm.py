import random

from evolution import *

from environment import FrozenLakeEnvironment

from simulation import Simulation


class GeneticAlgorithm:

    def __init__(
        self,
        population: Population,
        fitness_evaluator: FitnessEvaluator,
        agent_creator,
        generations: int,
        tournament_size: int,
        selection_amount: int,
        crossover_rate: float,
        mutation_rate: float,
        elite_size: int = 0,
        evolve: bool = True
    ):
        self.metrics: list[GenerationMetrics] = []

        self.population = population
        self.fitness_evaluator = fitness_evaluator
        self.agent_creator = agent_creator

        self.generations = generations
        self.evolve = evolve

        self.selection = Selection(
            tournament_size=tournament_size
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

            if self.evolve:
                self._evolve_population()

        return self.metrics

    def _evaluate_population(self):

        environment = FrozenLakeEnvironment()

        for genome in self.population:

            agent = self.agent_creator(genome)

            simulation = Simulation(
                agent=agent,
                environment=environment
            )

            result = simulation.run()

            genome.fitness = self.fitness_evaluator.evaluate(
                result
            )

            genome.result = result

        environment.close()

    def _evolve_population(self):

        elites = self.elitism.select_elites(
            population=self.population
        )

        parents = self.selection.select(
            population=self.population.individuals,
            amount=self.selection_amount
        )

        offspring_amount = (
            len(self.population) - len(elites)
        )

        offspring = self._create_offspring(
            parents=parents,
            amount=offspring_amount
        )

        self.population.individuals = (
            elites.individuals + offspring
        )

    def _collect_generation_metrics(
        self,
        population: Population,
        generation: int
    ) -> GenerationMetrics:

        if not population.individuals:
            raise ValueError("Population cannot be empty.")

        fitness_values = [
            genome.fitness
            for genome in population.individuals
        ]

        success_count = sum(
            genome.result.total_reward >= 1
            for genome in population.individuals
        )

        best_fitness = max(fitness_values)
        average_fitness = (
            sum(fitness_values) / len(population)
        )
        worst_fitness = min(fitness_values)
        success_rate = success_count / len(population)

        # print(
        #     f"Generation {generation}: "
        #     f"best={best_fitness:.4f}, "
        #     f"avg={average_fitness:.4f}, "
        #     f"worst={worst_fitness:.4f}, "
        #     f"success_rate={success_rate:.4f}"
        # )

        return GenerationMetrics(
            generation=generation,
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            worst_fitness=worst_fitness,
            success_rate=success_rate
        )

    def _create_offspring(
        self,
        parents: Population,
        amount: int
    ) -> list:

        offspring = []

        while len(offspring) < amount:

            parent_a = random.choice(
                parents.individuals
            )

            parent_b = random.choice(
                parents.individuals
            )

            child_a, child_b = self.crossover.crossover(
                parent_a_genome=parent_a,
                parent_b_genome=parent_b
            )

            self.mutation.mutate(child_a)
            self.mutation.mutate(child_b)

            offspring.append(child_a)

            if len(offspring) < amount:
                offspring.append(child_b)

        return offspring