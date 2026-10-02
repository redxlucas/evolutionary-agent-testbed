import random

from agents.genome import Genome

from evolution import (
    Crossover,
    Elitism,
    FitnessEvaluator,
    GenerationMetrics,
    Mutation,
    Population,
    Selection,
)

from environment import FrozenLakeEnvironment

from simulation import Simulation
from utils import Logger


class GeneticAlgorithm:
    """
    Implements a genetic algorithm for evolving a population of genomes.
    """

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
        logger: Logger,
        elite_size: int = 0,
        evolve: bool = True,
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
        self.logger = logger

    def run(self):
        """
        Runs the genetic algorithm and returns the generation metrics.
        """
        environment = FrozenLakeEnvironment()

        try:
            for generation in range(self.generations):
                self.logger.debug(
                    "Starting generation | generation=%d",
                    generation
                )

                self._evaluate_population(environment, generation)

                generation_metrics = self._collect_generation_metrics(
                    population=self.population,
                    generation=generation
                )

                self.metrics.append(generation_metrics)

                if self.evolve:
                    self._evolve_population()

            return self.metrics
        
        finally:
            environment.close()

    def _evaluate_population(
        self,
        environment: FrozenLakeEnvironment,
        generation: int,
    ):
        """
        Evaluates the fitness of all individuals in the population.
        """

        for individual_index, genome in enumerate(self.population):

            trace_context = (
                f"generation={generation} individual={individual_index}"
            )
            self.logger.debug(
                "Evaluating genome | %s | genes=%s",
                trace_context,
                genome.genes.tolist(),
            )

            agent = self.agent_creator(genome)

            simulation = Simulation(
                agent=agent,
                environment=environment,
                trace_context=trace_context,
            )

            result = simulation.run()

            genome.fitness = self.fitness_evaluator.evaluate(
                result
            )

            genome.result = result

            self.logger.debug(
                "Individual evaluated | %s | fitness=%.4f | success=%s",
                trace_context,
                genome.fitness,
                result.success,
            )

    def _collect_generation_metrics(
        self,
        population: Population,
        generation: int
    ) -> GenerationMetrics:
        """
        Collects fitness and success metrics for a generation.
        """

        if not population.individuals:
            raise ValueError("Population cannot be empty.")

        fitness_values = [
            genome.fitness
            for genome in population.individuals
        ]

        success_count = sum(
            genome.result.success
            for genome in population.individuals
        )

        best_fitness = max(fitness_values)
        average_fitness = (
            sum(fitness_values) / len(population)
        )
        worst_fitness = min(fitness_values)
        success_rate = success_count / len(population)

        self.logger.debug(
            "Generation metrics | generation=%d | best=%.4f | avg=%.4f | "
            "worst=%.4f | success_rate=%.4f",
            generation,
            best_fitness,
            average_fitness,
            worst_fitness,
            success_rate
        )

        return GenerationMetrics(
            generation=generation,
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            worst_fitness=worst_fitness,
            success_rate=success_rate
        )

    def _evolve_population(self):
        """
        Creates the next generation through selection, crossover, and mutation.
        """

        elites = self.elitism.select_elites(
            population=self.population
        )
        self.logger.debug(
            "Elites preserved | fitnesses=%s",
            [genome.fitness for genome in elites.individuals],
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

        self.logger.debug(
            "Evolving population | elites=%d | parents=%d | offspring=%d",
            len(elites),
            len(parents),
            offspring_amount
        )

    def _create_offspring(
        self,
        parents: Population,
        amount: int
    ) -> list[Genome]:
        """
        Creates offspring from selected parents using crossover and mutation.
        """

        offspring = []

        self.logger.debug(
            "Creating offspring | target_amount=%d",
            amount
        )

        while len(offspring) < amount:

            parent_a = random.choice(
                parents.individuals
            )

            parent_b = random.choice(
                parents.individuals
            )

            child_a, child_b = self.crossover.crossover(
                parent_a=parent_a,
                parent_b=parent_b
            )

            self.mutation.mutate(child_a)
            self.mutation.mutate(child_b)

            offspring.append(child_a)

            if len(offspring) < amount:
                offspring.append(child_b)

        return offspring
