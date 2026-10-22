import numpy as np

from agents.genome import Genome
from environment import FrozenLakeEnvironment
from evolution import FitnessEvaluator, GenerationMetrics
from simulation import Simulation
from utils import Logger


class CrossEntropyOptimizer:
    """
    Optimizes agent parameters using the Cross-Entropy Method (CEM).

    The optimizer maintains an independent Gaussian distribution for each
    genome parameter. At each iteration, candidates are sampled from the
    distribution, evaluated, and the distribution is updated using the
    elite candidates.
    """

    def __init__(
        self,
        parameter_count: int,
        population_size: int,
        elite_size: int,
        iterations: int,
        fitness_evaluator: FitnessEvaluator,
        agent_creator,
        genome_creator,
        logger: Logger,
        initial_std: float = 1.0,
        min_std: float = 1e-3,
        seed: int | None = None,
    ):
        if population_size <= 0:
            raise ValueError(
                "Population size must be greater than zero."
            )

        if elite_size <= 0:
            raise ValueError(
                "Elite size must be greater than zero."
            )

        if elite_size > population_size:
            raise ValueError(
                "Elite size cannot be greater than population size."
            )

        if parameter_count <= 0:
            raise ValueError(
                "Parameter count must be greater than zero."
            )

        if iterations <= 0:
            raise ValueError(
                "Iterations must be greater than zero."
            )

        if initial_std <= 0:
            raise ValueError(
                "Initial standard deviation must be greater than zero."
            )

        if min_std <= 0:
            raise ValueError(
                "Minimum standard deviation must be greater than zero."
            )

        self.parameter_count = parameter_count
        self.population_size = population_size
        self.elite_size = elite_size
        self.iterations = iterations

        self.fitness_evaluator = fitness_evaluator
        self.agent_creator = agent_creator
        self.genome_creator = genome_creator

        self.initial_std = initial_std
        self.min_std = min_std

        self.logger = logger

        self.rng = np.random.default_rng(seed)

        self.mean = np.zeros(self.parameter_count)
        self.std = np.full(
            self.parameter_count,
            self.initial_std,
        )

        self.metrics: list[GenerationMetrics] = []

        self.best_genome: Genome | None = None
        self.best_fitness = float("-inf")

    def run(self):
        """
        Runs the Cross-Entropy optimization process.
        """
        environment = FrozenLakeEnvironment()

        try:
            for iteration in range(self.iterations):
                self.logger.debug(
                    "Starting CEM iteration | iteration=%d",
                    iteration,
                )

                candidates = self._sample_population()

                self._evaluate_population(
                    candidates=candidates,
                    environment=environment,
                    iteration=iteration,
                )

                generation_metrics = (
                    self._collect_generation_metrics(
                        candidates=candidates,
                        generation=iteration,
                    )
                )

                self.metrics.append(generation_metrics)

                elites = self._select_elites(candidates)

                self._update_distribution(elites)

                self.logger.debug(
                    "CEM distribution updated | iteration=%d",
                    iteration,
                )

            return self.metrics

        finally:
            environment.close()

    def _sample_population(self) -> list[Genome]:
        """
        Samples candidate genomes from the current Gaussian distribution.
        """
        samples = self.rng.normal(
            loc=self.mean,
            scale=self.std,
            size=(
                self.population_size,
                self.parameter_count,
            ),
        )

        candidates = []

        for parameters in samples:
            genome = self.genome_creator(parameters)
            candidates.append(genome)

        return candidates

    def _evaluate_population(
        self,
        candidates: list[Genome],
        environment: FrozenLakeEnvironment,
        iteration: int,
    ):
        """
        Evaluates all sampled candidates using the existing simulation
        and fitness evaluation pipeline.
        """
        for candidate_index, genome in enumerate(candidates):
            trace_context = (
                f"iteration={iteration} "
                f"individual={candidate_index}"
            )

            self.logger.debug(
                "Evaluating CEM candidate | %s | genes=%s",
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
                "CEM candidate evaluated | %s | fitness=%.4f | "
                "success=%s",
                trace_context,
                genome.fitness,
                result.success,
            )

    def _select_elites(
        self,
        candidates: list[Genome],
    ) -> list[Genome]:
        """
        Selects the candidates with the highest fitness values.
        """
        sorted_candidates = sorted(
            candidates,
            key=lambda genome: genome.fitness,
            reverse=True,
        )

        return sorted_candidates[:self.elite_size]

    def _update_distribution(
        self,
        elites: list[Genome],
    ):
        """
        Updates the Gaussian distribution using the elite candidates.
        """
        elite_parameters = np.array(
            [
                genome.genes
                for genome in elites
            ]
        )

        self.mean = np.mean(
            elite_parameters,
            axis=0,
        )

        self.std = np.std(
            elite_parameters,
            axis=0,
        )

        self.std = np.maximum(
            self.std,
            self.min_std,
        )

    def _collect_generation_metrics(
        self,
        candidates: list[Genome],
        generation: int,
    ) -> GenerationMetrics:
        """
        Collects fitness and success metrics for the current CEM iteration.
        """
        if not candidates:
            raise ValueError(
                "Candidate population cannot be empty."
            )

        fitness_values = [
            genome.fitness
            for genome in candidates
        ]

        success_count = sum(
            genome.result.success
            for genome in candidates
        )

        best_fitness = max(fitness_values)

        average_fitness = (
            sum(fitness_values) / len(candidates)
        )

        worst_fitness = min(fitness_values)

        success_rate = (
            success_count / len(candidates)
        )

        self.logger.debug(
            "CEM iteration metrics | iteration=%d | best=%.4f | "
            "avg=%.4f | worst=%.4f | success_rate=%.4f",
            generation,
            best_fitness,
            average_fitness,
            worst_fitness,
            success_rate,
        )

        self._update_best(candidates)

        return GenerationMetrics(
            generation=generation,
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            worst_fitness=worst_fitness,
            success_rate=success_rate,
        )

    def _update_best(
        self,
        candidates: list[Genome],
    ):
        """
        Stores the best genome found during the optimization.
        """
        candidate = max(
            candidates,
            key=lambda genome: genome.fitness,
        )

        if candidate.fitness > self.best_fitness:
            self.best_fitness = candidate.fitness
            self.best_genome = candidate.copy()