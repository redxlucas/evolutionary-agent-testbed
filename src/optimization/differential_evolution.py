import numpy as np

from agents.genome import Genome
from environment import Environment, create_environment
from evolution import FitnessEvaluator, GenerationMetrics
from simulation import Simulation
from utils import Logger


class DifferentialEvolutionOptimizer:
    """Optimize a continuous genome with the DE/rand/1/bin strategy."""

    def __init__(
        self,
        parameter_count: int,
        population_size: int,
        generations: int,
        fitness_evaluator: FitnessEvaluator,
        agent_creator,
        genome_creator,
        logger: Logger,
        gene_bounds: tuple[float, float],
        mutation_factor_range: tuple[float, float] = (0.0, 1.0),
        crossover_rate_range: tuple[float, float] = (0.0, 1.0),
        seed: int | None = None,
        stop_event=None,
    ):
        if parameter_count <= 0:
            raise ValueError("Parameter count must be greater than zero.")
        if population_size < 4:
            raise ValueError(
                "Differential Evolution requires a population of at least four."
            )
        if generations <= 0:
            raise ValueError("Generations must be greater than zero.")
        lower_bound, upper_bound = gene_bounds
        if not np.isfinite([lower_bound, upper_bound]).all():
            raise ValueError("Gene bounds must be finite.")
        if lower_bound >= upper_bound:
            raise ValueError("Lower gene bound must be less than upper bound.")

        for name, value_range in (
            ("Mutation factor", mutation_factor_range),
            ("Crossover rate", crossover_rate_range),
        ):
            range_min, range_max = value_range
            if not 0 <= range_min < range_max <= 1:
                raise ValueError(
                    f"{name} range must satisfy 0 <= min < max <= 1."
                )

        self.parameter_count = parameter_count
        self.population_size = population_size
        self.generations = generations
        self.fitness_evaluator = fitness_evaluator
        self.agent_creator = agent_creator
        self.genome_creator = genome_creator
        self.logger = logger
        self.stop_event = stop_event
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mutation_factor_range = mutation_factor_range
        self.crossover_rate_range = crossover_rate_range
        self.rng = np.random.default_rng(seed)

        self.metrics: list[GenerationMetrics] = []
        self.last_generation_candidates: list[Genome] = []
        self.best_genome: Genome | None = None
        self.best_fitness = float("-inf")

    def run(self):
        """Run DE and return metrics for each evaluated population."""
        environment = create_environment()
        try:
            population = self._initialize_population()
            for generation in range(self.generations):
                if self.stop_event is not None and self.stop_event.is_set():
                    break

                self.logger.debug(
                    "Starting Differential Evolution generation | "
                    "generation=%d",
                    generation,
                )

                if generation == 0:
                    population_complete = self._evaluate_population(
                        population,
                        environment,
                        generation,
                    )
                    if not population_complete:
                        break
                else:
                    next_population = self._evolve_population(
                        population,
                        environment,
                        generation,
                    )
                    if next_population is None:
                        break
                    population = next_population

                self.last_generation_candidates = population
                self.metrics.append(
                    self._collect_generation_metrics(population, generation)
                )

            return self.metrics
        finally:
            environment.close()

    def _initialize_population(self) -> list[Genome]:
        vectors = self.rng.uniform(
            self.lower_bound,
            self.upper_bound,
            size=(self.population_size, self.parameter_count),
        )
        return [self.genome_creator(vector) for vector in vectors]

    def _evolve_population(
        self,
        population: list[Genome],
        environment: Environment,
        generation: int,
    ) -> list[Genome] | None:
        next_population = []
        for target_index, target in enumerate(population):
            mutation_factor = self.rng.uniform(*self.mutation_factor_range)
            crossover_rate = self.rng.uniform(*self.crossover_rate_range)
            donor_indices = self.rng.choice(
                [
                    index
                    for index in range(self.population_size)
                    if index != target_index
                ],
                size=3,
                replace=False,
            )
            base, first, second = (population[index].genes for index in donor_indices)
            mutant = base + mutation_factor * (first - second)
            mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

            crossover_mask = (
                self.rng.random(self.parameter_count) < crossover_rate
            )
            crossover_mask[self.rng.integers(self.parameter_count)] = True
            trial_vector = np.where(crossover_mask, mutant, target.genes)
            next_population.append(self.genome_creator(trial_vector))

        population_complete = self._evaluate_population(
            next_population,
            environment,
            generation,
        )
        if not population_complete:
            return None

        return [
            trial if trial.fitness >= target.fitness else target
            for target, trial in zip(population, next_population)
        ]

    def _evaluate_population(
        self,
        candidates: list[Genome],
        environment: Environment,
        generation: int,
    ) -> bool:
        for candidate_index, genome in enumerate(candidates):
            if self.stop_event is not None and self.stop_event.is_set():
                return False

            trace_context = (
                f"generation={generation} individual={candidate_index}"
            )
            self.logger.debug(
                "Evaluating DE candidate | %s | genes=%s",
                trace_context,
                genome.genes.tolist(),
            )
            simulation = Simulation(
                agent=self.agent_creator(genome),
                environment=environment,
                trace_context=trace_context,
            )
            result = simulation.run()
            genome.fitness = self.fitness_evaluator.evaluate(result)
            genome.result = result
            self.logger.debug(
                "DE candidate evaluated | %s | fitness=%.4f | success=%s",
                trace_context,
                genome.fitness,
                result.success,
            )

        return True

    def _collect_generation_metrics(
        self,
        candidates: list[Genome],
        generation: int,
    ) -> GenerationMetrics:
        fitness_values = np.asarray(
            [genome.fitness for genome in candidates],
            dtype=float,
        )
        if not np.all(np.isfinite(fitness_values)):
            raise ValueError(
                "Differential Evolution requires finite fitness values."
            )

        best_index = int(np.argmax(fitness_values))
        candidate = candidates[best_index]
        if candidate.fitness > self.best_fitness:
            self.best_fitness = candidate.fitness
            self.best_genome = candidate.copy()

        success_rate = (
            sum(genome.result.success for genome in candidates)
            / len(candidates)
        )
        self.logger.debug(
            "DE generation metrics | generation=%d | best=%.4f | avg=%.4f | "
            "worst=%.4f | success_rate=%.4f",
            generation,
            float(np.max(fitness_values)),
            float(np.mean(fitness_values)),
            float(np.min(fitness_values)),
            success_rate,
        )
        return GenerationMetrics(
            generation=generation,
            best_fitness=float(np.max(fitness_values)),
            average_fitness=float(np.mean(fitness_values)),
            worst_fitness=float(np.min(fitness_values)),
            success_rate=success_rate,
        )
