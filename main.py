import random
import statistics

from evolution import GeneticAlgorithm
from evolution import Population
from evolution import FitnessEvaluator

import config

from visualization.generation_plotter import GenerationPlotter


def main():

    without_elitism_results = []

    with_elitism_results = []

    for seed in config.SEEDS:

        print(f"Running experiments with seed {seed}...")

        without_elitism = run_experiment(
            elite_size=0,
            seed=seed
        )

        with_elitism = run_experiment(
            elite_size=config.ELITE_SIZE,
            seed=seed
        )

        without_elitism_results.append(
            without_elitism.metrics[-1]
        )

        with_elitism_results.append(
            with_elitism.metrics[-1]
        )

    without_elitism_summary = aggregate_results(
        results=without_elitism_results
    )

    with_elitism_summary = aggregate_results(
        results=with_elitism_results
    )

    print_results(
        title="Without Elitism",
        results=without_elitism_summary
    )

    print_results(
        title=f"With Elitism ({config.ELITE_SIZE})",
        results=with_elitism_summary
    )


def run_experiment(
    elite_size: int,
    seed: int
) -> GeneticAlgorithm:

    random.seed(seed)

    population = Population.random(
        size=config.POPULATION_SIZE,
        genome_length=config.GENOME_LENGTH
    )

    algorithm = GeneticAlgorithm(
        population=population,
        fitness_evaluator=FitnessEvaluator(),
        generations=config.GENERATIONS,
        tournament_size=config.TOURNAMENT_SIZE,
        selection_amount=config.SELECTION_AMOUNT,
        crossover_rate=config.CROSSOVER_RATE,
        mutation_rate=config.MUTATION_RATE,
        elite_size=elite_size
    )

    algorithm.run()

    return algorithm


def aggregate_results(
    results
) -> dict:

    best_fitness_values = [
        result.best_fitness
        for result in results
    ]

    worst_fitness_values = [
        result.worst_fitness
        for result in results
    ]

    average_fitness_values = [
        result.average_fitness
        for result in results
    ]

    success_rate_values = [
        result.success_rate
        for result in results
    ]

    return {
        "best_fitness": {
            "mean": statistics.mean(best_fitness_values),
            "stdev": statistics.stdev(best_fitness_values),
            "min": min(best_fitness_values),
            "max": max(best_fitness_values)
        },

        "worst_fitness": {
            "mean": statistics.mean(worst_fitness_values),
            "stdev": statistics.stdev(worst_fitness_values),
            "min": min(worst_fitness_values),
            "max": max(worst_fitness_values)
        },

        "average_fitness": {
            "mean": statistics.mean(average_fitness_values),
            "stdev": statistics.stdev(average_fitness_values),
            "min": min(average_fitness_values),
            "max": max(average_fitness_values)
        },

        "success_rate": {
            "mean": statistics.mean(success_rate_values),
            "stdev": statistics.stdev(success_rate_values),
            "min": min(success_rate_values),
            "max": max(success_rate_values)
        }
    }


def print_results(
    title: str,
    results: dict
):

    print(f"\n{'=' * 50}")
    print(title)
    print(f"{'=' * 50}")

    for metric_name, values in results.items():

        print(f"""
{metric_name.replace("_", " ").title()}:

    Mean: {values["mean"]:.4f}
    Standard Deviation: {values["stdev"]:.4f}
    Min: {values["min"]:.4f}
    Max: {values["max"]:.4f}
""")


if __name__ == "__main__":
    main()