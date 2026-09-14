from datetime import datetime

import matplotlib.pyplot as plt

import random
import statistics

from evolution import (
    GeneticAlgorithm,
    Population,
    FitnessEvaluator,
    FitnessStrategy,
)

import config


def main():

    strategies = [
        FitnessStrategy.REWARD,
        FitnessStrategy.FINAL_DISTANCE,
        FitnessStrategy.PROGRESS,
    ]

    results = {}

    for strategy in strategies:

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Starting strategy: {strategy.value}"
        )

        experiments = []

        for seed in config.SEEDS:

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Running experiment with seed {seed}..."
            )

            algorithm = run_experiment(
                strategy=strategy,
                seed=seed,
            )

            experiments.append(algorithm)

        results[strategy] = experiments

    summaries = analyze_results(results)

    for strategy, summary in summaries.items():

        print_results(
            title=f"Fitness Strategy: {strategy.value}",
            results=summary,
        )

    plot_success_rate_comparison(
        results=results,
    )


def run_experiment(
    strategy: FitnessStrategy,
    seed: int,
) -> GeneticAlgorithm:

    random.seed(seed)

    population = Population.random(
        size=config.POPULATION_SIZE,
        genome_length=config.GENOME_LENGTH,
    )

    algorithm = GeneticAlgorithm(
        population=population,
        fitness_evaluator=FitnessEvaluator(
            strategy=strategy
        ),
        generations=config.GENERATIONS,
        tournament_size=config.TOURNAMENT_SIZE,
        selection_amount=config.SELECTION_AMOUNT,
        crossover_rate=config.CROSSOVER_RATE,
        mutation_rate=config.MUTATION_RATE,
        elite_size=config.ELITE_SIZE,
    )

    algorithm.run()

    return algorithm


def get_first_success_generation(
    algorithm: GeneticAlgorithm,
) -> int | None:

    for metric in algorithm.metrics:

        if metric.success_rate > 0:
            return metric.generation

    return None


def get_max_success_rate(
    algorithm: GeneticAlgorithm,
) -> float:

    return max(
        metric.success_rate
        for metric in algorithm.metrics
    )


def analyze_results(
    results: dict[FitnessStrategy, list[GeneticAlgorithm]],
) -> dict:

    summaries = {}

    for strategy, experiments in results.items():

        final_metrics = [
            experiment.metrics[-1]
            for experiment in experiments
        ]

        first_success_generations = [
            get_first_success_generation(
                algorithm=experiment
            )
            for experiment in experiments
        ]

        successful_first_generations = [
            generation
            for generation in first_success_generations
            if generation is not None
        ]

        max_success_rates = [
            get_max_success_rate(
                algorithm=experiment
            )
            for experiment in experiments
        ]

        summary = aggregate_results(
            results=final_metrics,
        )

        successful_experiments = len(
            successful_first_generations
        )

        total_experiments = len(experiments)

        summary["successful_experiments"] = {
            "successful": successful_experiments,
            "total": total_experiments,
            "rate": (
                successful_experiments
                / total_experiments
            ),
        }

        if successful_first_generations:

            summary["first_success_generation"] = (
                calculate_statistics(
                    values=successful_first_generations
                )
            )

        else:

            summary["first_success_generation"] = None

        summary["max_success_rate"] = (
            calculate_statistics(
                values=max_success_rates
            )
        )

        summaries[strategy] = summary

    return summaries


def calculate_statistics(
    values: list[float | int],
) -> dict:

    return {
        "mean": statistics.mean(values),
        "stdev": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "min": min(values),
        "max": max(values),
    }

def calculate_mean_metric_by_generation(
    experiments: list[GeneticAlgorithm],
    metric_name: str,
) -> tuple[list[int], list[float]]:

    generations = [
        metric.generation
        for metric in experiments[0].metrics
    ]

    mean_values = []

    for generation_index in range(len(generations)):

        values = [
            getattr(
                experiment.metrics[generation_index],
                metric_name
            )
            for experiment in experiments
        ]

        mean_values.append(
            statistics.mean(values)
        )

    return generations, mean_values


def aggregate_results(
    results,
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
        "best_fitness": calculate_statistics(
            best_fitness_values
        ),

        "worst_fitness": calculate_statistics(
            worst_fitness_values
        ),

        "average_fitness": calculate_statistics(
            average_fitness_values
        ),

        "success_rate": calculate_statistics(
            success_rate_values
        ),
    }


def print_results(
    title: str,
    results: dict,
):

    print(f"\n{'=' * 50}")

    print(title)

    print(f"{'=' * 50}")

    for metric_name, values in results.items():

        print(
            f"\n{metric_name.replace('_', ' ').title()}:"
        )

        if values is None:

            print(
                "\n    No successful experiments."
            )

            continue

        if metric_name == "successful_experiments":

            print(
                f"\n    Successful Experiments: "
                f"{values['successful']} "
                f"/ {values['total']}"

                f"\n    Success Rate: "
                f"{values['rate']:.2%}"
            )

            continue

        print(
            f"\n    Mean: {values['mean']:.4f}"
            f"\n    Standard Deviation: "
            f"{values['stdev']:.4f}"
            f"\n    Min: {values['min']:.4f}"
            f"\n    Max: {values['max']:.4f}"
        )

def plot_success_rate_comparison(
    results: dict[FitnessStrategy, list[GeneticAlgorithm]],
):

    strategy_labels = {
        FitnessStrategy.REWARD: "Recompensa",
        FitnessStrategy.FINAL_DISTANCE: "Distância final",
        FitnessStrategy.PROGRESS: "Progresso",
    }

    plt.figure(figsize=(10, 5))

    for strategy, experiments in results.items():

        generations = [
            metric.generation
            for metric in experiments[0].metrics
        ]

        mean_success_rates = []

        for generation_index in range(len(generations)):

            success_rates = [
                experiment.metrics[
                    generation_index
                ].success_rate
                for experiment in experiments
            ]

            mean_success_rates.append(
                statistics.mean(success_rates)
            )

        plt.plot(
            generations,
            mean_success_rates,
            label=strategy_labels[strategy],
        )

    plt.xlabel("Geração")

    plt.ylabel("Taxa média de sucesso")

    plt.title(
        "Evolução da taxa média de sucesso por estratégia de fitness"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()