from datetime import datetime
import statistics

import matplotlib.pyplot as plt

from evolution import (
    GeneticAlgorithm,
    Population,
    FitnessEvaluator,
    FitnessStrategy,
)

import config


EXPERIMENTS = 30


def main():
    strategies = [
        FitnessStrategy.REWARD,
        FitnessStrategy.FINAL_DISTANCE,
        FitnessStrategy.PROGRESS,
    ]

    experiment = Experiment(
        strategies=strategies,
        repetitions=EXPERIMENTS,
    )

    results = experiment.run()

    summaries = analyze_results(results)

    for strategy, summary in summaries.items():
        print_results(
            title=f"Fitness Strategy: {strategy.value}",
            results=summary,
        )

    plot_fitness_by_strategy(results)


class Experiment:

    def __init__(
        self,
        strategies: list[FitnessStrategy],
        repetitions: int,
    ):
        self.strategies = strategies
        self.repetitions = repetitions

    def run(
        self,
    ) -> dict[FitnessStrategy, list[GeneticAlgorithm]]:

        results = {}

        for strategy in self.strategies:

            print(
                f"\n{'=' * 60}"
            )
            print(
                f"Starting strategy: {strategy.value}"
            )
            print(
                f"Repetitions: {self.repetitions}"
            )
            print(
                f"{'=' * 60}"
            )

            experiments = []

            for repetition in range(self.repetitions):

                seed = repetition

                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Running experiment "
                    f"{repetition + 1}/{self.repetitions} "
                    f"with seed {seed}..."
                )

                algorithm = run_single_experiment(
                    strategy=strategy,
                    seed=seed,
                )

                experiments.append(algorithm)

            results[strategy] = experiments

        return results


def run_single_experiment(
    strategy: FitnessStrategy,
    seed: int,
) -> GeneticAlgorithm:

    population = Population.random(
        size=config.POPULATION_SIZE,
        genome_factory=experiment.genome_factory
    )

    algorithm = GeneticAlgorithm(
        population=population,
        fitness_evaluator=FitnessEvaluator(strategy),
        generations=config.GENERATIONS,
        tournament_size=config.TOURNAMENT_SIZE,
        selection_amount=config.SELECTION_AMOUNT,
        crossover_rate=config.CROSSOVER_RATE,
        mutation_rate=config.MUTATION_RATE,
        elite_size=config.ELITE_SIZE,
        seed=seed,
    )

    algorithm.run()

    return algorithm

if __name__ == "__main__":
    main()