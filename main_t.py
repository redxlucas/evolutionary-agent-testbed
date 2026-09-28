import random

import numpy as np

import config

from agents import Genome
from agents import AdalineGenome
from agents import AdalineAgent
from agents import GenomeAgent

from evolution import GeneticAlgorithm
from evolution import FitnessEvaluator
from evolution import Population

from evolution import FitnessStrategy
from experiment import ExperimentAnalyzer
from experiment import ExperimentConfig
from experiment import ExperimentRunner

from utils import Logger, logger

from visualization import ExperimentPlotter


def main():

    Logger.configure()

    if config.EXPERIMENT_MODE == "FITNESS_COMPARISON":
        run_fitness_comparison()
        return

    experiment = configure_experiment(
        config.EXPERIMENT_TYPE
    )

    runner = ExperimentRunner(
        runs=config.EXPERIMENT_RUNS,
        algorithm_factory=lambda seed: create_algorithm(
            seed,
            experiment
        ),
        base_seed=config.SEED
    )

    results = runner.run()

    analyzer = ExperimentAnalyzer(results)
    metrics = analyzer.analyze()

    experiment_plotter = ExperimentPlotter(metrics)
    experiment_plotter.plot_experiment()
    experiment_plotter.show()


def create_algorithm(
    seed: int,
    experiment: ExperimentConfig,
    fitness_strategy=None
):

    random.seed(seed)
    np.random.seed(seed)

    population = Population.random(
        size=config.POPULATION_SIZE,
        genome_factory=experiment.genome_factory
    )

    fitness_evaluator = FitnessEvaluator(
        strategy=fitness_strategy
    )

    return GeneticAlgorithm(
        population=population,
        fitness_evaluator=fitness_evaluator,
        agent_creator=experiment.agent_creator,
        generations=config.GENERATIONS,
        tournament_size=config.TOURNAMENT_SIZE,
        selection_amount=config.SELECTION_AMOUNT,
        crossover_rate=config.CROSSOVER_RATE,
        mutation_rate=config.MUTATION_RATE,
        elite_size=config.ELITE_SIZE,
        evolve=experiment.evolve
    )


def configure_experiment(
    experiment: str
) -> ExperimentConfig:

    if experiment == "ADALINE":

        return ExperimentConfig(
            genome_factory=lambda: AdalineGenome.random(config.GENOME_LENGTH),
            agent_creator=AdalineAgent.create_adaline_agent,
            evolve=False
        )

    if experiment == "MOVEMENT":

        return ExperimentConfig(
            genome_factory=lambda: Genome.random(config.GENOME_LENGTH),
            agent_creator=GenomeAgent.create_genome_agent,
            evolve=True
        )

    raise ValueError(
        f"Unknown experiment type: {experiment}"
    )

def run_fitness_comparison():

    experiment = configure_experiment(
        config.EXPERIMENT_TYPE
    )

    strategies = [
        FitnessStrategy.REWARD,
        FitnessStrategy.FINAL_DISTANCE,
        FitnessStrategy.PROGRESS,
    ]

    results = {}

    for strategy in strategies:

        # logger.info(
        #     "Starting fitness strategy experiment | strategy=%s | runs=%d | base_seed=%d",
        #     strategy.value,
        #     config.EXPERIMENT_RUNS,
        #     config.SEED
        # )

        runner = ExperimentRunner(
            runs=config.EXPERIMENT_RUNS,
            algorithm_factory=lambda seed, strategy=strategy: (
                create_algorithm(
                    seed,
                    experiment,
                    strategy
                )
            ),
            base_seed=config.SEED
        )

        results[strategy] = runner.run()

        # logger.info(
        #     "Finished fitness strategy experiment | strategy=%s",
        #     strategy.value
        # )

    experiment_plotter = ExperimentPlotter(results)

    experiment_plotter.plot_fitness_comparison()
    experiment_plotter.show()

if __name__ == "__main__":
    main()