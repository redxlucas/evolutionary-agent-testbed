import random
import time

import numpy as np

import config

from agents.genome import AdalineGenome
from agents.genome import MovementGenome
from agents import AdalineAgent
from agents import MovementAgent

from evolution import GeneticAlgorithm
from evolution import FitnessEvaluator
from evolution import Population

from evolution import FitnessStrategy
from experiment import ExperimentAnalyzer
from experiment import ExperimentConfig
from experiment import ExperimentRunner

from utils import Logger

from visualization import ExperimentPlotter

def main():

    Logger.configure(config.LOG_LEVEL)
    logger = Logger(__name__)

    start_time = time.perf_counter()

    if config.EXPERIMENT_MODE == "FITNESS_COMPARISON":
        run_fitness_comparison(start_time, logger)

        return

    experiment = configure_experiment(
        config.EXPERIMENT_TYPE
    )

    runner = ExperimentRunner(
        runs=config.EXPERIMENT_RUNS,
        experiment_type=config.EXPERIMENT_TYPE,
        fitness_strategy=None,
        base_seed=config.SEED,
        workers=config.WORKERS
    )

    results = runner.run()

    analyzer = ExperimentAnalyzer(results)
    metrics = analyzer.analyze()

    experiment_plotter = ExperimentPlotter(metrics)
    experiment_plotter.plot_experiment()

    elapsed_time = time.perf_counter() - start_time

    logger.info(
        "Total execution time | %.2f seconds",
        elapsed_time
    )

    experiment_plotter.show()


def create_algorithm(
    seed: int,
    experiment: ExperimentConfig,
    logger: Logger,
    fitness_strategy=None,
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
        logger=logger,
        elite_size=config.ELITE_SIZE,
        evolve=experiment.evolve
    )


def configure_experiment(
    experiment: str
) -> ExperimentConfig:

    if experiment == "ADALINE":

        return ExperimentConfig(
            genome_factory=lambda: AdalineGenome.random(config.GENOME_LENGTH),
            agent_creator=AdalineAgent.create,
            evolve=False
        )

    if experiment == "MOVEMENT":

        return ExperimentConfig(
            genome_factory=lambda: MovementGenome.random(config.GENOME_LENGTH),
            agent_creator=MovementAgent.create,
            evolve=True
        )

    raise ValueError(
        f"Unknown experiment type: {experiment}"
    )

def run_fitness_comparison(start_time, logger):

    strategies = [
        FitnessStrategy.REWARD,
        FitnessStrategy.FINAL_DISTANCE,
        FitnessStrategy.PROGRESS,
    ]

    results = {}

    for strategy in strategies:

        runner = ExperimentRunner(
            runs=config.EXPERIMENT_RUNS,
            experiment_type=config.EXPERIMENT_TYPE,
            fitness_strategy=strategy,
            base_seed=config.SEED,
            workers=config.WORKERS
        )

        results[strategy] = runner.run()

    experiment_plotter = ExperimentPlotter(results)

    experiment_plotter.plot_fitness_comparison()

    elapsed_time = time.perf_counter() - start_time

    logger.info(
        "Total execution time | %.2f seconds",
        elapsed_time
    )

    experiment_plotter.show()

if __name__ == "__main__":
    main()