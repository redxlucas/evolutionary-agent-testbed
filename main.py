import time

import config

from evolution import FitnessStrategy
from experiment import (
    ExperimentAnalyzer,
    ExperimentRunner,
)
from visualization import ExperimentPlotter
from utils import Logger


def main():
    Logger.configure(config.LOG_LEVEL)
    logger = Logger(__name__)

    start_time = time.perf_counter()

    if config.EXPERIMENT_MODE == "FITNESS_COMPARISON":
        run_fitness_comparison(logger=logger)
    else:
        run_experiment(logger=logger)

    elapsed_time = time.perf_counter() - start_time

    logger.info(
        "Total execution time | %.2f seconds",
        elapsed_time
    )


def run_experiment(logger: Logger):
    runner = ExperimentRunner(
        runs=config.EXPERIMENT_RUNS,
        experiment_type=config.EXPERIMENT_TYPE,
        fitness_strategy=None,
        logger=logger,
        base_seed=config.SEED,
        workers=config.WORKERS
    )

    results = runner.run()

    analyzer = ExperimentAnalyzer(results)
    metrics = analyzer.analyze()

    plotter = ExperimentPlotter(metrics)
    plotter.plot_experiment()
    plotter.show()


def run_fitness_comparison(logger: Logger):
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
            logger=logger,
            base_seed=config.SEED,
            workers=config.WORKERS
        )

        results[strategy] = runner.run()

    plotter = ExperimentPlotter(results)
    plotter.plot_fitness_comparison()
    plotter.show()


if __name__ == "__main__":
    main()