

from concurrent.futures import ProcessPoolExecutor

import config
from evolution import FitnessStrategy

from .experiment_types import ExperimentMode
from .experiment_analyzer import ExperimentAnalyzer
from .experiment_factory import configure_experiment, create_algorithm

from utils import Logger
from visualization.experiment_plotter import ExperimentPlotter

def _run_experiment(args):
    """
    Executes a single experiment run in a worker process.

    Creates the experiment and genetic algorithm using the provided
    configuration, executes the evolution, and returns its metrics.
    """
    seed, experiment_type, fitness_strategy = args

    Logger.configure(config.LOG_LEVEL)
    logger = Logger(__name__)

    experiment = configure_experiment(
        experiment_type
    )

    algorithm = create_algorithm(
        seed=seed,
        experiment=experiment,
        logger=logger,
        fitness_strategy=fitness_strategy,
    )

    algorithm.run()

    return algorithm.metrics

class ExperimentRunner:
    """
    Coordinates multiple independent experiment runs.

    Runs can be executed sequentially or in parallel using multiple
    worker processes, with a deterministic seed assigned to each run.
    """

    def __init__(
        self,
        logger: Logger,
        runs: int = config.EXPERIMENT_RUNS,
        experiment_type: str = config.EXPERIMENT_TYPE,
        base_seed: int = config.SEED,
        workers: int = config.WORKERS,
    ):
        self.logger = logger
        self.runs = runs
        self.experiment_type = experiment_type
        self.base_seed = base_seed
        self.workers = workers

    def run(self):
        """
        Executes the experiment configured in the application.
        """
        if config.EXPERIMENT_MODE == ExperimentMode.FITNESS_COMPARISON.value:
            return self._run_fitness_comparison()

        return self._run_experiment()

    def _run_experiment(self):
        results = self._execute_runs(
            fitness_strategy=FitnessStrategy.REWARD
        )

        analyzer = ExperimentAnalyzer(results)
        metrics = analyzer.analyze()

        plotter = ExperimentPlotter(metrics)
        plotter.plot_experiment()
        plotter.show()

        return metrics

    def _run_fitness_comparison(self):
        results = {}

        for strategy in FitnessStrategy:
            results[strategy] = self._execute_runs(
                fitness_strategy=strategy
            )

        plotter = ExperimentPlotter(results)
        plotter.plot_fitness_comparison()
        plotter.show()

        return results

    def _execute_runs(
        self,
        fitness_strategy: FitnessStrategy,
    ):
        self.logger.info(
            "Starting experiment | fitness_strategy=%s | "
            "runs=%d | base_seed=%d | workers=%d",
            fitness_strategy.value,
            self.runs,
            self.base_seed,
            self.workers,
        )

        tasks = [
            (
                self.base_seed + run,
                self.experiment_type,
                fitness_strategy,
            )
            for run in range(self.runs)
        ]

        with ProcessPoolExecutor(
            max_workers=self.workers
        ) as executor:
            results = list(
                executor.map(
                    _run_experiment,
                    tasks,
                )
            )

        self.logger.info(
            "Experiment completed | fitness_strategy=%s",
            fitness_strategy.value,
        )

        return results
