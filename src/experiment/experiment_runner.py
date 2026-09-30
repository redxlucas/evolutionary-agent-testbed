

from concurrent.futures import ProcessPoolExecutor

import config
from experiment.experiment_factory import configure_experiment, create_algorithm
from utils import Logger

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
        runs,
        experiment_type,
        fitness_strategy,
        logger: Logger,
        base_seed=42,
        workers=1,
    ):
        self.runs = runs
        self.experiment_type = experiment_type
        self.fitness_strategy = fitness_strategy
        self.base_seed = base_seed
        self.workers = workers
        self.results = []
        self.logger = logger

    def run(self):
        """
        Executes all configured experiment runs.

        Each run receives a unique seed derived from the base seed.
        Results are collected after all processes complete.
        """

        self.logger.info(
            "Starting experiment | fitness_strategy=%s | runs=%d | base_seed=%d | workers=%d",
            self.fitness_strategy.name if self.fitness_strategy else None,
            self.runs,
            self.base_seed,
            self.workers
        )

        tasks = [
            (
                self.base_seed + run,
                self.experiment_type,
                self.fitness_strategy
            )
            for run in range(self.runs)
        ]

        with ProcessPoolExecutor(
            max_workers=self.workers
        ) as executor:
            self.results = list(
                executor.map(
                    _run_experiment,
                    tasks
                )
            )

        self.logger.info("Experiment completed")

        return self.results
