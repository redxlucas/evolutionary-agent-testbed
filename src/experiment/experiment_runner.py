

from concurrent.futures import ProcessPoolExecutor

from experiment.experiment_factory import configure_experiment, create_algorithm
from utils import Logger

def _run_experiment(args):
    seed, experiment_type, fitness_strategy = args

    experiment = configure_experiment(
        experiment_type
    )

    algorithm = create_algorithm(
        seed,
        experiment,
        fitness_strategy
    )

    algorithm.run()

    return algorithm.metrics

class ExperimentRunner:

    def __init__(
        self,
        runs,
        experiment_type,
        fitness_strategy,
        base_seed=42,
        workers=1
    ):
        self.runs = runs
        self.experiment_type = experiment_type
        self.fitness_strategy = fitness_strategy
        self.base_seed = base_seed
        self.workers = workers
        self.results = []

        self.logger = Logger(__name__)

    def run(self):

        self.logger.info(
            "Starting experiment | fitness_strategy=%s | runs=%d | base_seed=%d | workers=%d",
            self.fitness_strategy.name,
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
