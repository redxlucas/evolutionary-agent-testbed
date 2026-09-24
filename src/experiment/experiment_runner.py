

from utils import Logger

class ExperimentRunner:

    def __init__(
        self,
        runs,
        algorithm_factory,
        base_seed=42
    ):
        self.runs = runs
        self.algorithm_factory = algorithm_factory
        self.base_seed = base_seed
        self.results = []
        self.logger = Logger(__name__)

    def run(self):

        self.logger.info(
            "Starting experiment | runs=%d | base_seed=%d",
            self.runs,
            self.base_seed
        )

        for run in range(self.runs):
            seed = self.base_seed + run

            self.logger.info(
                "Starting run %d/%d | seed=%d",
                run + 1,
                self.runs,
                seed
            )

            algorithm = self.algorithm_factory(seed)
            algorithm.run()

            self.results.append(algorithm.metrics)

            self.logger.info(
                "Finished run %d/%d | seed=%d",
                run + 1,
                self.runs,
                seed
            )

        self.logger.info("Experiment completed")

        return self.results
