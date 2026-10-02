import time

from experiment import ExperimentRunner
from utils import Logger
import config

def main():
    Logger.configure(config.LOG_LEVEL)
    logger = Logger(__name__)

    start_time = time.perf_counter()

    runner = ExperimentRunner(logger=logger)
    runner.run()

    elapsed_time = time.perf_counter() - start_time

    logger.info(
        "Total execution time | %.2f seconds",
        elapsed_time
    )

if __name__ == "__main__":
    main()