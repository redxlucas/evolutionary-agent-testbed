import time
import sys

from experiment import ExperimentRunner
from experiment.experiment_runner import ExperimentCancelled
from utils import Logger
import config
from visualization.terminal_setup import ExperimentTerminalUI

def main():
    interactive = (
        getattr(sys.stdin, "isatty", lambda: False)()
        and getattr(sys.stdout, "isatty", lambda: False)()
    )
    Logger.configure(
        config.LOG_LEVEL,
        console_enabled=not interactive,
    )
    logger = Logger(__name__)

    if interactive:
        terminal = ExperimentTerminalUI(config, logger)
        terminal.run()
        return

    start_time = time.perf_counter()

    runner = ExperimentRunner(
        logger=logger,
        experiment_type=config.EXPERIMENT_TYPE,
    )
    try:
        runner.run()
    except (KeyboardInterrupt, ExperimentCancelled):
        return

    elapsed_time = time.perf_counter() - start_time

    logger.info(
        "Execução concluída · tempo=%.2f segundos",
        elapsed_time
    )

if __name__ == "__main__":
    main()
