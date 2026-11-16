"""Profile every worker in the currently configured experiment run."""

from __future__ import annotations

import cProfile
from pathlib import Path
import pstats

from experiment import experiment_runner
from experiment.experiment_factory import configure_experiment
from evolution import FitnessStrategy
from utils import Logger


PROFILE_DIRECTORY = Path("profiling_output")
PROFILE_DIRECTORY.mkdir(exist_ok=True)

_run_experiment = experiment_runner._run_experiment


def _profiled_run_experiment(args):
    seed = args[0]
    profiler = cProfile.Profile()
    result = profiler.runcall(_run_experiment, args)
    profiler.dump_stats(PROFILE_DIRECTORY / f"run_{seed}.prof")
    return result


def main():
    experiment_runner._run_experiment = _profiled_run_experiment

    logger = Logger("profile")
    runner = experiment_runner.ExperimentRunner(logger=logger)
    configuration = configure_experiment(runner.experiment_type)

    print(
        "Profiling configured run | "
        f"environment={__import__('config').ENVIRONMENT_TYPE} | "
        f"experiment={runner.experiment_type.value} | "
        f"runs={runner.runs} | workers={runner.workers}"
    )
    runner._execute_runs(
        fitness_strategy=FitnessStrategy.REWARD,
        configuration=configuration,
    )

    profile_files = sorted(PROFILE_DIRECTORY.glob("run_*.prof"))
    if not profile_files:
        raise RuntimeError("No worker profile files were produced.")

    combined = pstats.Stats(str(profile_files[0]))
    for profile_file in profile_files[1:]:
        combined.add(str(profile_file))

    combined.dump_stats(PROFILE_DIRECTORY / "combined.prof")
    combined.strip_dirs().sort_stats("cumulative").print_stats(40)


if __name__ == "__main__":
    main()
