"""Run the configured MiniGrid MLP+CEM experiment and save per-seed failures."""

import config
from evolution import FitnessStrategy
from experiment.experiment_factory import configure_experiment
from experiment.experiment_runner import ExperimentRunner
from experiment.experiment_types import ActivationFunction, ExperimentType
from utils import Logger


def main():
    if config.ENVIRONMENT_TYPE != "minigrid":
        raise ValueError("Set ENVIRONMENT_TYPE='minigrid' to run this diagnostic.")
    if config.EXPERIMENT_OPTIMIZER != "cross_entropy":
        raise ValueError("Set EXPERIMENT_OPTIMIZER='cross_entropy'.")

    Logger.configure(
        config.LOG_LEVEL,
        filename="minigrid_cem_diagnostics.log",
    )
    runner = ExperimentRunner(
        logger=Logger(__name__),
        runs=config.EXPERIMENT_RUNS,
        experiment_type=ExperimentType.MLP.value,
        base_seed=config.SEED,
        workers=config.WORKERS,
    )
    experiment = configure_experiment(
        ExperimentType.MLP,
        activation=ActivationFunction.RELU,
        hidden_size=config.MLP_HIDDEN_SIZES[0],
    )
    experiment.name = f"MLP + ReLU | {experiment.hidden_size} neurônios"

    results = runner._execute_runs(
        fitness_strategy=FitnessStrategy.REWARD,
        configuration=experiment,
    )

    for run in results:
        details = run.diagnostics
        print(
            f"seed={run.seed} | "
            f"success={details['success_count']}/{details['population_size']} "
            f"({details['success_rate_percent']:.2f}%) | "
            f"lava={details['lava_failures']} | "
            f"timeout={details['timeout_failures']} | "
            f"other={details['other_failures']} | "
            f"failure_endpoints={details['failure_endpoints']}"
        )

    print(f"CSV: {runner.last_diagnostics_path}")


if __name__ == "__main__":
    main()
