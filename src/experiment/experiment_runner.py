

from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np

import config
from evolution import FitnessStrategy

from .experiment_types import (
    ActivationFunction,
    ExperimentMode,
    ExperimentType,
)
from .experiment_analyzer import ExperimentAnalyzer
from .experiment_factory import (
    ADALINE_CONFIG,
    MOVEMENT_CONFIG,
    configure_experiment,
    create_algorithm,
)

from utils import Logger
from visualization.experiment_plotter import ExperimentPlotter

def _run_experiment(args):
    """
    Executes a single experiment run in a worker process.

    Creates the experiment and optimization algorithm using the provided
    configuration, executes the evolution, and returns its metrics.
    """
    (
        seed,
        experiment_type,
        fitness_strategy,
        activation,
        evolve_alpha,
        hidden_size,
    ) = args

    log_filename = (
        f"{experiment_type.value}_{fitness_strategy.value}_debug.log"
    )

    Logger.configure(
        config.LOG_LEVEL,
        filename=log_filename,
    )

    logger = Logger(__name__)

    experiment = configure_experiment(
        experiment_type,
        activation=activation,
        evolve_alpha=evolve_alpha,
        hidden_size=hidden_size,
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
        self.experiment_type = ExperimentType(experiment_type)
        self.base_seed = base_seed
        self.workers = workers

    def run(self):
        """
        Executes the experiment configured in the application.
        """
        if config.EXPERIMENT_MODE == ExperimentMode.SINGLE.value:
            return self._run_single()

        if config.EXPERIMENT_MODE == ExperimentMode.FITNESS_COMPARISON.value:
            return self._run_fitness_comparison()

        if (
            config.EXPERIMENT_MODE
            == ExperimentMode.CONFIGURATION_COMPARISON.value
        ):
            return self._run_configuration_comparison()

        raise ValueError(
            f"Unknown experiment mode: {config.EXPERIMENT_MODE}"
        )

    def _run_single(self):
        """Run the configured agent and optimizer using reward fitness."""
        if self.experiment_type == ExperimentType.MLP:
            if not config.MLP_HIDDEN_SIZES:
                raise ValueError(
                    "MLP_HIDDEN_SIZES must contain a hidden size."
                )

            hidden_size = config.MLP_HIDDEN_SIZES[0]
            configuration = configure_experiment(
                ExperimentType.MLP,
                activation=ActivationFunction.RELU,
                hidden_size=hidden_size,
            )
            configuration.name = (
                f"MLP + ReLU | {hidden_size} neurônios"
            )
        else:
            configuration = configure_experiment(self.experiment_type)

        results = self._execute_runs(
            fitness_strategy=FitnessStrategy.REWARD,
            configuration=configuration,
        )
        metrics = ExperimentAnalyzer(results).analyze()
        optimizer = self._optimizer_for(configuration)

        plotter = ExperimentPlotter(metrics, self.experiment_type)
        plotter.plot_experiment(
            title=(
                f"Taxa de sucesso por geração - {configuration.name} | "
                f"{optimizer}"
            )
        )
        plotter.show()

        return metrics

    def _run_fitness_comparison(self):
        results = {}

        comparison_strategies = (
            FitnessStrategy.REWARD,
            FitnessStrategy.FINAL_DISTANCE,
            FitnessStrategy.PROGRESS,
        )

        for strategy in comparison_strategies:
            results[strategy] = self._execute_runs(
                fitness_strategy=strategy
            )

            print(f"\n{'=' * 60}")
            print(f"Fitness strategy: {strategy.value}")
            print(f"{'=' * 60}")

            success_rates = []

            for run_index, run_metrics in enumerate(
                results[strategy],
                start=1
            ):
                final_metrics = run_metrics[-1]

                success_rate = final_metrics.success_rate * 100
                success_rates.append(success_rate)

                print(
                    f"Run {run_index:02d}: "
                    f"Success = {success_rate:.2f}% | "
                    f"Best fitness = {final_metrics.best_fitness:.4f} | "
                    f"Average fitness = {final_metrics.average_fitness:.4f}"
                )

            average = np.mean(success_rates)
            std = np.std(success_rates, ddof=1)

            print("-" * 60)
            print(f"Average success rate: {average:.2f}%")
            print(f"Std Dev: {std:.2f} p.p.")

        plotter = ExperimentPlotter(
            results,
            self.experiment_type
        )

        plotter.plot_fitness_comparison()
        plotter.show()

        return results

    def _run_configuration_comparison(self):

        configurations = []
        for hidden_size in config.MLP_HIDDEN_SIZES:
            relu_configuration = configure_experiment(
                ExperimentType.MLP,
                activation=ActivationFunction.RELU,
                hidden_size=hidden_size,
            )
            relu_configuration.name = (
                f"MLP + ReLU | {hidden_size} neurônios"
            )
            configurations.append(relu_configuration)

            leaky_relu_configuration = configure_experiment(
                ExperimentType.MLP,
                activation=ActivationFunction.PRELU,
                hidden_size=hidden_size,
            )
            leaky_relu_configuration.name = (
                "MLP + Leaky ReLU (alpha = 0.25) | "
                f"{hidden_size} neurônios"
            )
            configurations.append(leaky_relu_configuration)

            prelu_configuration = configure_experiment(
                ExperimentType.MLP,
                activation=ActivationFunction.PRELU,
                evolve_alpha=True,
                hidden_size=hidden_size,
            )
            prelu_configuration.name = (
                "MLP + PReLU + Alpha | "
                f"{hidden_size} neurônios"
            )
            configurations.append(prelu_configuration)

        fitness_strategy = FitnessStrategy.REWARD

        results = {}

        for configuration in configurations:
            results[configuration.name] = self._execute_runs(
                fitness_strategy=fitness_strategy,
                configuration=configuration,
            )

        plotter = ExperimentPlotter(
            results,
            self.experiment_type,
        )
        plotter.plot_configuration_comparison()
        plotter.show()

        return results

    def _execute_runs(
        self,
        fitness_strategy: FitnessStrategy,
        configuration=None,
    ):
        agent_name = (
            configuration.name
            if configuration is not None
            else self.experiment_type.value
        )
        optimizer = self._optimizer_for(configuration)
        started_at = time.perf_counter()

        self.logger.info(
            "Experiment started | agent=%s | optimizer=%s | "
            "fitness_strategy=%s | runs=%d | base_seed=%d | workers=%d",
            agent_name,
            optimizer,
            fitness_strategy.value,
            self.runs,
            self.base_seed,
            self.workers,
        )

        tasks = [
            (
                self.base_seed + run,
                self._experiment_type_for(configuration),
                fitness_strategy,
                configuration.activation if configuration else None,
                configuration.evolve_alpha if configuration else False,
                configuration.hidden_size if configuration else None,
            )
            for run in range(self.runs)
        ]

        try:
            with ProcessPoolExecutor(
                max_workers=self.workers
            ) as executor:
                results = list(
                    executor.map(
                        _run_experiment,
                        tasks,
                    )
                )
        except Exception:
            self.logger.exception(
                "Experiment failed | agent=%s | optimizer=%s | "
                "elapsed=%.2f seconds",
                agent_name,
                optimizer,
                time.perf_counter() - started_at,
            )
            raise

        self.logger.info(
            "Experiment finished | agent=%s | optimizer=%s | "
            "fitness_strategy=%s | elapsed=%.2f seconds",
            agent_name,
            optimizer,
            fitness_strategy.value,
            time.perf_counter() - started_at,
        )

        return results

    def _optimizer_for(self, configuration):
        if (
            configuration is not None
            and configuration.name in {
                ADALINE_CONFIG.name,
                MOVEMENT_CONFIG.name,
            }
        ):
            return "genetic"

        return config.EXPERIMENT_OPTIMIZER

    def _experiment_type_for(self, configuration):
        if configuration is None:
            return self.experiment_type

        configuration_types = {
            ADALINE_CONFIG.name: ExperimentType.ADALINE,
            MOVEMENT_CONFIG.name: ExperimentType.MOVEMENT,
        }
        return configuration_types.get(
            configuration.name,
            ExperimentType.MLP,
        )
