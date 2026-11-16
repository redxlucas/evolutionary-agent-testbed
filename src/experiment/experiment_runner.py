

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
from datetime import datetime
import json
from multiprocessing import get_context
from pathlib import Path
import statistics
import time

import numpy as np

import config
from environment import create_environment
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
from .experiment_run_result import ExperimentRunResult

from utils import Logger
from visualization.experiment_plotter import ExperimentPlotter
from visualization.terminal_dashboard import TerminalDashboard


_WORKER_STOP_EVENT = None
_WORKER_CONSOLE_ENABLED = True


class ExperimentCancelled(Exception):
    """Raised when an experiment is cooperatively cancelled by the user."""


def _initialize_worker(
    stop_event,
    environment_type,
    optimizer,
    console_enabled,
):
    """Share cancellation and selected runtime settings with each worker."""
    global _WORKER_STOP_EVENT
    global _WORKER_CONSOLE_ENABLED
    _WORKER_STOP_EVENT = stop_event
    _WORKER_CONSOLE_ENABLED = console_enabled
    config.ENVIRONMENT_TYPE = environment_type
    config.EXPERIMENT_OPTIMIZER = optimizer


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
    stop_event = _WORKER_STOP_EVENT

    log_filename = (
        f"{experiment_type.value}_{config.ENVIRONMENT_TYPE}_"
        f"{fitness_strategy.value}_debug.log"
    )

    Logger.configure(
        config.LOG_LEVEL,
        filename=log_filename,
        console_enabled=_WORKER_CONSOLE_ENABLED,
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
        stop_event=stop_event,
    )

    algorithm.run()

    diagnostics = None
    if stop_event is None or not stop_event.is_set():
        diagnostics = _summarize_final_population(
            algorithm=algorithm,
            environment_type=config.ENVIRONMENT_TYPE,
            activation=activation,
            evolve_alpha=evolve_alpha,
        )

    return ExperimentRunResult(
        seed=seed,
        metrics=algorithm.metrics,
        diagnostics=diagnostics,
    )


def _to_json_safe(value):
    """Convert NumPy values and non-finite floats to valid JSON values."""
    if isinstance(value, np.ndarray):
        return _to_json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return "NaN"
        if np.isposinf(number):
            return "Infinity"
        if np.isneginf(number):
            return "-Infinity"
        return number
    return value


def _trace_failed_candidate(
    algorithm,
    candidate,
    candidate_index: int,
    environment_type: str,
    activation,
    evolve_alpha: bool,
):
    """Replay a failed final-generation policy and capture its parameters."""
    result = candidate.result
    genes = np.asarray(candidate.genes, dtype=float)
    finite_genes = genes[np.isfinite(genes)]

    details = {
        "candidate_index": candidate_index,
        "failure_type": (
            "timeout"
            if result.truncated or (
                not result.terminated and result.steps >= config.MAX_STEPS
            )
            else "terminal_failure"
        ),
        "fitness": _to_json_safe(candidate.fitness),
        "expected_steps": int(result.steps),
        "expected_endpoint": _to_json_safe(result.final_position),
        "activation": getattr(activation, "value", activation),
        "evolve_alpha": bool(evolve_alpha),
        "parameter_check": {
            "parameter_count": int(genes.size),
            "finite_count": int(np.isfinite(genes).sum()),
            "nan_count": int(np.isnan(genes).sum()),
            "infinite_count": int(np.isinf(genes).sum()),
            "max_absolute_finite_parameter": (
                float(np.max(np.abs(finite_genes)))
                if finite_genes.size
                else None
            ),
            "l2_norm_finite_parameters": _to_json_safe(
                np.linalg.norm(finite_genes)
            ) if finite_genes.size else None,
        },
        "parameters": {},
    }

    parameter_getters = (
        ("input_weights", "get_input_weights"),
        ("hidden_bias", "get_hidden_bias"),
        ("output_weights", "get_output_weights"),
        ("output_bias", "get_output_bias"),
        ("alpha_genes", "get_alpha"),
    )
    for parameter_name, getter_name in parameter_getters:
        getter = getattr(candidate, getter_name, None)
        if callable(getter):
            details["parameters"][parameter_name] = _to_json_safe(getter())

    environment = create_environment(environment_type)
    actions = []
    action_names = {
        "minigrid": {0: "turn_left", 1: "turn_right", 2: "forward"},
        "frozen_lake": {0: "left", 1: "down", 2: "right", 3: "up"},
    }
    total_reward = 0.0
    terminated = truncated = False
    try:
        environment.reset()
        agent = algorithm.agent_creator(candidate)
        agent.reset()

        for step_index in range(config.MAX_STEPS):
            observation = environment.get_local_observation()
            action = int(agent.act(
                observation,
                agent_observation_index=(
                    environment.get_agent_observation_index()
                ),
            ))
            actions.append({
                "id": action,
                "name": action_names.get(environment_type, {}).get(
                    action,
                    "unknown",
                ),
            })
            _, reward, terminated, truncated = environment.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        replay_endpoint = environment.get_agent_position()
        details["replay"] = {
            "actions": actions,
            "action_counts": {
                action_name: sum(
                    item["name"] == action_name for item in actions
                )
                for action_name in sorted({
                    item["name"] for item in actions
                })
            },
            "steps": len(actions),
            "total_reward": _to_json_safe(total_reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "endpoint": _to_json_safe(replay_endpoint),
            "matches_evaluated_result": (
                len(actions) == result.steps
                and tuple(replay_endpoint) == tuple(result.final_position)
                and bool(terminated) == bool(result.terminated)
                and bool(truncated) == bool(result.truncated)
            ),
        }
    except Exception as error:
        details["replay"] = {"error": f"{type(error).__name__}: {error}"}
    finally:
        environment.close()

    return details


def _summarize_final_population(
    algorithm,
    environment_type: str,
    activation=None,
    evolve_alpha: bool = False,
):
    """Summarize failures in the final evaluated population of a run."""
    candidates = getattr(algorithm, "last_generation_candidates", None)
    if not candidates:
        return None

    diagnostics = {
        "population_size": len(candidates),
        "success_count": 0,
        "lava_failures": 0,
        "hole_failures": 0,
        "timeout_failures": 0,
        "other_failures": 0,
        "failure_endpoints": {},
        "failure_candidate_details": [],
    }

    for candidate_index, candidate in enumerate(candidates):
        result = candidate.result
        if result.success:
            diagnostics["success_count"] += 1
            continue

        position = ",".join(str(value) for value in result.final_position)
        endpoints = diagnostics["failure_endpoints"]
        endpoints[position] = endpoints.get(position, 0) + 1

        if result.truncated or (
            not result.terminated
            and result.steps >= config.MAX_STEPS
        ):
            diagnostics["timeout_failures"] += 1
        elif result.terminated and environment_type == "minigrid":
            # On the current MiniGrid map, lava is the only terminal
            # failure; reaching the goal is already counted as success.
            diagnostics["lava_failures"] += 1
        elif result.terminated and environment_type == "frozen_lake":
            diagnostics["hole_failures"] += 1
        else:
            diagnostics["other_failures"] += 1

        diagnostics["failure_candidate_details"].append(
            _trace_failed_candidate(
                algorithm=algorithm,
                candidate=candidate,
                candidate_index=candidate_index,
                environment_type=environment_type,
                activation=activation,
                evolve_alpha=evolve_alpha,
            )
        )

    diagnostics["failure_count"] = (
        diagnostics["population_size"] - diagnostics["success_count"]
    )
    diagnostics["success_rate_percent"] = (
        100 * diagnostics["success_count"] / diagnostics["population_size"]
    )
    return diagnostics

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
        self.last_diagnostics_path: Path | None = None
        self.last_plot_data_path: Path | None = None

    @staticmethod
    def _configured_mlp_experiment(hidden_size: int):
        activation_settings = {
            "relu": (ActivationFunction.RELU, False, "ReLU"),
            "leaky_relu": (
                ActivationFunction.PRELU,
                False,
                "Leaky ReLU (alpha = 0.25)",
            ),
            "prelu": (
                ActivationFunction.PRELU,
                True,
                "PReLU + Alpha",
            ),
        }
        activation_name = config.MLP_ACTIVATION_FUNCTION.strip().lower()
        try:
            activation, evolve_alpha, display_name = (
                activation_settings[activation_name]
            )
        except KeyError as error:
            supported = ", ".join(activation_settings)
            raise ValueError(
                "MLP_ACTIVATION_FUNCTION must be one of: "
                f"{supported}. Got {config.MLP_ACTIVATION_FUNCTION!r}."
            ) from error

        configuration = configure_experiment(
            ExperimentType.MLP,
            activation=activation,
            evolve_alpha=evolve_alpha,
            hidden_size=hidden_size,
        )
        configuration.name = (
            f"MLP + {display_name} | {hidden_size} neurônios"
        )
        return configuration

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
            configuration = self._configured_mlp_experiment(hidden_size)
        else:
            configuration = configure_experiment(self.experiment_type)

        results = self._execute_runs(
            fitness_strategy=FitnessStrategy.REWARD,
            configuration=configuration,
        )
        metrics = ExperimentAnalyzer(results).analyze()
        optimizer = self._optimizer_for(configuration)
        cycle_label = "iteração" if optimizer == "cross_entropy" else "geração"
        plot_title = (
            f"Taxa de sucesso por {cycle_label} | "
            f"ambiente={config.ENVIRONMENT_TYPE} | "
            f"{configuration.name} | {optimizer}"
        )
        self.last_plot_data_path = self._save_single_plot_data(
            metrics=metrics,
            results=results,
            configuration=configuration,
            optimizer=optimizer,
            plot_title=plot_title,
        )
        self.logger.info(
            "Single-mode plot data saved | path=%s",
            self.last_plot_data_path,
        )

        plotter = ExperimentPlotter(metrics, self.experiment_type)
        plotter.plot_experiment(title=plot_title)
        self._save_plot(
            plotter,
            self.last_plot_data_path.with_suffix(".png").name,
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

        configuration = None
        if self.experiment_type == ExperimentType.MLP:
            if not config.MLP_HIDDEN_SIZES:
                raise ValueError(
                    "MLP_HIDDEN_SIZES must contain a hidden size."
                )
            configuration = self._configured_mlp_experiment(
                config.MLP_HIDDEN_SIZES[0]
            )

        for strategy in comparison_strategies:
            results[strategy] = self._execute_runs(
                fitness_strategy=strategy,
                configuration=configuration,
            )

            print(f"\n{'=' * 60}")
            print(f"Fitness strategy: {strategy.value}")
            print(f"{'=' * 60}")

            success_rates = []

            for run_index, run_metrics in enumerate(
                results[strategy],
                start=1
            ):
                final_metrics = run_metrics.metrics[-1]

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
        self._save_plot(
            plotter,
            f"fitness_comparison_{self.experiment_type.value}_"
            f"{config.ENVIRONMENT_TYPE}",
        )
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

        optimizer = self._optimizer_for(configurations[0])
        self.last_plot_data_path = self._save_configuration_comparison_data(
            results=results,
            configurations=configurations,
            fitness_strategy=fitness_strategy,
            optimizer=optimizer,
        )
        self.logger.info(
            "Configuration-comparison plot data saved | path=%s",
            self.last_plot_data_path,
        )

        plotter = ExperimentPlotter(
            results,
            self.experiment_type,
        )
        plotter.plot_configuration_comparison()
        self._save_plot(
            plotter,
            f"configuration_comparison_{config.ENVIRONMENT_TYPE}",
        )
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
            "Experiment started | environment=%s | agent=%s | optimizer=%s | "
            "fitness_strategy=%s | runs=%d | base_seed=%d | workers=%d",
            config.ENVIRONMENT_TYPE,
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
        dashboard = TerminalDashboard(
            agent=agent_name,
            environment=config.ENVIRONMENT_TYPE,
            optimizer=optimizer,
            fitness_strategy=fitness_strategy.value,
            runs=self.runs,
            workers=self.workers,
            enabled=config.LOG_LEVEL.upper() != "DEBUG",
        )

        try:
            dashboard.start()
            interrupted = False
            ordered_results = [None] * len(tasks)
            process_context = get_context("spawn")
            stop_event = process_context.Event()
            with ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=process_context,
                initializer=_initialize_worker,
                initargs=(
                    stop_event,
                    config.ENVIRONMENT_TYPE,
                    config.EXPERIMENT_OPTIMIZER,
                    not dashboard.enabled,
                ),
            ) as executor:
                future_metadata = {}
                next_task_index = 0

                while future_metadata or next_task_index < len(tasks):
                    try:
                        while (
                            not interrupted
                            and next_task_index < len(tasks)
                            and len(future_metadata) < self.workers
                        ):
                            task = tasks[next_task_index]
                            future = executor.submit(_run_experiment, task)
                            future_metadata[future] = {
                                "run_index": next_task_index + 1,
                                "seed": task[0],
                            }
                            next_task_index += 1

                        if not future_metadata:
                            break

                        completed, _ = wait(
                            future_metadata,
                            return_when=FIRST_COMPLETED,
                            timeout=0.2,
                        )
                        for future in list(future_metadata):
                            if future not in completed:
                                continue

                            metadata = future_metadata.pop(future)
                            if future.cancelled():
                                continue

                            run_result = future.result()
                            ordered_results[
                                metadata["run_index"] - 1
                            ] = run_result
                            run_is_complete = (
                                len(run_result.metrics)
                                >= config.GENERATIONS
                            )
                            if run_is_complete and dashboard.enabled:
                                dashboard.update(
                                    metadata["run_index"],
                                    metadata["seed"],
                                    run_result,
                                )
                            if run_is_complete:
                                self.logger.info(
                                    "Experiment run finished | "
                                    "run=%d/%d | seed=%d",
                                    metadata["run_index"],
                                    self.runs,
                                    metadata["seed"],
                                )
                    except KeyboardInterrupt:
                        if not interrupted:
                            interrupted = True
                            stop_event.set()
                            dashboard.set_status("stopping")
                            for future in list(future_metadata):
                                if future.cancel():
                                    future_metadata.pop(future)

            if interrupted:
                dashboard.finish(status="stopped")
                raise ExperimentCancelled

            results = ordered_results
            dashboard.finish()
        except ExperimentCancelled:
            raise
        except Exception:
            dashboard.finish(status="failed")
            self.logger.exception(
                "Experiment failed | environment=%s | agent=%s | optimizer=%s | "
                "elapsed=%.2f seconds",
                config.ENVIRONMENT_TYPE,
                agent_name,
                optimizer,
                time.perf_counter() - started_at,
            )
            raise

        self.logger.info(
            "Experiment finished | environment=%s | agent=%s | optimizer=%s | "
            "fitness_strategy=%s | elapsed=%.2f seconds",
            config.ENVIRONMENT_TYPE,
            agent_name,
            optimizer,
            fitness_strategy.value,
            time.perf_counter() - started_at,
        )

        diagnostics_path = self._save_run_diagnostics(
            results=results,
            agent_name=agent_name,
            fitness_strategy=fitness_strategy,
            optimizer=optimizer,
        )
        self.last_diagnostics_path = diagnostics_path
        if diagnostics_path is not None:
            self.logger.info(
                "Per-seed diagnostics saved | path=%s",
                diagnostics_path,
            )

        return results

    def _save_run_diagnostics(
        self,
        results,
        agent_name: str,
        fitness_strategy: FitnessStrategy,
        optimizer: str,
    ) -> Path | None:
        rows = []
        for run in results:
            if run.diagnostics is None:
                continue

            rows.append({
                "seed": run.seed,
                "environment": config.ENVIRONMENT_TYPE,
                "agent": agent_name,
                "optimizer": optimizer,
                "fitness_strategy": fitness_strategy.value,
                "generations": len(run.metrics),
                "final_generation": run.metrics[-1].generation,
                "population_size": run.diagnostics["population_size"],
                "success_count": run.diagnostics["success_count"],
                "success_rate_percent": run.diagnostics[
                    "success_rate_percent"
                ],
                "failure_count": run.diagnostics["failure_count"],
                "lava_failures": run.diagnostics["lava_failures"],
                "hole_failures": run.diagnostics["hole_failures"],
                "timeout_failures": run.diagnostics["timeout_failures"],
                "other_failures": run.diagnostics["other_failures"],
                "failure_endpoints": json.dumps(
                    run.diagnostics["failure_endpoints"],
                    sort_keys=True,
                ),
                "failure_candidate_details_json": json.dumps(
                    run.diagnostics["failure_candidate_details"],
                    sort_keys=True,
                    allow_nan=False,
                ),
            })

        if not rows:
            return None

        output_directory = Path("experiment_diagnostics")
        output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_directory / (
            f"{self.experiment_type.value}_{config.ENVIRONMENT_TYPE}_"
            f"{fitness_strategy.value}_{optimizer}_{timestamp}.csv"
        )

        with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        return output_path

    def _save_single_plot_data(
        self,
        metrics,
        results,
        configuration,
        optimizer: str,
        plot_title: str,
    ) -> Path:
        """Save all values used by the single-mode graph to a CSV file."""
        run_columns = [
            f"run_{run_index:02d}_success_rate_percent"
            for run_index in range(1, len(results) + 1)
        ]
        context = {
            "experiment_mode": ExperimentMode.SINGLE.value,
            "experiment_type": self.experiment_type.value,
            "environment": config.ENVIRONMENT_TYPE,
            "agent": configuration.name,
            "optimizer": optimizer,
            "fitness_strategy": FitnessStrategy.REWARD.value,
            "runs": len(results),
            "base_seed": self.base_seed,
            "first_run_seed": self.base_seed,
            "last_run_seed": self.base_seed + len(results) - 1,
            "population_size": config.POPULATION_SIZE,
            "configured_generations": config.GENERATIONS,
            "actual_generations": len(metrics),
            "max_steps": config.MAX_STEPS,
            "mlp_activation_function": (
                config.MLP_ACTIVATION_FUNCTION
                if self.experiment_type == ExperimentType.MLP
                else ""
            ),
            "mlp_hidden_size": configuration.hidden_size or "",
            "plot_title": plot_title,
            "plot_xlabel": (
                "Iteração" if optimizer == "cross_entropy" else "Geração"
            ),
            "plot_ylabel": "Taxa de sucesso (%)",
            "plot_series": "Taxa de sucesso",
            "confidence_interval": "95%",
        }
        metric_columns = [
            "generation",
            "mean_success_rate_percent",
            "std_success_rate_percentage_points",
            "margin_of_error_percentage_points",
            "lower_bound_percent",
            "upper_bound_percent",
            "mean_std_success_rate_percentage_points",
            "mean_margin_of_error_percentage_points",
        ]
        fieldnames = [*context, *metric_columns, *run_columns]

        mean_std = statistics.mean([
            metric.std_success_rate * 100 for metric in metrics
        ])
        mean_margin = statistics.mean([
            metric.margin_of_error * 100 for metric in metrics
        ])

        rows = []
        for metric_index, metric in enumerate(metrics):
            row = {
                **context,
                "generation": metric.generation,
                "mean_success_rate_percent": metric.mean_success_rate * 100,
                "std_success_rate_percentage_points": (
                    metric.std_success_rate * 100
                ),
                "margin_of_error_percentage_points": (
                    metric.margin_of_error * 100
                ),
                "lower_bound_percent": metric.lower_bound * 100,
                "upper_bound_percent": metric.upper_bound * 100,
                "mean_std_success_rate_percentage_points": mean_std,
                "mean_margin_of_error_percentage_points": mean_margin,
            }
            for run_index, run in enumerate(results, start=1):
                row[run_columns[run_index - 1]] = (
                    run.metrics[metric_index].success_rate * 100
                )
            rows.append(row)

        output_directory = Path("experiment_results")
        output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_directory / (
            f"single_{self.experiment_type.value}_"
            f"{config.ENVIRONMENT_TYPE}_{optimizer}_{timestamp}.csv"
        )

        with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return output_path

    def _save_configuration_comparison_data(
        self,
        *,
        results,
        configurations,
        fitness_strategy: FitnessStrategy,
        optimizer: str,
    ) -> Path:
        """Save the aggregated curves and per-seed values for MLP comparison."""
        run_count = len(next(iter(results.values())))
        run_columns = [
            f"run_{run_index:02d}_success_rate_percent"
            for run_index in range(1, run_count + 1)
        ]
        context = {
            "experiment_mode": ExperimentMode.CONFIGURATION_COMPARISON.value,
            "experiment_type": self.experiment_type.value,
            "environment": config.ENVIRONMENT_TYPE,
            "optimizer": optimizer,
            "fitness_strategy": fitness_strategy.value,
            "runs": run_count,
            "base_seed": self.base_seed,
            "first_run_seed": self.base_seed,
            "last_run_seed": self.base_seed + run_count - 1,
            "population_size": config.POPULATION_SIZE,
            "configured_generations": config.GENERATIONS,
            "max_steps": config.MAX_STEPS,
            "plot_title": (
                "Comparação de ativações com fitness por distância Manhattan"
            ),
            "plot_xlabel": (
                "Iteração" if optimizer == "cross_entropy" else "Geração"
            ),
            "plot_ylabel": "Taxa de sucesso (%)",
            "confidence_interval": "95%",
        }
        fieldnames = [
            *context,
            "configuration_name",
            "hidden_size",
            "activation",
            "generation",
            "mean_success_rate_percent",
            "std_success_rate_percentage_points",
            "margin_of_error_percentage_points",
            "lower_bound_percent",
            "upper_bound_percent",
            "mean_std_success_rate_percentage_points",
            "mean_margin_of_error_percentage_points",
            *run_columns,
        ]

        rows = []
        for experiment_configuration in configurations:
            configuration_name = experiment_configuration.name
            runs = results[configuration_name]
            metrics = ExperimentAnalyzer(runs).analyze()
            mean_std = statistics.mean(
                metric.std_success_rate * 100 for metric in metrics
            )
            mean_margin = statistics.mean(
                metric.margin_of_error * 100 for metric in metrics
            )
            activation = configuration_name.removeprefix("MLP + ").rsplit(
                " | ", 1
            )[0]

            for metric_index, metric in enumerate(metrics):
                row = {
                    **context,
                    "configuration_name": configuration_name,
                    "hidden_size": experiment_configuration.hidden_size,
                    "activation": activation,
                    "generation": metric.generation,
                    "mean_success_rate_percent": (
                        metric.mean_success_rate * 100
                    ),
                    "std_success_rate_percentage_points": (
                        metric.std_success_rate * 100
                    ),
                    "margin_of_error_percentage_points": (
                        metric.margin_of_error * 100
                    ),
                    "lower_bound_percent": metric.lower_bound * 100,
                    "upper_bound_percent": metric.upper_bound * 100,
                    "mean_std_success_rate_percentage_points": mean_std,
                    "mean_margin_of_error_percentage_points": mean_margin,
                }
                for run_index, run in enumerate(runs, start=1):
                    row[run_columns[run_index - 1]] = (
                        run.metrics[metric_index].success_rate * 100
                    )
                rows.append(row)

        output_directory = Path("experiment_results")
        output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_directory / (
            f"configuration_comparison_mlp_{config.ENVIRONMENT_TYPE}_"
            f"{optimizer}_{timestamp}.csv"
        )
        with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return output_path

    def _save_plot(self, plotter, filename: str) -> Path:
        """Save every generated experiment chart to the shared plots folder."""
        output_directory = Path("experiment_results") / "plots"
        output_directory.mkdir(parents=True, exist_ok=True)
        if not filename.endswith(".png"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{filename}_{timestamp}.png"
        output_path = output_directory / filename
        plotter.save(output_path)
        self.logger.info("Experiment plot saved | path=%s", output_path)
        return output_path

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
