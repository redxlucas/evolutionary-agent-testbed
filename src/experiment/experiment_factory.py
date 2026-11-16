import random

import numpy as np

import config

from agents import (
    AdalineAgent,
    MovementAgent,
    MLPAgent,
)

from agents.genome import (
    AdalineGenome,
    MovementGenome,
)

from agents.genome.mlp_genome import MLPGenome

from evolution import (
    GeneticAlgorithm,
    FitnessEvaluator,
    Population,
)
from environment import get_environment_action_count

from optimization import (
    CrossEntropyOptimizer,
    DifferentialEvolutionOptimizer,
)

from utils import Logger

from .experiment_config import ExperimentConfig
from .experiment_types import ActivationFunction, ExperimentType


def configure_experiment(
    experiment_type: ExperimentType,
    activation: ActivationFunction = ActivationFunction.NONE,
    evolve_alpha: bool = False,
    hidden_size: int | None = None,
) -> ExperimentConfig:
    """
    Creates the configuration associated with an experiment type.

    The configuration defines the genome factory, agent creator,
    and whether genetic operators should be applied during evolution.
    """

    if experiment_type == ExperimentType.ADALINE:
        output_size = get_environment_action_count()
        return ExperimentConfig(
            name="ADALINE",
            genome_factory=lambda: AdalineGenome.random(
                num_outputs=output_size,
            ),
            agent_creator=AdalineAgent.create,
            evolve=True,
            activation=ActivationFunction.NONE,
            evolve_alpha=False,
            output_size=output_size,
        )

    if experiment_type == ExperimentType.MLP:
        hidden_size = hidden_size or MLPGenome.HIDDEN_SIZE
        output_size = get_environment_action_count()
        return ExperimentConfig(
            name="MLP",
            genome_factory=lambda: MLPGenome.random(
                hidden_size=hidden_size,
                output_size=output_size,
            ),
            agent_creator=lambda genome: MLPAgent.create(
                genome,
                activation=activation,
                evolve_alpha=evolve_alpha,
            ),
            evolve=True,
            activation=activation,
            evolve_alpha=evolve_alpha,
            hidden_size=hidden_size,
            output_size=output_size,
        )

    if experiment_type == ExperimentType.MOVEMENT:
        action_count = get_environment_action_count()
        return ExperimentConfig(
            name="MOVEMENT",
            genome_factory=lambda: MovementGenome.random(
                config.GENOME_LENGTH,
                action_count=action_count,
            ),
            agent_creator=MovementAgent.create,
            evolve=True,
            activation=ActivationFunction.NONE,
            evolve_alpha=False,
            output_size=action_count,
        )

    raise ValueError(
        f"Unknown experiment type: {experiment_type}"
    )

ADALINE_CONFIG = configure_experiment(
    ExperimentType.ADALINE
)

MLP_RELU_CONFIG = configure_experiment(
    ExperimentType.MLP,
    activation=ActivationFunction.RELU,
)
MLP_RELU_CONFIG.name = "MLP + ReLU"

MLP_PRELU_CONFIG = configure_experiment(
    ExperimentType.MLP,
    activation=ActivationFunction.PRELU,
)
MLP_PRELU_CONFIG.name = "MLP + PReLU"

MLP_PRELU_ALPHA_CONFIG = configure_experiment(
    ExperimentType.MLP,
    activation=ActivationFunction.PRELU,
    evolve_alpha=True,
)
MLP_PRELU_ALPHA_CONFIG.name = "MLP + PReLU + Alpha"

MOVEMENT_CONFIG = configure_experiment(
    ExperimentType.MOVEMENT
)


def create_algorithm(
    seed: int,
    experiment: ExperimentConfig,
    logger: Logger,
    fitness_strategy=None,
    stop_event=None,
):
    """
    Creates the optimization algorithm configured for the experiment.
    """

    random.seed(seed)
    np.random.seed(seed)

    fitness_evaluator = FitnessEvaluator(
        strategy=fitness_strategy
    )

    if (
        experiment.name in {"ADALINE", "MOVEMENT"}
        or config.EXPERIMENT_OPTIMIZER == "genetic"
    ):
        return _create_genetic_algorithm(
            experiment=experiment,
            fitness_evaluator=fitness_evaluator,
            logger=logger,
            stop_event=stop_event,
        )

    if config.EXPERIMENT_OPTIMIZER == "cross_entropy":
        return _create_cross_entropy_optimizer(
            experiment=experiment,
            fitness_evaluator=fitness_evaluator,
            logger=logger,
            seed=seed,
            stop_event=stop_event,
        )

    if config.EXPERIMENT_OPTIMIZER == "differential_evolution":
        return _create_differential_evolution_optimizer(
            experiment=experiment,
            fitness_evaluator=fitness_evaluator,
            logger=logger,
            seed=seed,
            stop_event=stop_event,
        )

    raise ValueError(
        f"Unknown optimizer: {config.EXPERIMENT_OPTIMIZER}"
    )


def _create_genetic_algorithm(
    experiment: ExperimentConfig,
    fitness_evaluator: FitnessEvaluator,
    logger: Logger,
    stop_event=None,
):
    """
    Creates the existing Genetic Algorithm.
    """

    population = Population.random(
        size=config.POPULATION_SIZE,
        genome_factory=experiment.genome_factory,
    )

    return GeneticAlgorithm(
        population=population,
        fitness_evaluator=fitness_evaluator,
        agent_creator=experiment.agent_creator,
        generations=config.GENERATIONS,
        tournament_size=config.TOURNAMENT_SIZE,
        selection_amount=config.SELECTION_AMOUNT,
        crossover_rate=config.CROSSOVER_RATE,
        mutation_rate=config.MUTATION_RATE,
        logger=logger,
        elite_size=config.ELITE_SIZE,
        evolve=experiment.evolve,
        stop_event=stop_event,
    )


def _create_cross_entropy_optimizer(
    experiment: ExperimentConfig,
    fitness_evaluator: FitnessEvaluator,
    logger: Logger,
    seed: int,
    stop_event=None,
):
    """
    Creates the Cross-Entropy optimizer.

    CEM uses continuous parameter vectors instead of the
    discrete genomes used by the Genetic Algorithm.
    """

    if experiment.hidden_size is None:
        raise ValueError(
            "Cross-Entropy optimization currently supports "
            "MLP experiments only."
        )

    genome_type = MLPGenome
    hidden_size = experiment.hidden_size or MLPGenome.HIDDEN_SIZE
    output_size = experiment.output_size or MLPGenome.DEFAULT_OUTPUT_SIZE

    return CrossEntropyOptimizer(
        parameter_count=genome_type.genome_length_for(
            hidden_size,
            output_size,
        ),
        population_size=config.POPULATION_SIZE,
        elite_size=config.ELITE_SIZE,
        iterations=config.GENERATIONS,
        fitness_evaluator=fitness_evaluator,
        agent_creator=experiment.agent_creator,
        genome_creator=lambda parameters: genome_type(
            parameters,
            hidden_size=hidden_size,
            output_size=output_size,
        ),
        logger=logger,
        initial_std=config.CEM_INITIAL_STD,
        min_std=config.CEM_MIN_STD,
        seed=seed,
        stop_event=stop_event,
    )


def _create_differential_evolution_optimizer(
    experiment: ExperimentConfig,
    fitness_evaluator: FitnessEvaluator,
    logger: Logger,
    seed: int,
    stop_event=None,
):
    """Creates the Differential Evolution optimizer for MLP experiments."""
    if experiment.hidden_size is None:
        raise ValueError(
            "Differential Evolution optimization currently supports "
            "MLP experiments only."
        )

    genome_type = MLPGenome
    hidden_size = experiment.hidden_size
    output_size = experiment.output_size or MLPGenome.DEFAULT_OUTPUT_SIZE

    return DifferentialEvolutionOptimizer(
        parameter_count=genome_type.genome_length_for(
            hidden_size,
            output_size,
        ),
        population_size=config.POPULATION_SIZE,
        generations=config.GENERATIONS,
        fitness_evaluator=fitness_evaluator,
        agent_creator=experiment.agent_creator,
        genome_creator=lambda parameters: genome_type(
            parameters,
            hidden_size=hidden_size,
            output_size=output_size,
        ),
        logger=logger,
        gene_bounds=(genome_type.GENE_MIN, genome_type.GENE_MAX),
        mutation_factor_range=config.DE_MUTATION_FACTOR_RANGE,
        crossover_rate_range=config.DE_CROSSOVER_RATE_RANGE,
        seed=seed,
        stop_event=stop_event,
    )
