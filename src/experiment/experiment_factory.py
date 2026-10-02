import random

import numpy as np

import config

from agents.genome import MovementGenome
from agents.genome import AdalineGenome
from agents import AdalineAgent
from agents import MovementAgent

from evolution import GeneticAlgorithm
from evolution import FitnessEvaluator
from evolution import Population

from .experiment_types import ExperimentType
from utils import Logger

from .experiment_config import ExperimentConfig

def configure_experiment(
    experiment_type: ExperimentType
) -> ExperimentConfig:
    """
    Creates the configuration associated with an experiment type.

    The configuration defines the genome factory, agent creator, and
    whether genetic operators should be applied during evolution.
    """

    if experiment_type == ExperimentType.ADALINE:
        return ExperimentConfig(
            genome_factory=lambda: AdalineGenome.random(
                AdalineGenome.GENOME_LENGTH
            ),
            agent_creator=AdalineAgent.create,
            evolve=True
        )

    if experiment_type == ExperimentType.MOVEMENT:
        return ExperimentConfig(
            genome_factory=lambda: MovementGenome.random(
                config.GENOME_LENGTH
            ),
            agent_creator=MovementAgent.create,
            evolve=True
        )

    raise ValueError(
        f"Unknown experiment type: {experiment_type}"
    )


def create_algorithm(
    seed: int,
    experiment: ExperimentConfig,
    logger: Logger,
    fitness_strategy=None
):
    """
    Creates and configures the genetic algorithm for an experiment.

    Initializes the random number generators, creates the initial
    population and fitness evaluator, and assembles the genetic
    algorithm using the provided experiment configuration.
    """

    random.seed(seed)
    np.random.seed(seed)

    population = Population.random(
        size=config.POPULATION_SIZE,
        genome_factory=experiment.genome_factory
    )

    fitness_evaluator = FitnessEvaluator(
        strategy=fitness_strategy
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
        evolve=experiment.evolve
    )