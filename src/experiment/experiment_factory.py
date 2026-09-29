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

from .experiment_config import ExperimentConfig

def configure_experiment(
    experiment: str
) -> ExperimentConfig:

    if experiment == "ADALINE":
        return ExperimentConfig(
            genome_factory=lambda: AdalineGenome.random(
                config.GENOME_LENGTH
            ),
            agent_creator=AdalineAgent.create,
            evolve=False
        )

    if experiment == "MOVEMENT":
        return ExperimentConfig(
            genome_factory=lambda: MovementGenome.random(
                config.GENOME_LENGTH
            ),
            agent_creator=MovementAgent.create,
            evolve=True
        )

    raise ValueError(
        f"Unknown experiment type: {experiment}"
    )


def create_algorithm(
    seed: int,
    experiment: ExperimentConfig,
    fitness_strategy=None
):

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
        elite_size=config.ELITE_SIZE,
        evolve=experiment.evolve
    )