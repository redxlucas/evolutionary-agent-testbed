import random

import numpy as np

from agents.adaline import Adaline
import config
from environment import FrozenLakeEnvironment
from evolution import GeneticAlgorithm
from evolution import FitnessEvaluator
from evolution.population import Population
from simulation.simulation import Simulation
from visualization.generation_plotter import GenerationPlotter

random.seed(42)
np.random.seed(42)

population = Population.random(
    size=config.POPULATION_SIZE,
    genome_length=config.GENOME_LENGTH,
)

algorithm = GeneticAlgorithm(
    population=population,
    fitness_evaluator=FitnessEvaluator(),
    generations=config.GENERATIONS,
    tournament_size=config.TOURNAMENT_SIZE,
    selection_amount=config.SELECTION_AMOUNT,
    crossover_rate=config.CROSSOVER_RATE,
    mutation_rate=config.MUTATION_RATE,
    elite_size=config.ELITE_SIZE,
)

algorithm.run()

plotter = GenerationPlotter(algorithm.metrics)

plotter.plot_fitness()
plotter.show()

# genome = population.genomes[5]

# adaline = Adaline(
#     weights=genome.get_weights(),
#     bias=genome.get_bias()
# )

# environment = FrozenLakeEnvironment()

# simulation = Simulation(adaline, environment)

# simulation.run_debug()

# algorithm = GeneticAlgorithm(
#     population=population,
#     fitness_evaluator=FitnessEvaluator(),
#     generations=config.GENERATIONS,
#     tournament_size=config.TOURNAMENT_SIZE,
#     selection_amount=config.SELECTION_AMOUNT,
#     crossover_rate=config.CROSSOVER_RATE,
#     mutation_rate=config.MUTATION_RATE,
#     elite_size=config.ELITE_SIZE,
# )

# algorithm._evaluate_population()

# for i, genome in enumerate(population.genomes):
#     result = genome.result

#     print(
#         f"Fitness={genome.fitness} | "
#         f"Reward={result.total_reward} | "
#         f"Steps={result.steps} | "
#         f"Final={result.final_position} | "
#         f"Goal={result.goal_position} | "
#         f"Terminated={result.terminated} | "
#         f"Truncated={result.truncated}"
#     )


