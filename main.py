
import random

import numpy as np

import config
from evolution import GeneticAlgorithm
from evolution import FitnessEvaluator
from evolution import Population
from experiment.experiment_analyzer import ExperimentAnalyzer
from experiment.experiment_runner import ExperimentRunner
from utils import Logger
from visualization import ExperimentPlotter

def main():
    Logger.configure()

    runner = ExperimentRunner(
        runs=config.EXPERIMENT_RUNS,
        algorithm_factory=create_algorithm,
        base_seed=config.SEED
    )

    results = runner.run()

    analyzer = ExperimentAnalyzer(results)
    metrics = analyzer.analyze()

    experiment_plotter = ExperimentPlotter(metrics)

    experiment_plotter.plot_experiment()
    experiment_plotter.show()

    # plotter = ExperimentPlotter(metrics)
    # plotter.plot_fitness()
    # plotter.show()

def create_algorithm(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    population = Population.random(
        size=config.POPULATION_SIZE,
        genome_length=config.GENOME_LENGTH,
    )

    return GeneticAlgorithm(
        population=population,
        fitness_evaluator=FitnessEvaluator(),
        generations=config.GENERATIONS,
        tournament_size=config.TOURNAMENT_SIZE,
        selection_amount=config.SELECTION_AMOUNT,
        crossover_rate=config.CROSSOVER_RATE,
        mutation_rate=config.MUTATION_RATE,
        elite_size=config.ELITE_SIZE,
    )

if __name__ == "__main__":
    main()