from algorithms import GeneticAlgorithm
from algorithms import Population
from algorithms import FitnessEvaluator
import config

population = Population.random(
    size=config.POPULATION_SIZE,
    genome_length=config.GENOME_LENGTH
)

fitness_evaluator = FitnessEvaluator()

algorithm = GeneticAlgorithm(
    population=population,
    fitness_evaluator=fitness_evaluator,
    generations=config.GENERATIONS,
)

algorithm.run()