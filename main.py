from evolution import GeneticAlgorithm
from evolution import Population
from evolution import FitnessEvaluator
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
    selection_size=config.SELECTION_SIZE,
)

algorithm.run()