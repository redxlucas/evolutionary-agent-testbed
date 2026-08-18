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
    tournament_size=config.TOURNAMENT_SIZE,
    selection_amount=config.SELECTION_AMOUNT
)

algorithm.run()

for i in range(len(algorithm.metrics)):
    print(f"""Geração: {algorithm.metrics[i].generation}
Melhor Fitness: {algorithm.metrics[i].best_fitness}
Média Fitness: {algorithm.metrics[i].average_fitness}
Pior Fitness: {algorithm.metrics[i].worst_fitness}
% de Sucesso: {algorithm.metrics[i].success_rate}
    """)