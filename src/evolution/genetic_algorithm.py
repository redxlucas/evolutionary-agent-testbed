from evolution import FitnessEvaluator
from evolution import Population
from environment import FrozenLakeEnvironment
from agents import *
from simulation import Simulation

class GeneticAlgorithm:

    def __init__(
        self,
        population: Population,
        fitness_evaluator: FitnessEvaluator,
        generations: int,
    ):
        self.population = population
        self.fitness_evaluator = fitness_evaluator
        self.generations = generations

    def run(self):
        for generation in range(self.generations):

            print(f"\nGeração: {generation}")

            for agent in self.population.individuals:

                environment = FrozenLakeEnvironment()

                simulation = Simulation(
                    agent=agent, 
                    environment=environment
                )

                result = simulation.run()

                fitness = self.fitness_evaluator.evaluate(result)

                print(f"Fitness: {round(fitness, 2)}      | Passos: {result.steps}     | Truncado? {result.truncated}")
        
