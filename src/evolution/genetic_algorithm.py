from evolution import FitnessEvaluator
from evolution import Population
from environment import FrozenLakeEnvironment
from agents import *
from evolution import Selection
from simulation import Simulation

class GeneticAlgorithm:

    def __init__(
        self,
        population: Population,
        fitness_evaluator: FitnessEvaluator,
        generations: int,
        selection_size
    ):
        self.population = population
        self.fitness_evaluator = fitness_evaluator
        self.generations = generations
        self.selection = Selection(
            selection_size=selection_size
        )

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

                agent.fitness = self.fitness_evaluator.evaluate(result)

                print(f"Fitness: {round(agent.fitness, 2)}      | Passos: {result.steps}     | Truncado? {result.truncated}")

            parents = self.selection.select(
                self.population.individuals,
                amount=3
            )

            print(parents.individuals)
