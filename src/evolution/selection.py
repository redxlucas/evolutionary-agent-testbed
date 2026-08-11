import random

from agents import Agent
from evolution import Population


class Selection:

    def __init__(self, selection_size: int):
        self.selection_size = selection_size

    def select(self, population: Population, amount: int) -> Population: # seleção baseada no tournament selection
        if self.selection_size > len(population):
            raise ValueError(
                "Tournament size cannot be greater than population size."
        )

        if amount <= 0:
            raise ValueError(
                "Selection amount must be greater than zero."
        )

        selected = []

        for _ in range(amount):
            tournament = random.sample(
                population,
                self.selection_size
            )

            winner = max(
                tournament,
                key=lambda agent: agent.fitness
            )

            selected.append(winner)

        return Population(selected)

