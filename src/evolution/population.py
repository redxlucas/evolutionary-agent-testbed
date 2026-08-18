from agents import Agent
from agents import Genome


class Population:
    
    def __init__(self, individuals: list[Agent]):
        self.individuals = individuals

    def __len__(self):
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)

    @classmethod
    def random(cls, size: int, genome_length: int):
        individuals = [Agent(genome=Genome.random(genome_length)) for _ in range(size)]

        return cls(individuals)