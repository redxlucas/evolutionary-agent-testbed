from agents import Agent
from agents import Genome


class Population:
    
    def __init__(self, individuals: list):
        self.individuals = individuals

    @classmethod
    def random(cls, size: int, genome_length: int):
        agents = [Agent(genome=Genome.random(genome_length)) for _ in range(size)]

        return cls(agents)