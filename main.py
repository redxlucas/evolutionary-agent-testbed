from src.agents.agent import Agent
from src.agents.genome import Genome
from src.simulation.simulation import Simulation
from src.environment.frozen_lake_environment import FrozenLakeEnvironment

environment = FrozenLakeEnvironment()

ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}

agent = Agent(genome=Genome.random(10))
simulation = Simulation(agent=agent, environment=environment)

fitness = simulation.run()

print(f"Fitness: {fitness}")