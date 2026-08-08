from agents import *
from simulation import Simulation
from environment import FrozenLakeEnvironment

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