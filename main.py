from src.agents.agent import Agent
from src.agents.genome import Genome
from src.environment.frozen_lake_environment import FrozenLakeEnvironment

import pygame

environment = FrozenLakeEnvironment()

state = environment.reset()

done = False

pygame.init()

ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}

agent = Agent(genome=Genome.random(10))

print(agent.genome.genes)

running = True

while not done:

    previous_state = state

    acao = agent.act()

    print(f"\nAção escolhida: {ACTION_NAMES[acao]}")

    state, reward, done = environment.step(acao)

    print(f"Estado anterior: {previous_state}")
    print(f"Novo estado: {state}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")

    input("")

pygame.quit()
environment.close()