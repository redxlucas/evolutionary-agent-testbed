from environment.frozen_lake_environment import FrozenLakeEnvironment

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

running = True

while running and not done:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            action = None

            if event.key == pygame.K_LEFT:
                action = 0

            elif event.key == pygame.K_DOWN:
                action = 1

            elif event.key == pygame.K_RIGHT:
                action = 2

            elif event.key == pygame.K_UP:
                action = 3

            if action is not None:

                print(f"\nAção escolhida: {ACTION_NAMES[action]}")

                previous_state = state

                state, reward, done = environment.step(action)

                print(f"Estado anterior: {previous_state}")
                print(f"Novo estado: {state}")
                print(f"Reward: {reward}")
                print(f"Done: {done}")

                if done:
                    print("\nFim do episódio!")

pygame.quit()
environment.close()