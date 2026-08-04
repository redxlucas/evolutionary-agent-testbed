import gymnasium as gym
import minigrid
import pygame

TILE_SIZE = 64

env = gym.make("MiniGrid-Empty-8x8-v0")
obs, info = env.reset()

pygame.init()

grid = env.unwrapped.grid

screen = pygame.display.set_mode(
    (grid.width * TILE_SIZE, grid.height * TILE_SIZE)
)

grass = pygame.image.load("assets/grass.png")
wall = pygame.image.load("assets/wall.png")
player = pygame.image.load("assets/player.png")
goal = pygame.image.load("assets/goal.png")

grass = pygame.transform.scale(grass, (TILE_SIZE, TILE_SIZE))
wall = pygame.transform.scale(wall, (TILE_SIZE, TILE_SIZE))
player = pygame.transform.scale(player, (TILE_SIZE, TILE_SIZE))
goal = pygame.transform.scale(goal, (TILE_SIZE, TILE_SIZE))

running = True
clock = pygame.time.Clock()

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # apenas para demonstrar o agente andando
    env.step(env.action_space.sample())

    screen.fill((0, 0, 0))

    grid = env.unwrapped.grid

    for y in range(grid.height):
        for x in range(grid.width):

            screen.blit(grass, (x * TILE_SIZE, y * TILE_SIZE))

            obj = grid.get(x, y)

            if obj is None:
                continue

            if obj.type == "wall":
                screen.blit(wall, (x * TILE_SIZE, y * TILE_SIZE))

            elif obj.type == "goal":
                screen.blit(goal, (x * TILE_SIZE, y * TILE_SIZE))

    ax, ay = env.unwrapped.agent_pos
    screen.blit(player, (ax * TILE_SIZE, ay * TILE_SIZE))

    pygame.display.flip()

    clock.tick(30)  # 2 FPS = 2 movimentos por segundo

pygame.quit()