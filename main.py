from environment.frozen_lake_environment import FrozenLakeEnvironment

environment = FrozenLakeEnvironment()

state = environment.reset()

done = False

while not done:

    action = environment.env.action_space.sample()

    state, reward, done = environment.step(action)

environment.close()