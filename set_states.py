import random
import time

import gym
import gym_lunar_lander_set_state


print(gym.__version__)
env = gym.make('LunarLanderSetState-v0')
env.reset()

for x in range(50):
    env.render()
    env.step(env.action_space.sample()) # take a random action

print("horizonal position set state")
time.sleep(0.3)
for x in range(5):
    env.set_state(random.uniform(2.0, 10.0),
                  random.uniform(6.0, 10.0),
                  0.001)
    for x in range(50):
        env.render()
        env.step(env.action_space.sample()) # take a random action

print("rotate")
for x in range(6):
    env.set_state(7.0, 6.0, 0.8*x)
    for _ in range(20):
        env.render()
        env.step(env.action_space.sample())  # take a random action

print("apply force to center")
env.render()
env.set_state(7.0, 6.0, .0, center_force=[.0, 2000.0])
for x in range(50):
    env.render()
    env.step(env.action_space.sample()) # take a random action

print("")

env.close()