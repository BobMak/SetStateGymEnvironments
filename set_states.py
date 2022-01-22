import random
import time

import gym

# from gym import error
# from gym.version import VERSION as __version__
#
# from gym.core import (
#     Env,
#     Wrapper,
#     ObservationWrapper,
#     ActionWrapper,
#     RewardWrapper,
# )
# from gym.spaces import Space
# from gym.envs import make, spec, register
# from gym import logger
# from gym import vector
# from gym import wrappers
#
# __all__ = ["Env", "Space", "Wrapper", "make", "spec", "register"]

import gym_lunar_lander_set_state


print(gym.__version__)
env = gym.make('LunarLanderSetState-v0')
# env = gym.make('LunarLander-v2')
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
        # env.set_state(1.0 + 0.3*x, None, None, [.0,.0])

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