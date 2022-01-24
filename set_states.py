import random
import time

import gym
import numpy as np

import gym_set_state
from gym_set_state.envs import MyLunarLander, MyBipedalWalker


print(gym.__version__)
env_name = 'BipedalWalkerSetState-v0'
# env_name = 'LunarLanderSetState-v0'

if env_name == 'LunarLanderSetState-v0' or env_name == 'LunarLanderContinuousSetState-v0':
    MyEnv = MyLunarLander
elif env_name == 'BipedalWalkerSetState-v0' or env_name == 'BipedalWalkerHardcoreSetState-v0':
    MyEnv = MyBipedalWalker

env = gym.make(env_name)
env.reset()


def get_random_state():
    r = MyEnv.lo_clip, MyEnv.hi_clip
    vec = np.random.uniform(low=r[0], high=r[1], size=env.observation_space.shape)
    return vec


print("position set state")
time.sleep(0.3)
for x in range(50):
    set_to = env.observation_space.sample()
    res_before_sim = env.set_state(set_to)
    print("state", res_before_sim)
    env.render()
    time.sleep(1.5)
    for x in range(30):
        env.render()
        env.step(env.action_space.sample())  # take a random action

env.close()