import random
import time

import gym
import numpy as np

import gym_lunar_lander_set_state
from gym_lunar_lander_set_state.envs import MyLunarLander


print(gym.__version__)
env = gym.make('LunarLanderSetState-v0')
env.reset()


def get_random_state():
    r = MyLunarLander.lo_clip, MyLunarLander.hi_clip
    vec = np.random.uniform(low=r[0], high=r[1], size=env.observation_space.shape)
    return vec


print("position set state")
time.sleep(0.3)
for x in range(5):
    print(env.set_state(get_random_state()))
    env.render()
    time.sleep(1.0)
    for x in range(30):
        env.render()
        env.step(env.action_space.sample()) # take a random action

env.close()