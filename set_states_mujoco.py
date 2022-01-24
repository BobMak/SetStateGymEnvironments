import time

import gym
import numpy as np

import gym_set_state
from gym_set_state.envs import MyReacherEnv, MyHopperEnv


print(gym.__version__)
env_name = 'ReacherSetState-v2'
# env_name = 'HopperSetState-v2'

if env_name == 'ReacherSetState-v2':
    MyEnv = MyReacherEnv
elif env_name == 'HopperSetState-v2':
    MyEnv = MyHopperEnv

env = gym.make(env_name)
env.reset()


print("position set state")
time.sleep(0.3)
for x in range(50):
    qpos = np.random.rand(env.model.nq) * 2 - 1
    qval = np.random.rand(env.model.nv) * 2 - 1
    res_before_sim = env.set_state(qpos, qval)
    print("state", res_before_sim)
    env.render()
    time.sleep(1.0)
    for x in range(120):
        env.render()
        env.step(env.action_space.sample())  # take a random action

env.close()