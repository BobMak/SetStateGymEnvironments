import numpy as np
from gym import utils
from gym.envs.mujoco import ReacherEnv


class MyReacherEnv(ReacherEnv):

    def set_state(self, qpos, qvel):
        """
        :param state: np.array([])
        :return: updated state
        """
        super().set_state(qpos, qvel)

        return self.get_state()

    def get_state(self):
        return super()._get_obs()