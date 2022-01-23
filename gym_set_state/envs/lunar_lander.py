import numpy as np
from Box2D import b2Vec2
from gym.envs.box2d import LunarLander
from gym.envs.box2d.lunar_lander import VIEWPORT_H, VIEWPORT_W, SCALE, LEG_DOWN, FPS, LEG_AWAY


class MyLunarLander(LunarLander):
    lo_clip = np.array([-1.0, 0.0, -2.0, -2.0, -np.pi / 2, -6.0, 0., 0.])
    hi_clip = np.array([1.0, 1.4, 2.0, 2.0, np.pi / 2, 6.0, 1., 1.])

    def set_state(self, state):
        """
        :param state: np.array([x, y, x_vel, y_vel, angle, angular_velocity, contact_left, contact_right])
        :return: updated state
        """
        state = np.clip(state, self.lo_clip, self.hi_clip)
        x = state[0] * (VIEWPORT_W / SCALE / 2) + VIEWPORT_W / SCALE / 2
        y = state[1] * (VIEWPORT_H / SCALE / 2) + self.helipad_y + LEG_DOWN / SCALE

        # legs
        i = -1  # -1 is for the left, 1 is for the right leg
        for leg in self.legs:
            init_leg_pos = np.array([-i * LEG_AWAY / SCALE, 0])
            leg_ang = state[4]
            rot_matrix = np.array([[np.cos(leg_ang), -np.sin(leg_ang)],
                                   [np.sin(leg_ang), np.cos(leg_ang)]])
            leg_pos = np.matmul(rot_matrix, init_leg_pos) + np.array([x, y])

            leg.position = b2Vec2(leg_pos[0], leg_pos[1])
            leg.angle = leg_ang + 0.05 * -i
            leg.linearVelocity = b2Vec2(state[2], state[3])
            leg.angularVelocity = state[5] * FPS / 20.0
            i+=2

        # lander
        self.lander.position = b2Vec2(x, y)
        self.lander.linearVelocity = b2Vec2(
            state[2] * FPS / (VIEWPORT_W / SCALE / 2),
            state[3] * FPS / (VIEWPORT_H / SCALE / 2)
        )
        self.lander.angle = state[4]
        self.lander.angularVelocity = state[5] * FPS / 20.0

        # evaluate leg states
        self.world.Step(1 / FPS,1,1)
        # self.world.Step(0,0,0)

        return self.get_state()

    def get_state(self):
        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]

        return np.array(state, dtype=np.float32)