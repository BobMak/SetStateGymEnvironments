import math

import numpy as np
from Box2D import b2Vec2
import Box2D
from Box2D.b2 import (
    edgeShape,
    circleShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    contactListener,
)

from gym.envs.box2d import BipedalWalker
from gym.envs.box2d.bipedal_walker import VIEWPORT_W, VIEWPORT_H, SCALE, LEG_DOWN, FPS, LEG_H, LIDAR_RANGE, SPEED_HIP, \
    SPEED_KNEE, LEG_FD, MOTORS_TORQUE, LOWER_FD, TERRAIN_STEP, TERRAIN_STARTPAD, TERRAIN_HEIGHT


class MyBipedalWalker(BipedalWalker):
    lo_clip = np.array([0.0, -1.0, -1.0, -1.0, -np.pi / 2, -1.0, -np.pi / 2, -1.0, 0.0, -np.pi / 2, -1.0,-np.pi / 2, -1.0, 0.0, ]
                       + [0.0] * 10)
    hi_clip = np.array([ np.pi / 2,  1.0,  1.0,  1.0,  np.pi / 2,  1.0,  np.pi / 2, 1.0,  1.0,  np.pi / 2, 1.0,  np.pi / 2,  1.0, 1.0, ]
                       + [1.0] * 10)

    knee_pos = (0,0)

    def set_state(self, state):
        """
        :param state: np.array([
            self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
            self.hull.angularVelocity,
            self.hull.linearVelocity.x,
            self.hull.linearVelocity.y,
            self.joints[0].angle,
            self.joints[0].speed,
            self.joints[1].angle,
            self.joints[1].speed,
            self.legs[1].ground_contact, (ignored; set by the simulator)
            self.joints[2].angle,
            self.joints[2].speed,
            self.joints[3].angle,
            self.joints[3].speed,
            self.legs[3].ground_contact, (ignored; set by the simulator)
        ]])
        :return: updated state
        """
        # state = np.clip(state, self.lo_clip, self.hi_clip).astype(np.float32)
        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.scroll = 0.0

        # legs
        # hull
        self.hull.position = b2Vec2(init_x, init_y)
        self.hull.angle = state[0].item()  # Normal angles up to 0.5 here, but sure more is possible.
        self.hull.angularVelocity = state[1].item() * FPS / 2.0
        self.hull.linearVelocity.x = state[2] / 0.3 / (VIEWPORT_W / SCALE) * FPS  # Normalized to get -1..1 range
        self.hull.linearVelocity.y = state[3] / 0.3 / (VIEWPORT_H / SCALE) * FPS  # Normalized to get -1..1 range

        # hip 1 angle and velocity
        target_ang = state[4]
        self.joints[0].bodyB.angle = target_ang - 0.05 + self.joints[0].bodyA.angle
        self.joints[0].bodyB.angularVelocity = state[5] * SPEED_HIP \
                                               + self.joints[0].bodyA.angularVelocity

        # hip 1 position
        hip_start_pos = self.hull.position - LEG_DOWN * np.array([np.cos(self.hull.angle-np.pi/2), np.sin(self.hull.angle-np.pi/2)])
        self.joints[0].bodyB.position = hip_start_pos + LEG_H / 2 * np.array([
            np.cos(self.joints[0].bodyB.angle-np.pi/2),
            np.sin(self.joints[0].bodyB.angle-np.pi/2)])

        # low 1
        target_ang = state[6] - 1.0
        self.joints[1].bodyB.angle = target_ang + self.joints[1].bodyA.angle
        self.joints[1].bodyB.angularVelocity = state[7] * SPEED_KNEE \
                                               + self.joints[1].bodyA.angularVelocity

        knee_pos = self.joints[1].bodyA.position + LEG_H / 2 * np.array([
            np.cos(self.joints[1].bodyA.angle-np.pi/2),
            np.sin(self.joints[1].bodyA.angle-np.pi/2)])

        self.joints[1].bodyB.position = knee_pos + LEG_H / 2 * np.array([
            np.cos(self.joints[1].bodyB.angle-np.pi/2),
            np.sin(self.joints[1].bodyB.angle-np.pi/2)])

        # hip 2 angle and veolcity
        target_ang = state[9]
        self.joints[2].bodyB.angle = target_ang + 0.05 + self.joints[2].bodyA.angle
        self.joints[2].bodyB.angularVelocity = state[10] * SPEED_HIP \
                                               + self.joints[2].bodyA.angularVelocity

        # hip 2 position
        self.joints[2].bodyB.position = hip_start_pos + LEG_H / 2 * np.array([
            np.cos(self.joints[2].bodyB.angle-np.pi/2),
            np.sin(self.joints[2].bodyB.angle-np.pi/2)])
        # np.matmul(rot_matrix, np.array([LEG_H / 2, 0])) \

        # low 2 angle and velocity
        target_ang = state[11] - 1.0
        self.joints[3].bodyB.angle = target_ang + self.joints[3].bodyA.angle
        self.joints[3].bodyB.angularVelocity = state[12] * SPEED_KNEE \
                                               + self.joints[3].bodyA.angularVelocity

        # low 2 position
        knee_pos = self.joints[3].bodyA.position + LEG_H / 2 * np.array(
            [np.cos(self.joints[3].bodyA.angle-np.pi/2),
             np.sin(self.joints[3].bodyA.angle-np.pi/2)])
        self.joints[3].bodyB.position = knee_pos + LEG_H / 2 * np.array([
            np.cos(self.joints[3].bodyB.angle-np.pi/2),
            np.sin(self.joints[3].bodyB.angle-np.pi/2)])

        self.world.Step(1.0 / (FPS*100), 6 * 30, 2 * 30)

        pos = self.hull.position
        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        return self.get_state()

    def get_state(self):
        vel = self.hull.linearVelocity

        state = [
            self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
            2.0 * self.hull.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            self.joints[0].angle,
            # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0,
        ]
        state += [l.fraction for l in self.lidar]
        assert len(state) == 24

        return np.array(state, dtype=np.float32)

        # self.viewer.draw_circle(5, 50, position=self.knee_pos, color=(1,0,0))
