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
        state = np.clip(state, self.lo_clip, self.hi_clip).astype(np.float32)
        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H

        # hull
        self.hull.position = b2Vec2(init_x, init_y)
        self.hull.angle = state[0].item()  # Normal angles up to 0.5 here, but sure more is possible.
        self.hull.angularVelocity = state[1].item() * FPS / 2.0
        self.hull.linearVelocity.x = state[2] / 0.3 / (VIEWPORT_W / SCALE) * FPS  # Normalized to get -1..1 range
        self.hull.linearVelocity.y = state[3] / 0.3 / (VIEWPORT_H / SCALE) * FPS  # Normalized to get -1..1 range

        # legs
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []
        for i in [-1, +1]:
            hip_speed = state[5] if i == -1 else state[10]
            hip_angle = state[4] if i == -1 else state[11]
            knee_speed = state[7] if i == -1 else state[12]
            knee_angle = state[6] if i == -1 else state[13]

            leg = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LEG_FD,
            )
            leg.color1 = (0.6 - i / 10.0, 0.3 - i / 10.0, 0.5 - i / 10.0)
            leg.color2 = (0.4 - i / 10.0, 0.2 - i / 10.0, 0.3 - i / 10.0)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=i,
                lowerAngle=-0.8,
                upperAngle=1.1,
            )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd, speed=hip_speed, angle=hip_angle))

            lower = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LOWER_FD,
            )
            lower.color1 = (0.6 - i / 10.0, 0.3 - i / 10.0, 0.5 - i / 10.0)
            lower.color2 = (0.4 - i / 10.0, 0.2 - i / 10.0, 0.3 - i / 10.0)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-1.6,
                upperAngle=-0.1,
            )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd, speed=knee_speed, angle=knee_angle))

        self.drawlist = self.terrain + self.legs + [self.hull]

        # evaluate leg states
        self.world.Step(1 / FPS,1,1)

        # get lidar data
        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction

        self.lidar = [LidarCallback() for _ in range(10)]

        # self.world.Step(0,0,0)
        assert len(state) == 24

        return self.get_state()

    def get_state(self):
        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
            2.0 * self.hull.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            self.joints[
                0
            ].angle,
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