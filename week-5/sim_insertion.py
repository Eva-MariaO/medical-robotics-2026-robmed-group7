"""Visible S2RK insertion station, adapted from the course's week-4 robot.

j1..j5 hold HOME; j6 alone slides the instrument. Feedback is the axial
environment-on-tip force from actual PyBullet contacts, excluding gravity.
The liver mesh is visual; a one-sided compliant entry surface avoids trapping
the tool against the inside of a closed triangle shell during withdrawal.
"""
from dataclasses import dataclass
from pathlib import Path
import math
import os
import sys

import numpy as np


@dataclass(frozen=True)
class InsertionState:
    x: float           # insertion [m]
    velocity: float    # [m/s]
    force: float       # environment-on-tip axial force [N]; negative at forward contact
    applied_force: float


class InsertionFollower:
    kind = "insertion"
    TISSUES = {"bone": (50_000., 100.), "liver": (300., 20.)}

    def __init__(self, period=.01, tissue="bone", gui=True, force_limit=15.,
                 metres_per_rad=.04):
        import pybullet as p
        if gui and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            raise RuntimeError("PyBullet GUI needs a desktop display. Use sim_gui=False for headless tests.")
        self.p, self.gui, self.period = p, gui, period
        self.force_limit, self.metres_per_rad = force_limit, metres_per_rad
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("could not connect to PyBullet")
        self.assets = Path(__file__).resolve().parent / "assets" / "s2rk"
        self.obstacle = None
        self.visual_body = None
        self.applied_force = 0.
        self.time, self._last_draw = 0., -1.
        self._label = self._arrow = -1
        try:
            p.setGravity(0, 0, -9.81, physicsClientId=self.client)
            self.set_period(period)
            p.setPhysicsEngineParameter(numSolverIterations=100, physicsClientId=self.client)
            # Relative paths preserve PyBullet support for non-ASCII Windows home paths.
            urdf = os.path.relpath(self.assets / "s2rk_robot.urdf")
            self.robot = p.loadURDF(urdf, basePosition=[0, 0, .5], useFixedBase=True,
                                    flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.client)
            joints = {p.getJointInfo(self.robot, i, physicsClientId=self.client)[1].decode(): i
                      for i in range(p.getNumJoints(self.robot, physicsClientId=self.client))}
            self.joints = [joints[f"j{i}"] for i in range(1, 7)]
            self.insertion_joint = joints["j6"]
            self.tip_link = joints["tool_tip_fixed"]
            self.axis_link = joints["instrument_axis_fixed"]
            tip = p.getLinkState(self.robot, self.tip_link, computeForwardKinematics=True,
                                 physicsClientId=self.client)
            self.tip_home = np.array(tip[4])
            rotation = np.array(p.getMatrixFromQuaternion(tip[5])).reshape(3, 3)
            self.axis = rotation[:, 2]
            # Contact only at the instrument tip; the frozen arm cannot hit the obstacle.
            for link in range(-1, p.getNumJoints(self.robot, physicsClientId=self.client)):
                p.setCollisionFilterGroupMask(self.robot, link, 1, 2, physicsClientId=self.client)
            p.changeVisualShape(self.robot, self.tip_link, rgbaColor=[1, .55, .05, 1],
                                 physicsClientId=self.client)
            self._add_surroundings()
            self.camera_target = np.array([.5, 0, .75])
            self.camera_distance = 1.4
            if gui:
                p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, physicsClientId=self.client)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
                p.resetDebugVisualizerCamera(cameraDistance=self.camera_distance, cameraYaw=55, cameraPitch=-20,
                                             cameraTargetPosition=self.camera_target.tolist(),
                                             physicsClientId=self.client)
            self.set_tissue(tissue)
            self.reset()
        except BaseException:
            self.close()
            raise

    def set_period(self, period):
        if not math.isfinite(period) or period <= 0:
            raise ValueError("period must be positive")
        self.period = period
        # Stable contact integration while the student's controller remains sample-and-hold.
        self.substeps = max(1, math.ceil(period/(1/1000)))
        self.p.setTimeStep(period/self.substeps, physicsClientId=self.client)

    def _visual_box(self, half, position, colour):
        p = self.p
        shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=colour,
                                    physicsClientId=self.client)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape,
                                  basePosition=position, physicsClientId=self.client)

    def _add_surroundings(self):
        self._visual_box([.07, .08, .25], [0, 0, .25], [.4, .43, .48, 1])
        self._visual_box([.8, .6, .01], [.4, 0, -.02], [.85, .87, .9, 1])
        # A small work table under the organ; no contacts except the intended tip contact.
        self._visual_box([.18, .2, .015], [self.tip_home[0]+.03, 0, .37], [.4, .48, .55, 1])

    def set_tissue(self, tissue):
        if tissue not in self.TISSUES:
            raise ValueError("tissue must be 'bone' or 'liver'")
        p = self.p
        if self.visual_body is not None:
            p.removeBody(self.visual_body, physicsClientId=self.client)
            self.visual_body = None
        if self.obstacle is not None:
            p.removeBody(self.obstacle, physicsClientId=self.client)
        self.tissue = tissue
        if tissue == "liver":
            mesh = os.path.relpath(self.assets / "human_liver.obj")
            shape = p.createCollisionShape(p.GEOM_MESH, fileName=mesh, meshScale=[.05]*3,
                                           flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self.client)
            visual = p.createVisualShape(p.GEOM_MESH, fileName=mesh, meshScale=[.05]*3,
                                         rgbaColor=[.62, .16, .16, 1], physicsClientId=self.client)
            # As in week 4, the liver's thin dimension faces the instrument.
            normal = -self.axis
            cross = np.cross([0, 0, 1], normal)
            orientation = p.getQuaternionFromAxisAngle(cross/np.linalg.norm(cross),
                                                       math.acos(normal[2]))
        else:
            shape = p.createCollisionShape(p.GEOM_CAPSULE, radius=.025, height=.10,
                                           physicsClientId=self.client)
            visual = p.createVisualShape(p.GEOM_CAPSULE, radius=.025, length=.10,
                                         rgbaColor=[.94, .91, .77, 1], physicsClientId=self.client)
            orientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])
        centre = self.tip_home + .09*self.axis
        self.obstacle = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                          baseVisualShapeIndex=visual, basePosition=centre,
                                          baseOrientation=orientation, physicsClientId=self.client)
        p.setCollisionFilterGroupMask(self.obstacle, -1, 2, 1, physicsClientId=self.client)
        stiffness, damping = self.TISSUES[tissue]
        p.changeDynamics(self.obstacle, -1, contactStiffness=stiffness, contactDamping=damping,
                          lateralFriction=.1, restitution=0, physicsClientId=self.client)
        for link in range(-1, p.getNumJoints(self.robot, physicsClientId=self.client)):
            p.setCollisionFilterPair(self.robot, self.obstacle, link, -1,
                                     int(link == self.tip_link), physicsClientId=self.client)
        # Put both tissue surfaces at the SAME insertion depth for a fair comparison.
        hit = p.rayTest(self.tip_home, self.tip_home+.4*self.axis,
                        collisionFilterMask=2, physicsClientId=self.client)[0]
        if hit[0] != self.obstacle:
            raise RuntimeError("insertion axis does not intersect the tissue")
        distance = float(np.dot(np.array(hit[3])-self.tip_home, self.axis))
        centre += (.03-distance)*self.axis
        p.resetBasePositionAndOrientation(self.obstacle, centre, orientation, physicsClientId=self.client)
        if tissue == "liver":
            # Keep the week-4 mesh and the entry point found above, but do not
            # collide with its closed shell. Deep soft penetration otherwise
            # makes inner triangles push the instrument back INTO the organ.
            self.visual_body = self.obstacle
            p.setCollisionFilterGroupMask(self.visual_body, -1, 0, 0, physicsClientId=self.client)
            for link in range(-1, p.getNumJoints(self.robot, physicsClientId=self.client)):
                p.setCollisionFilterPair(self.robot, self.visual_body, link, -1, 0,
                                         physicsClientId=self.client)
            plane = p.createCollisionShape(p.GEOM_PLANE, planeNormal=[0, 0, 1],
                                            physicsClientId=self.client)
            invisible = p.createVisualShape(p.GEOM_SPHERE, radius=.0001, rgbaColor=[1, 1, 1, 0],
                                             physicsClientId=self.client)
            self.obstacle = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane,
                                              baseVisualShapeIndex=invisible,
                                              basePosition=self.tip_home+.03*self.axis,
                                              baseOrientation=orientation, physicsClientId=self.client)
            p.setCollisionFilterGroupMask(self.obstacle, -1, 2, 1, physicsClientId=self.client)
            p.changeDynamics(self.obstacle, -1, contactStiffness=stiffness, contactDamping=damping,
                              lateralFriction=0., restitution=0., physicsClientId=self.client)
            for link in range(-1, p.getNumJoints(self.robot, physicsClientId=self.client)):
                p.setCollisionFilterPair(self.robot, self.obstacle, link, -1,
                                         int(link == self.tip_link), physicsClientId=self.client)
        self.reset()

    def reset(self):
        p = self.p
        for joint in self.joints:
            p.resetJointState(self.robot, joint, 0., targetVelocity=0., physicsClientId=self.client)
        p.setJointMotorControl2(self.robot, self.insertion_joint, p.VELOCITY_CONTROL,
                                force=0, physicsClientId=self.client)
        self.applied_force = 0.
        self.time, self._last_draw = 0., -1.
        p.performCollisionDetection(physicsClientId=self.client)
        self._draw(self.read())

    def read(self):
        p = self.p
        x, velocity, _, _ = p.getJointState(self.robot, self.insertion_joint,
                                           physicsClientId=self.client)
        axial = 0.
        for contact in p.getContactPoints(bodyA=self.robot, bodyB=self.obstacle,
                                           linkIndexA=self.tip_link, physicsClientId=self.client):
            force = (np.array(contact[7])*contact[9] + np.array(contact[11])*contact[10]
                     + np.array(contact[13])*contact[12])
            axial += float(np.dot(force, self.axis))
        return InsertionState(x, velocity, axial, self.applied_force)

    def step(self, force):
        from dxl_utils import clip
        p = self.p
        self.applied_force = clip(force, self.force_limit)
        for _ in range(self.substeps):
            p.setJointMotorControlArray(self.robot, self.joints[:5], p.POSITION_CONTROL,
                                        targetPositions=[0.]*5, forces=[1000.]*5,
                                        physicsClientId=self.client)
            states = p.getJointStates(self.robot, self.joints, physicsClientId=self.client)
            q = [s[0] for s in states]
            gravity = p.calculateInverseDynamics(self.robot, q, [0.]*6, [0.]*6,
                                                 physicsClientId=self.client)[-1]
            # Limit the insertion workspace before a controller can hit the URDF end stop.
            drive = self.applied_force
            if (q[-1] >= .08 and drive > 0) or (q[-1] <= -.03 and drive < 0):
                drive = -30.*states[-1][1]
            p.setJointMotorControl2(self.robot, self.insertion_joint, p.TORQUE_CONTROL,
                                    force=drive+gravity, physicsClientId=self.client)
            p.stepSimulation(physicsClientId=self.client)
        self.time += self.period
        state = self.read()
        if self.time-self._last_draw >= 1/30:
            self._draw(state)
            self._last_draw = self.time
        return state

    def _draw(self, state):
        if not self.gui:
            return
        p = self.p
        touching = abs(state.force) > .05
        p.changeVisualShape(self.robot, self.tip_link,
                            rgbaColor=[1, .1, .05, 1] if touching else [1, .55, .05, 1],
                            physicsClientId=self.client)
        self._label = p.addUserDebugText(
            f"{self.tissue.upper()}  insertion {1000*state.x:.1f} mm  reaction {state.force:.2f} N",
            self.tip_home+np.array([-.10, 0, .18]), textColorRGB=[.1, .1, .1], textSize=1.3,
            replaceItemUniqueId=self._label, physicsClientId=self.client)
        tip = np.array(p.getLinkState(self.robot, self.tip_link, physicsClientId=self.client)[4])
        self._arrow = p.addUserDebugLine(tip, tip + self.axis*state.force*.005,
                                        lineColorRGB=[1, 0, 0], lineWidth=4,
                                        replaceItemUniqueId=self._arrow, physicsClientId=self.client)

    def render(self, width=800, height=600):
        """RGB image for notebook snapshots and headless visual verification."""
        p = self.p
        view = p.computeViewMatrixFromYawPitchRoll(self.camera_target, self.camera_distance, 55, -20, 0, 2)
        projection = p.computeProjectionMatrixFOV(50, width/height, .02, 3.)
        return np.array(p.getCameraImage(width, height, view, projection,
                                         renderer=p.ER_TINY_RENDERER,
                                         physicsClientId=self.client)[2])[:, :, :3]

    def focus_tool(self):
        """Optional close-up of insertion/contact; mouse control remains available."""
        self.camera_target = self.tip_home + .025*self.axis
        self.camera_distance = .32
        if self.gui:
            self.p.resetDebugVisualizerCamera(cameraDistance=self.camera_distance, cameraYaw=55,
                                              cameraPitch=-20, cameraTargetPosition=self.camera_target,
                                              physicsClientId=self.client)

    def close(self):
        if self.client is not None and self.p.isConnected(self.client):
            self.p.disconnect(self.client)
        self.client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
