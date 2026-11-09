"""Helper functions for the week 7 lab: learning to move the S2RK in a circle through the trocar.

You do not need to edit this file. It reuses the robot and the scene of the week 4 lab (keyhole_utils)
and adds:

* KinematicRobot: a second copy of the robot in its own simulation, with no gravity, dynamics or
  contacts. Setting its joints is instantaneous, so we use it to train quickly and to compute the
  geometry (tip position, shaft direction, Jacobian) at any joint values.
* the "machinery" that turns a tool-tip velocity and a pitch tilt of the shaft into joint velocities;
* the circle: its frame and the measurements we need (angle, radial error, height, shaft offset);
* drawing and plotting.

* CircleLab: the states, the actions, the training loop and the runs in the full simulation.

The notebook writes only the reward and the Q-learning update; CircleLab calls them.
"""
import os
import sys
import time

import numpy as np
import pybullet as p
import plotly.graph_objects as go

_HERE = os.path.dirname(os.path.abspath(__file__))
for _folder in (_HERE, os.path.join(_HERE, "..", "week-04-cirurgia-keyhole"), os.path.join(_HERE, "..", "week-4")):
    if os.path.exists(os.path.join(_folder, "keyhole_utils.py")):
        sys.path.insert(0, os.path.abspath(_folder))
        break
import keyhole_utils as kh  # noqa: E402

# ----------------------------------------------------------------------------
# The task
# ----------------------------------------------------------------------------
RADIUS = 0.015            # radius of the circle [m]
SPEED = 0.01              # desired speed of the tip along the circle [m/s]
INSERTION_START = 0.03    # insertion j6 at the start pose [m]
POLICY_PERIOD = 0.1       # time between two decisions of the agent [s]
TILT_RATE = 0.07          # pitch rate of the shaft for one tilt action [rad/s]
CONTACT_OFFSET = 0.0075   # the shaft touches the port ring beyond this distance from the RCM [m]


def start_joints():
    """Joint values [j1..j6] of the start pose: on the RCM, pitch 0, insertion INSERTION_START."""
    return np.array([0.0, *kh.parallelogram_joints(0.0), INSERTION_START])



def rcm_joints(swing, pitch, insertion):
    """The six joint values [j1..j6] for the reduced coordinates swing [rad], pitch [rad], insertion [m].
    Any values keep the shaft on the RCM (week 4)."""
    return np.array([swing, *kh.parallelogram_joints(pitch), insertion])


def joints_to_reach(kinematic, point, start=(0.0, 0.0, INSERTION_START), iterations=20):
    """Joint values [j1..j6] that put the tool tip at `point` (world frame, [m]) with the shaft through the RCM.

    Through a trocar the orientation is not free: the shaft must lie on the line from the RCM to the
    point, so solving for the tip position fixes the orientation too. Newton's method on the reduced
    coordinates (swing, pitch, insertion), with a numerical Jacobian. Leaves the kinematic robot at the result.
    """
    def tip_at(x):
        kinematic.set_joints(rcm_joints(*x))
        return kinematic.tip_position()
    x = np.array(start, dtype=float)
    for _ in range(iterations):
        error = point - tip_at(x)
        if np.linalg.norm(error) < 1e-7:
            break
        step = 1e-6
        jacobian = np.column_stack([(tip_at(x + dx) - tip_at(x - dx)) / (2 * step) for dx in np.eye(3) * step])
        x = x + np.linalg.solve(jacobian, error)
    q = rcm_joints(*x)
    kinematic.set_joints(q)
    return q

# ----------------------------------------------------------------------------
# A kinematic copy of the robot
# ----------------------------------------------------------------------------
class KinematicRobot:
    """The S2RK in a separate simulation without gravity, contacts or dynamics.

    Create it after kh.connect(). Its joints jump to the values you set; nothing else moves it.
    """

    def __init__(self):
        self.client = p.connect(p.DIRECT)
        if self.client == 0:
            raise RuntimeError("call kh.connect() before creating the KinematicRobot")
        self.robot = p.loadURDF(kh._lab_file("s2rk_robot.urdf"), basePosition=kh.BASE_POSITION.tolist(),
                                useFixedBase=True, physicsClientId=self.client)
        names = {}
        for i in range(p.getNumJoints(self.robot, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.client)
            names[info[1].decode()] = i
            names[info[12].decode()] = i
        self.joints = [names[name] for name in kh.JOINT_NAMES]
        self.tip_link = names["tool_tip_link"]
        self.axis_link = names["instrument_axis_link"]
        limits = [p.getJointInfo(self.robot, j, physicsClientId=self.client)[8:10] for j in self.joints]
        self.lower_limits = np.array([low for low, high in limits])
        self.upper_limits = np.array([high for low, high in limits])
        self.set_joints(start_joints())

    def set_joints(self, q):
        """Place the six joints at q [rad x5, m]."""
        self.q = np.array(q, dtype=float)
        for j, value in zip(self.joints, self.q):
            p.resetJointState(self.robot, j, value, physicsClientId=self.client)
        tip_state = p.getLinkState(self.robot, self.tip_link, computeForwardKinematics=True,
                                   physicsClientId=self.client)
        axis_state = p.getLinkState(self.robot, self.axis_link, computeForwardKinematics=True,
                                    physicsClientId=self.client)
        self._tip = np.array(tip_state[4])
        axis_rotation = np.array(p.getMatrixFromQuaternion(axis_state[5])).reshape(3, 3)
        self._shaft = axis_rotation[:, 2]

    def joint_values(self):
        """Current joint values [j1..j6]."""
        return self.q.copy()

    def tip_position(self):
        """Position of the tool tip in the world [m]."""
        return self._tip.copy()

    def shaft_direction(self):
        """Unit vector along the instrument shaft, pointing to the tip [-]."""
        return self._shaft.copy()

    def joint_velocities(self, tip_velocity, tilt_rate):
        """Joint velocities [rad/s x5, m/s] that move the tip at tip_velocity (world frame, [m/s]) and
        turn the shaft about the y axis (pitch) at tilt_rate [rad/s].

        It uses the Jacobian of the tip: 3 rows for the linear velocity and the row of the angular
        velocity about y. With six joints and four task rows the robot is redundant; the damped
        least-squares solution picks the smallest joint velocities.
        Nothing here keeps the shaft at the trocar: that depends on the tilt you choose.
        """
        zeros = [0.0] * 6
        linear, angular = p.calculateJacobian(self.robot, self.tip_link, [0, 0, 0], self.q.tolist(), zeros, zeros,
                                              physicsClientId=self.client)
        jacobian = np.vstack([np.array(linear), np.array(angular)[1]])        # 4 x 6
        task_velocity = np.append(tip_velocity, tilt_rate)                     # (vx, vy, vz, pitch rate)
        damping = 1e-3
        return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(4), task_velocity)

    def move(self, tip_velocity, tilt_rate, duration):
        """Move for `duration` seconds at constant joint velocities (one Euler step), within the joint limits."""
        q_new = self.q + self.joint_velocities(tip_velocity, tilt_rate) * duration
        self.set_joints(np.clip(q_new, self.lower_limits, self.upper_limits))


# ----------------------------------------------------------------------------
# The circle
# ----------------------------------------------------------------------------
class Circle:
    """The circle the tip should follow, in the plane normal to the shaft at the start pose.

    Frame: e1 sideways (the swing direction), e2 in the pitch plane, n along the shaft at the start.
    The start pose is on the circle, at the angle -90 degrees. "Anticlockwise" is about n.
    """

    def __init__(self, kinematic):
        q_saved = kinematic.joint_values()
        kinematic.set_joints(start_joints())
        start_tip = kinematic.tip_position()
        self.n = kinematic.shaft_direction()
        self.e1 = np.array([0.0, 1.0, 0.0])
        self.e2 = np.cross(self.n, self.e1)
        self.centre = start_tip + RADIUS * self.e2
        kinematic.set_joints(q_saved)

    def to_world(self, v1, v2, vn):
        """A vector given in the circle frame (e1, e2, n), expressed in the world frame."""
        return v1 * self.e1 + v2 * self.e2 + vn * self.n

    def coordinates(self, tip):
        """Coordinates (x1, x2, height) of the tip in the circle frame [m]."""
        relative = tip - self.centre
        return np.dot(relative, self.e1), np.dot(relative, self.e2), np.dot(relative, self.n)

    def angle(self, tip):
        """Angle of the tip around the circle [rad], in (-pi, pi]."""
        x1, x2, _ = self.coordinates(tip)
        return np.arctan2(x2, x1)

    def radial_error(self, tip):
        """Distance from the centre minus the radius [m]: > 0 outside the circle."""
        x1, x2, _ = self.coordinates(tip)
        return np.hypot(x1, x2) - RADIUS

    def height(self, tip):
        """Distance of the tip from the plane of the circle [m]."""
        _, _, h = self.coordinates(tip)
        return h

    def tangent(self, tip):
        """Unit vector along the circle, anticlockwise, at the angle of the tip (world frame)."""
        angle = self.angle(tip)
        return self.to_world(-np.sin(angle), np.cos(angle), 0.0)

    def shaft_offset(self, tip, shaft_direction):
        """Signed distance from the RCM to the shaft axis, along e2 [m].

        On this robot the swing joint keeps the RCM by itself, so the shaft can only leave the RCM
        in the pitch plane, along e2. |offset| > CONTACT_OFFSET means the shaft touches the port ring.
        """
        tip_to_trocar = kh.RCM_POSITION - tip
        along_shaft = np.dot(tip_to_trocar, shaft_direction) * shaft_direction
        across_shaft = tip_to_trocar - along_shaft
        return np.dot(across_shaft, self.e2)


def draw_circle(circle, colour=(0.1, 0.3, 0.9)):
    """Draw the circle in the 3D window."""
    angles = np.linspace(0, 2 * np.pi, 61)
    points = [circle.centre + RADIUS * (np.cos(a) * circle.e1 + np.sin(a) * circle.e2) for a in angles]
    for a, b in zip(points[:-1], points[1:]):
        p.addUserDebugLine(a.tolist(), b.tolist(), lineColorRGB=list(colour), lineWidth=2)


def look_at_circle(circle):
    """Point the 3D window camera at the circle and the trocar."""
    if not kh.IN_COLAB:
        p.resetDebugVisualizerCamera(cameraDistance=0.35, cameraYaw=60, cameraPitch=-25,
                                     cameraTargetPosition=(0.5 * (circle.centre + kh.RCM_POSITION)).tolist())


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
plot_series = kh.plot_series


def plot_paths(paths, title="Tip path in the circle plane"):
    """Plot tip paths in the circle plane. paths is a dict {label: (x1 list [m], x2 list [m])}."""
    angles = np.linspace(0, 2 * np.pi, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=1000 * RADIUS * np.cos(angles), y=1000 * RADIUS * np.sin(angles), mode="lines",
                             name="circle", line=dict(dash="dash", color="gray")))
    for label, (x1, x2) in paths.items():
        fig.add_trace(go.Scatter(x=1000 * np.array(x1), y=1000 * np.array(x2), mode="lines", name=label))
    fig.update_layout(title=title, xaxis_title="x1 [mm]", yaxis_title="x2 [mm]",
                      yaxis=dict(scaleanchor="x", scaleratio=1), margin=dict(l=10, r=10, t=40, b=10))
    fig.show()


# ----------------------------------------------------------------------------
# The learning problem: actions, states, environment, training and test runs
# ----------------------------------------------------------------------------
# Actions: tip velocity in the circle frame (v1, v2, vn), each in {-1, 0, +1} x SPEED, and a pitch tilt in
# {-1, 0, +1} x TILT_RATE: 3 x 3 x 3 x 3 = 81 actions.
ACTIONS = [(v1, v2, vn, tilt) for v1 in (-1, 0, 1) for v2 in (-1, 0, 1) for vn in (-1, 0, 1) for tilt in (-1, 0, 1)]

# States: 4 sectors around the circle x 4 radial bins x 3 bins of the shaft offset x 3 height bins = 144 states.
N_SECTORS = 4
RADIAL_EDGES = [-0.003, 0.0, 0.003]      # [m]
OFFSET_EDGES = [-0.0025, 0.0025]         # [m]
HEIGHT_EDGES = [-0.0015, 0.0015]         # [m]
EPISODE_STEPS = 300                      # 30 s per episode, about three laps


class CircleLab:
    """Everything around the learning: the scene, the states and actions, training and test runs.

    The notebook writes the reward and the Q-learning update; this class calls them.
    """

    def __init__(self, real_time=True):
        self.real_time = real_time and not kh.IN_COLAB
        kh.connect()
        self.robot = kh.load_robot()
        self.scene = kh.add_scene(self.robot)
        self.port = self.scene["port"]
        self.joints = kh.joint_indices(self.robot)
        self.max_forces = [30, 30, 30, 30, 30, 40]
        self.kinematic = KinematicRobot()
        self.circle = Circle(self.kinematic)
        draw_circle(self.circle)
        look_at_circle(self.circle)
        kh.set_joints(self.robot, start_joints())
        self.actions = ACTIONS
        self.n_actions = len(ACTIONS)
        self.n_states = N_SECTORS * (len(RADIAL_EDGES) + 1) * (len(OFFSET_EDGES) + 1) * (len(HEIGHT_EDGES) + 1)

    # -------------------------------------------------------------- states and measurements
    def state(self, tip, shaft):
        """State number (0 .. n_states-1) for a tip position and a shaft direction."""
        sector = int((self.circle.angle(tip) + np.pi) / (2 * np.pi) * N_SECTORS) % N_SECTORS
        radial_bin = int(np.digitize(self.circle.radial_error(tip), RADIAL_EDGES))
        offset_bin = int(np.digitize(self.circle.shaft_offset(tip, shaft), OFFSET_EDGES))
        height_bin = int(np.digitize(self.circle.height(tip), HEIGHT_EDGES))
        state = sector
        state = state * (len(RADIAL_EDGES) + 1) + radial_bin
        state = state * (len(OFFSET_EDGES) + 1) + offset_bin
        state = state * (len(HEIGHT_EDGES) + 1) + height_bin
        return state

    def measurements(self, tip_before, tip_after, shaft_after):
        """The quantities the reward uses, after one decision period."""
        velocity = (tip_after - tip_before) / POLICY_PERIOD
        return dict(tangential_speed=float(np.dot(velocity, self.circle.tangent(tip_after))),
                    radial_error=float(self.circle.radial_error(tip_after)),
                    height=float(self.circle.height(tip_after)),
                    shaft_offset=float(self.circle.shaft_offset(tip_after, shaft_after)))

    def _command(self, action_index):
        v1, v2, vn, tilt = self.actions[action_index]
        tip_velocity = self.circle.to_world(v1 * SPEED, v2 * SPEED, vn * SPEED)
        return tip_velocity, tilt * TILT_RATE

    # -------------------------------------------------------------- training (kinematic robot)
    def reset(self):
        self.kinematic.set_joints(start_joints())
        return self.state(self.kinematic.tip_position(), self.kinematic.shaft_direction())

    def step(self, action_index, reward_function, port_penalty):
        tip_velocity, tilt_rate = self._command(action_index)
        tip_before = self.kinematic.tip_position()
        self.kinematic.move(tip_velocity, tilt_rate, POLICY_PERIOD)
        tip_after = self.kinematic.tip_position()
        shaft_after = self.kinematic.shaft_direction()
        m = self.measurements(tip_before, tip_after, shaft_after)
        r = reward_function(m["tangential_speed"], m["radial_error"], m["height"], m["shaft_offset"], port_penalty)
        return self.state(tip_after, shaft_after), r

    def train(self, reward_function, q_update, choose_action, port_penalty, episodes=1000, seed=0):
        """Q-learning on the kinematic robot. reward_function, q_update and choose_action come from the notebook.

        choose_action(Q, state, epsilon, rng) is called at every step, with epsilon going from 0.3 down to 0.02
        over the first 80% of the episodes. rng is a numpy random generator.
        Returns the table Q, the return of each episode, and how many times each entry was updated.
        """
        rng = np.random.default_rng(seed)
        Q = np.zeros((self.n_states, self.n_actions))
        visits = np.zeros((self.n_states, self.n_actions), dtype=int)
        returns = []
        start = time.time()
        for episode in range(episodes):
            epsilon = max(0.02, 0.3 * (1 - episode / (0.8 * episodes)))
            state = self.reset()
            episode_return = 0.0
            for k in range(EPISODE_STEPS):
                action = int(choose_action(Q, state, epsilon, rng))
                next_state, r = self.step(action, reward_function, port_penalty)
                q_update(Q, state, action, r, next_state, k == EPISODE_STEPS - 1)
                visits[state, action] += 1
                state = next_state
                episode_return += r
            returns.append(episode_return)
            if (episode + 1) % 100 == 0:
                print(f"episode {episode + 1}: mean return of the last 100 = {np.mean(returns[-100:]):.0f}", flush=True)
        print(f"training time: {time.time() - start:.0f} s")
        return Q, returns, visits

    # -------------------------------------------------------------- runs in the full simulation
    def _run(self, choose_command, duration):
        steps_per_decision = round(POLICY_PERIOD / kh.DT)
        kh.set_joints(self.robot, start_joints())
        log = {"time": [], "offset_mm": [], "force": [], "x1": [], "x2": []}
        t = 0.0
        while t < duration:
            q = kh.get_joints(self.robot)
            self.kinematic.set_joints(q)
            tip = self.kinematic.tip_position()
            shaft = self.kinematic.shaft_direction()
            tip_velocity, tilt_rate = choose_command(tip, shaft)
            q_target = q + self.kinematic.joint_velocities(tip_velocity, tilt_rate) * POLICY_PERIOD
            port_force = 0.0
            for _ in range(steps_per_decision):
                p.setJointMotorControlArray(self.robot, self.joints, p.POSITION_CONTROL, targetPositions=q_target,
                                            forces=self.max_forces)
                p.stepSimulation()
                port_force = max(port_force, float(np.linalg.norm(kh.port_force(self.robot, self.port))))
            if self.real_time:
                time.sleep(POLICY_PERIOD)
            t += POLICY_PERIOD
            x1, x2, _ = self.circle.coordinates(tip)
            log["time"].append(t)
            log["offset_mm"].append(1000 * self.circle.shaft_offset(tip, shaft))
            log["force"].append(port_force)
            log["x1"].append(x1)
            log["x2"].append(x2)
        return log

    def run_without_tilt(self, duration=4.0):
        """Full simulation: the tip follows the circle at SPEED and the shaft never tilts."""
        return self._run(lambda tip, shaft: (SPEED * self.circle.tangent(tip), 0.0), duration)

    def run_policy(self, Q, pick_best_action, duration=30.0):
        """Full simulation: at every decision, the action pick_best_action(Q, state) chooses (no exploration)."""
        def greedy(tip, shaft):
            return self._command(int(pick_best_action(Q, self.state(tip, shaft))))
        log = self._run(greedy, duration)
        print(f"largest force on the port: {max(log['force']):.1f} N")
        return log
