"""Helper functions for the kinematics lab.

You do not need to edit this file. Read the docstrings to see what each function does.
"""
import os
import sys
import time
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p

IN_COLAB = "google.colab" in sys.modules
REPO_RAW = "https://raw.githubusercontent.com/LxViRaL-teaching/medical-robotics-2026/main/week-2/"


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
def fetch(filename):
    """Download a lab file from the course repository if it is not in this folder (Colab)."""
    if not os.path.exists(filename):
        urllib.request.urlretrieve(REPO_RAW + filename, filename)
        print("downloaded", filename)


def connect():
    """Start PyBullet (3D window locally, no window on Colab) with gravity OFF.

    This is a kinematics lab: we set joint angles directly and never step the dynamics.
    If a simulator is already running from an earlier cell, it is closed first.
    """
    if p.isConnected():
        p.disconnect()
    client = p.connect(p.DIRECT if IN_COLAB else p.GUI)
    p.setGravity(0, 0, 0)
    if not IN_COLAB:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=3.2, cameraYaw=0, cameraPitch=-15,
                                     cameraTargetPosition=[0.4, 0, 1.1])
    return client


def load_robot(urdf_file):
    """Load a URDF with its base fixed at the world origin and return its body id."""
    return p.loadURDF(urdf_file, basePosition=[0, 0, 0], useFixedBase=True)


def reset_scene(urdf_file):
    """Remove every body in the simulation and load `urdf_file` again. Returns the new body id."""
    for i in reversed(range(p.getNumBodies())):
        p.removeBody(p.getBodyUniqueId(i))
    return load_robot(urdf_file)


# ----------------------------------------------------------------------------
# Joints and links (always by NAME, never by index)
# ----------------------------------------------------------------------------
_TYPES = {p.JOINT_REVOLUTE: "revolute", p.JOINT_PRISMATIC: "prismatic", p.JOINT_FIXED: "fixed"}


def print_joints(robot):
    """Print every joint: index, name, type, parent link -> child link."""
    names = {-1: p.getBodyInfo(robot)[0].decode()}
    for j in range(p.getNumJoints(robot)):
        names[j] = p.getJointInfo(robot, j)[12].decode()
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        parent = names[info[16]]
        print(f"{j:2d}  {info[1].decode():14s} {_TYPES.get(info[2], str(info[2])):9s} "
              f"{parent} -> {info[12].decode()}")


def moving_joints(robot):
    """Names of the non-fixed joints, in the order they appear in the URDF."""
    return [p.getJointInfo(robot, j)[1].decode() for j in range(p.getNumJoints(robot))
            if p.getJointInfo(robot, j)[2] != p.JOINT_FIXED]


def joint_index(robot, joint_name):
    for j in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, j)[1].decode() == joint_name:
            return j
    raise KeyError(f"no joint called '{joint_name}'")


def link_index(robot, link_name):
    """In PyBullet a link has the same index as the joint that connects it to its parent."""
    for j in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, j)[12].decode() == link_name:
            return j
    raise KeyError(f"no link called '{link_name}'")


def set_joints(robot, joint_names, q):
    """Teleport the joints `joint_names` to the angles `q` (radians). No dynamics involved."""
    for name, angle in zip(joint_names, q):
        p.resetJointState(robot, joint_index(robot, name), float(angle))


def get_joints(robot, joint_names):
    return np.array([p.getJointState(robot, joint_index(robot, n))[0] for n in joint_names])


def link_position(robot, link_name):
    """World position of the ORIGIN of a link frame, as a numpy array [x, y, z].

    Note: getLinkState()[0] is the centre of mass, which is not what we want here.
    We use getLinkState()[4] (the link frame) and force PyBullet to recompute the kinematics.
    """
    state = p.getLinkState(robot, link_index(robot, link_name), computeForwardKinematics=True)
    return np.array(state[4])


# ----------------------------------------------------------------------------
# Numerics
# ----------------------------------------------------------------------------
def numerical_jacobian(f, q, eps=1e-6):
    """Jacobian of f at q by central finite differences.

    f takes a 1D array of joint angles and returns a 1D array (for example a position).
    Returns a matrix with shape (len(f(q)), len(q)).
    """
    q = np.asarray(q, dtype=float)
    cols = []
    for i in range(len(q)):
        dq = np.zeros_like(q)
        dq[i] = eps
        cols.append((np.asarray(f(q + dq)) - np.asarray(f(q - dq))) / (2 * eps))
    return np.column_stack(cols)


def wrap_angles(q):
    """Bring angles back to (-pi, pi]. Does not change the pose, only how the angles are written."""
    q = np.asarray(q, dtype=float)
    return np.arctan2(np.sin(q), np.cos(q))


def check(label, got, expected, tol=1e-3):
    """Print PASS/FAIL comparing two arrays (default tolerance 1 mm)."""
    got, expected = np.asarray(got, float), np.asarray(expected, float)
    err = np.linalg.norm(got - expected)
    status = "PASS" if err < tol else "FAIL"
    print(f"[{status}] {label}: got {np.round(got, 4)}, expected {np.round(expected, 4)}, error {err:.2e}")
    return err < tol


# ----------------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------------
def add_marker(position, color=(1, 0.5, 0, 0.8), radius=0.04):
    """Draw a sphere (visual only, no collision, no mass) at `position`. Returns its body id."""
    shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=list(position))


def move_marker(marker, position):
    p.resetBasePositionAndOrientation(marker, list(position), [0, 0, 0, 1])


def animate(robot, joint_names, q_history, duration=0.5, steps=25):
    """Move smoothly through a list of joint configurations (e.g. the iterations of an IK solver).

    Starts from the current configuration and spends `duration` seconds on each segment,
    turning each joint the short way round. On Colab it only sets the last configuration.
    """
    q_prev = get_joints(robot, joint_names)
    for q in q_history:
        q = np.asarray(q, dtype=float)
        dq = wrap_angles(q - q_prev)
        if not IN_COLAB:
            for s in np.linspace(0, 1, steps)[1:]:
                set_joints(robot, joint_names, q_prev + s * dq)
                time.sleep(duration / steps)
        set_joints(robot, joint_names, q)
        q_prev = q


def move_to(robot, joint_names, q, duration=1.0):
    """Move smoothly from the current configuration to `q`."""
    animate(robot, joint_names, [q], duration=duration)


def plot_workspace(fk, n=3000, joint_range=(-np.pi, np.pi), labels=("x", "z")):
    """Scatter the positions reached by `fk` for random joint angles (2D fk only)."""
    rng = np.random.default_rng(0)
    q = rng.uniform(*joint_range, size=(n, 2))
    pts = np.array([fk(qi) for qi in q])
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.4)
    plt.plot(0, 0, "k+", markersize=12)
    plt.gca().set_aspect("equal")
    plt.xlabel(f"{labels[0]} [m]")
    plt.ylabel(f"{labels[1]} [m]")
    plt.title("Reachable positions")
    plt.grid(True, alpha=0.3)
    plt.show()



def plot_convergence(fk, target, q_history, label=None, ax=None):
    """Plot the position error |target - fk(q)| at every iteration (log scale)."""
    err = [np.linalg.norm(np.asarray(target) - np.asarray(fk(q))) for q in q_history]
    ax = ax or plt.figure(figsize=(5, 3.2)).gca()
    ax.semilogy(err, "-o", ms=4, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("position error [m]")
    ax.grid(True, which="both", alpha=0.3)
    if label:
        ax.legend()
    return ax
