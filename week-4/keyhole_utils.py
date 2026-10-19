"""Helper functions for the keyhole surgery lab.

You do not need to edit this file. These helpers set up the simulation, load the robot and the
scene, grab images from the camera on the instrument and find the coloured dots in them.
The RCM kinematics, the Jacobian and the control loop are written in the notebook.
"""
import os
import sys

import numpy as np
import pybullet as p
import pybullet_data
import plotly.express as px
import plotly.graph_objects as go

IN_COLAB = "google.colab" in sys.modules
_HERE = os.path.dirname(os.path.abspath(__file__))


def _lab_file(*parts):
    """Path to a file in the lab folder, relative to the working directory when possible.

    PyBullet cannot open absolute paths with non-ASCII characters on Windows (e.g. a folder
    called "Ciências"), so files are passed to it with a relative path.
    """
    path = os.path.join(_HERE, *parts)
    try:
        return os.path.relpath(path)
    except ValueError:      # on another drive than the working directory
        return path


DT = 1.0 / 240.0   # simulation time step [s], also our control period
G = 9.81           # gravity [m/s^2]

JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # joint values at the start [rad x5, m]

BASE_POSITION = np.array([0.0, 0.0, 0.5])                    # the robot base, on a pedestal [m]
RCM_POSITION = BASE_POSITION + np.array([0.7235, 0.0, 0.18])  # the trocar: remote centre of motion [m]
SHAFT_DIRECTION_HOME = np.array([0.4747819, 0.0, -0.8801035])  # instrument axis at HOME [-]

# The target: a liver with three coloured dots on its surface, in a triangle.
# The liver lies below the trocar, its thin side facing it.
TARGET_POSITION = np.array([0.76, 0.07, 0.50])   # centre of the triangle of dots (before projecting on the liver) [m]
LIVER_FILE = "human_liver.obj"         # liver mesh, about 4.2 x 2.9 x 1.3 units, centred at the origin
LIVER_SCALE = 0.05                     # mesh units to metres: the liver is about 21 x 14 x 7 cm
LIVER_DEPTH = 0.015                    # the liver centre is this far below TARGET_POSITION, away from the trocar [m]
LIVER_SHIFT = np.array([0.0, -0.02])   # sideways shift of the liver in the target frame (x, y), so the dots fall on it [m]
LIVER_STIFFNESS = 1500.0               # contact stiffness of the liver surface [N/m]
LIVER_DAMPING = 100.0                  # contact damping of the liver surface [N s/m]
DOT_RADIUS = 0.006                     # radius of each dot [m]
DOT_DISTANCE = 0.035                   # distance from the triangle centre to each dot [m]
DOT_COLOURS = ("yellow", "green", "blue")
DOT_ANGLES = np.radians([90.0, 210.0, 330.0])   # where each dot is around the centre (in the target frame)
DOT_RGBA = {"yellow": [1, 0.85, 0, 1], "green": [0, 0.8, 0, 1], "blue": [0, 0, 1, 1]}

# The trocar port: a rigid ring around the RCM. Its inner radius leaves about 9 mm of clearance around
# the shaft (radius 4.2 mm): if the instrument moves sideways at the trocar, it touches the ring.
PORT_FILE = "port_torus.obj"   # visual: major radius 18 mm, tube radius 5 mm
PORT_RADIUS = 0.018            # radius of the ring, to the centre of its tube [m]
PORT_TUBE_RADIUS = 0.005       # radius of the tube [m]

# Camera settings
IMAGE_WIDTH = 640    # [pixels]
IMAGE_HEIGHT = 480   # [pixels]
FOV_Y_DEG = 60.0     # vertical field of view [degrees]
NEAR = 0.01          # closest distance the camera renders [m]
FAR = 3.0            # farthest distance the camera renders [m]

# The RCM coupling: for each pitch of the instrument about the RCM, the values of j2..j5
_COUPLING = np.loadtxt(os.path.join(_HERE, "rcm_coupling.csv"), delimiter=",")

COLLIDED=False

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
def connect(camera_target=(0.55, 0.0, 0.5), camera_distance=1.3, yaw=35, pitch=-20):
    """Start PyBullet (3D window locally, no window on Colab) with gravity and time step DT.

    The 3D window shows, in its side panel, the last image taken with get_camera_image.

    If a simulation is already running from an earlier cell, it is closed first.
    """
    if p.isConnected():
        p.disconnect()
    client = p.connect(p.DIRECT if IN_COLAB else p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -G)
    p.setTimeStep(DT)
    if not IN_COLAB:
        # Keep the side panel: it shows the last picture taken with get_camera_image (RGB only)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        #p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        #p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.resetDebugVisualizerCamera(cameraDistance=camera_distance, cameraYaw=yaw,
                                     cameraPitch=pitch, cameraTargetPosition=list(camera_target))
    return client


def load_robot(urdf_file="s2rk_robot.urdf"):
    """Load the S2RK robot with a fixed base at BASE_POSITION, at HOME. Returns the body id."""
    robot = p.loadURDF(_lab_file(urdf_file), basePosition=BASE_POSITION.tolist(),
                       useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
    p.changeVisualShape(robot, link_index(robot, "Link6"), rgbaColor=[0.75, 0.75, 0.78, 1])  # instrument
    p.enableJointForceTorqueSensor(robot, joint_index(robot, "tool_tip_fixed"), True)   # tip F/T sensor
    set_joints(robot, HOME)
    return robot


def joint_index(body, name):
    """Index of the joint with this name."""
    for i in range(p.getNumJoints(body)):
        if p.getJointInfo(body, i)[1].decode() == name:
            return i
    raise ValueError(f"no joint named {name}")


def joint_indices(body):
    """Indices of the six joints j1..j6, in order."""
    return [joint_index(body, name) for name in JOINT_NAMES]


def link_index(body, name):
    """Index of the link with this name (the base link is -1)."""
    if p.getBodyInfo(body)[0].decode() == name:
        return -1
    for i in range(p.getNumJoints(body)):
        if p.getJointInfo(body, i)[12].decode() == name:
            return i
    raise ValueError(f"no link named {name}")


def joint_summary(body):
    """Print the movable joints of a body with their type and limits."""
    names = {p.JOINT_REVOLUTE: "revolute", p.JOINT_PRISMATIC: "prismatic"}
    for i in range(p.getNumJoints(body)):
        info = p.getJointInfo(body, i)
        if info[2] in names:
            unit = "m" if info[2] == p.JOINT_PRISMATIC else "rad"
            print(f"{i}  {info[1].decode()}  {names[info[2]]:10s} "
                  f"limits [{info[8]:+.3f}, {info[9]:+.3f}] {unit:3s}  moves link: {info[12].decode()}")


def set_joints(body, q):
    """Place the six joints at the values q = [j1, ..., j6] instantly (no dynamics)."""
    for name, value in zip(JOINT_NAMES, q):
        p.resetJointState(body, joint_index(body, name), value)


def get_joints(body):
    """Current values of the six joints [j1, ..., j6]."""
    return np.array([p.getJointState(body, i)[0] for i in joint_indices(body)])


def link_pose(body, link_name):
    """Position [m] and rotation matrix of a link frame in the world."""
    state = p.getLinkState(body, link_index(body, link_name), computeForwardKinematics=True)
    position = np.array(state[4])
    rotation = np.array(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
    return position, rotation


def parallelogram_joints(pitch):
    """Values of [j2, j3, j4, j5] that pivot the instrument by `pitch` [rad] about the RCM.

    pitch = 0 is HOME. Valid for pitch in [-0.5, 0.5] rad. The values come from rcm_coupling.csv,
    computed offline (see the notebook).
    """
    pitch_table = _COUPLING[:, 0]
    return np.array([np.interp(pitch, pitch_table, _COUPLING[:, k]) for k in range(1, 5)])


def pitch_range():
    """Smallest and largest pitch [rad] covered by the RCM coupling."""
    return _COUPLING[0, 0], _COUPLING[-1, 0]


# ----------------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------------
def target_frame():
    """Position and rotation matrix of the target frame: origin at TARGET_POSITION, z axis pointing
    to the trocar (the liver is turned so that its thin side faces this axis)."""
    normal = RCM_POSITION - TARGET_POSITION
    z = normal / np.linalg.norm(normal)
    x = np.cross([0.0, 1.0, 0.0], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return TARGET_POSITION.copy(), np.column_stack([x, y, z])


def _box(half_extents, position, rgba, collision=True):
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=(p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
                                 if collision else -1),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba),
        basePosition=list(position),
    )


_DOT_POSITIONS = None    # filled by add_scene: the dots, on the surface of the liver


def add_scene(robot):
    """Add the floor, the robot pedestal, the trocar port, and the liver with its three dots.

    Returns a dict with the body ids: 'floor', 'pedestal', 'port', 'liver' and 'dots' (a list).
    """
    global _DOT_POSITIONS
    floor = p.loadURDF("plane.urdf")
    pedestal = _box([0.06, 0.07, BASE_POSITION[2] / 2], [0.0, 0, BASE_POSITION[2] / 2],
                    [0.45, 0.45, 0.50, 1])
    for link in range(-1, p.getNumJoints(robot)):     # the arm may swing over the pedestal
        p.setCollisionFilterPair(robot, pedestal, link, -1, 0)
    instrument_links = [link_index(robot, "instrument_axis_link"), link_index(robot, "tool_tip_link")]

    # The trocar port: a rigid ring around the instrument at the RCM, normal to the shaft at HOME.
    # Its collision shape is a ring of small spheres; only the instrument shaft can touch it.
    mesh_axis = np.array([0.0, 0.0, 1.0])                     # the ring mesh is around its z axis
    rotation_axis = np.cross(mesh_axis, SHAFT_DIRECTION_HOME)
    angle = np.arccos(np.dot(mesh_axis, SHAFT_DIRECTION_HOME))
    orientation = p.getQuaternionFromAxisAngle((rotation_axis / np.linalg.norm(rotation_axis)).tolist(), angle)
    n_spheres = 24
    sphere_angles = 2 * np.pi * np.arange(n_spheres) / n_spheres
    sphere_centres = [[PORT_RADIUS * np.cos(a), PORT_RADIUS * np.sin(a), 0.0] for a in sphere_angles]
    port_collision = p.createCollisionShapeArray(shapeTypes=[p.GEOM_SPHERE] * n_spheres,
                                                 radii=[PORT_TUBE_RADIUS] * n_spheres,
                                                 collisionFramePositions=sphere_centres)
    port_visual = p.createVisualShape(p.GEOM_MESH, fileName=_lab_file(PORT_FILE),
                                      rgbaColor=[0.35, 0.65, 0.35, 1])
    port = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=port_collision,
                             baseVisualShapeIndex=port_visual, basePosition=RCM_POSITION.tolist(),
                             baseOrientation=orientation)
    for link in range(-1, p.getNumJoints(robot)):
        if link not in instrument_links:
            p.setCollisionFilterPair(robot, port, link, -1, 0)

    # The liver: a fixed body with a soft surface (contact stiffness and damping); only the instrument
    # touches it.
    origin, rotation = target_frame()
    liver_orientation = _matrix_to_quaternion(rotation)
    scale = [LIVER_SCALE] * 3
    liver = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_MESH, fileName=_lab_file(LIVER_FILE),
                                                       meshScale=scale, flags=p.GEOM_FORCE_CONCAVE_TRIMESH),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_MESH, fileName=_lab_file(LIVER_FILE), meshScale=scale,
                                                 rgbaColor=[0.62, 0.22, 0.2, 1]),
        basePosition=(origin - LIVER_DEPTH * rotation[:, 2] + LIVER_SHIFT[0] * rotation[:, 0]
                      + LIVER_SHIFT[1] * rotation[:, 1]).tolist(),
        baseOrientation=liver_orientation)
    p.changeDynamics(liver, -1, contactStiffness=LIVER_STIFFNESS, contactDamping=LIVER_DAMPING,
                     lateralFriction=0.1)
    for link in range(-1, p.getNumJoints(robot)):
        if link not in instrument_links:
            p.setCollisionFilterPair(robot, liver, link, -1, 0)

    # The dots: drawn only, on the liver surface where the lines from the trocar to the triangle hit it
    _DOT_POSITIONS = np.zeros((3, 3))
    dots = []
    for k, (colour, angle) in enumerate(zip(DOT_COLOURS, DOT_ANGLES)):
        aim = origin + DOT_DISTANCE * (np.cos(angle) * rotation[:, 0] + np.sin(angle) * rotation[:, 1])
        far = RCM_POSITION + 2.0 * (aim - RCM_POSITION)
        hit = p.rayTest(RCM_POSITION.tolist(), far.tolist())[0]
        if hit[0] != liver:
            raise RuntimeError(f"the {colour} dot is not on the liver")
        position, normal = np.array(hit[3]), np.array(hit[4])
        _DOT_POSITIONS[k] = position
        z = normal / np.linalg.norm(normal)
        x = np.cross([0.0, 1.0, 0.0], z)
        x = x / np.linalg.norm(x)
        dot_rotation = np.column_stack([x, np.cross(z, x), z])
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=DOT_RADIUS, length=0.001,
                                     rgbaColor=DOT_RGBA[colour])
        dots.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
                                      basePosition=(position + 0.0007 * z).tolist(),
                                      baseOrientation=_matrix_to_quaternion(dot_rotation)))
    return {"floor": floor, "pedestal": pedestal, "port": port, "liver": liver, "dots": dots}


def add_patient():
    """Add, for illustration only, a transparent patient lying on an operating table.

    The patient (pybullet_data humanoid, scaled to 1.75 m) lies on its back with the navel at the
    trocar. Nothing here collides with the robot, and the camera sees through it. Skipped on Colab: the simple renderer used there cannot draw
    transparent bodies, so they would hide the target from the camera.
    Returns a dict with the body ids: 'table' and 'patient' (empty on Colab).
    """
    if IN_COLAB:
        print("add_patient: skipped on Colab (transparency is not supported by its renderer)")
        return {}
    scale = 0.27                                   # the humanoid model is about 6.5 units tall
    # model axes -> world: x (front of the body) up, y (towards the head) along +y, z (right side) along -x
    rotation = np.array([[0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0]])
    navel_in_model = np.array([0.36, 0.55, 0.0])  # front of the belly, below the chest, in model units
    base = RCM_POSITION - rotation @ (scale * navel_in_model)
    patient = p.loadURDF("humanoid/humanoid.urdf", basePosition=base.tolist(),
                         baseOrientation=_matrix_to_quaternion(rotation), useFixedBase=True,
                         globalScaling=scale)
    for link in range(-1, p.getNumJoints(patient)):
        p.changeVisualShape(patient, link, rgbaColor=[0.8, 0.85, 0.9, 0.5])
        p.setCollisionFilterGroupMask(patient, link, 0, 0)

    back_height = base[2] - scale * 0.40          # z of the patient's back [m]
    table_top = min(back_height, 0.40)            # below the patient and the liver [m]
    table_half = [0.27, 0.95, table_top / 2]
    table = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=table_half,
                                                 rgbaColor=[0.55, 0.62, 0.66, 1]),
        basePosition=[RCM_POSITION[0], base[1] - 0.07, table_top / 2],
    )

    return {"table": table, "patient": patient}


def port_force(robot, port):
    """Force sensor on the trocar port: total contact force of the instrument on the port [N].

    Returns the force vector (3,) in the world frame. Its norm is 0 when the instrument does not
    touch the port. Call it after p.stepSimulation().
    """
    global COLLIDED
    force = np.zeros(3)
    for contact in p.getContactPoints(bodyA=robot, bodyB=port):
        normal_on_port = np.array(contact[7])        # contact normal, pointing from the port to the robot
        normal_force = contact[9]                    # [N]
        force = force - normal_force * normal_on_port
        if np.linalg.norm(force)>1:
            #print('collision!')
            for link in range(-1, p.getNumJoints(port)):
                if not COLLIDED:
                    p.changeVisualShape(port, link, rgbaColor=[0.8, 0.35, 0.35, 0.5])
                    COLLIDED=True
        else:
            for link in range(-1, p.getNumJoints(port)):
                if COLLIDED:
                    p.changeVisualShape(port, link, rgbaColor=[0.35, 0.65, 0.35, 0.5])
                    COLLIDED=False

    return force


def dot_positions():
    """3D positions of the three dots in the world, (3, 3) [m], in the order yellow, green, blue."""
    if _DOT_POSITIONS is None:
        raise RuntimeError("call add_scene first")
    return _DOT_POSITIONS.copy()


def tip_wrench(robot):
    """Force/torque sensor at the instrument tip (on the fixed joint tool_tip_fixed).

    Returns (force, torque) that the tip applies to whatever it touches, in the tip frame
    (z along the instrument, pointing out of the tip): force [N], torque [N m] about the tip frame
    origin. force[2] > 0 means the tip pushes forward. Only contacts on the last centimetre of the
    shaft are felt. Both are zero without contact. Call it after p.stepSimulation().
    """
    reading = p.getJointState(robot, joint_index(robot, "tool_tip_fixed"))[2]
    force = np.array(reading[:3])
    torque = np.array(reading[3:])
    return force, torque


def _matrix_to_quaternion(rotation):
    """Quaternion (x, y, z, w) of a rotation matrix."""
    m = rotation
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2
    x = np.copysign(x, m[2, 1] - m[1, 2])
    y = np.copysign(y, m[0, 2] - m[2, 0])
    z = np.copysign(z, m[1, 0] - m[0, 1])
    return [x, y, z, w]


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
def camera_matrix():
    """Intrinsic matrix K of the simulated camera (pinhole model, square pixels) [pixels]."""
    fy = (IMAGE_HEIGHT / 2) / np.tan(np.radians(FOV_Y_DEG) / 2)
    fx = fy
    cx = IMAGE_WIDTH / 2
    cy = IMAGE_HEIGHT / 2
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])


def get_camera_image(robot):
    """Take a picture with the camera on the instrument.

    Returns
      rgb   : (IMAGE_HEIGHT, IMAGE_WIDTH, 3) uint8 colour image
      depth : (IMAGE_HEIGHT, IMAGE_WIDTH) distance along the optical axis of each pixel [m]
    """
    position, rotation = link_pose(robot, "camera_link")
    forward = rotation[:, 2]
    up = -rotation[:, 1]              # image y points down, so "up" is -y
    view = p.computeViewMatrix(position, position + forward, up)
    projection = p.computeProjectionMatrixFOV(FOV_Y_DEG, IMAGE_WIDTH / IMAGE_HEIGHT, NEAR, FAR)
    connection = p.getConnectionInfo()["connectionMethod"]
    renderer = p.ER_BULLET_HARDWARE_OPENGL if connection == p.GUI else p.ER_TINY_RENDERER
    _, _, rgba, depth_buffer, _ = p.getCameraImage(IMAGE_WIDTH, IMAGE_HEIGHT, view, projection,
                                                   renderer=renderer)
    rgb = np.reshape(rgba, (IMAGE_HEIGHT, IMAGE_WIDTH, 4))[:, :, :3].astype(np.uint8)
    depth_buffer = np.reshape(depth_buffer, (IMAGE_HEIGHT, IMAGE_WIDTH))
    depth = FAR * NEAR / (FAR - (FAR - NEAR) * depth_buffer)   # OpenGL depth buffer to metres
    return rgb, depth


def project(points_world, camera_position, camera_rotation):
    """Pixel coordinates (u, v) of 3D points, (N, 2), for a camera at this pose (pinhole model)."""
    points_camera = (np.asarray(points_world) - camera_position) @ camera_rotation   # world -> camera frame
    homogeneous = points_camera @ camera_matrix().T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def find_dots(rgb):
    """Pixel coordinates (u, v) of the centre of the yellow, green and blue dots, (3, 2) array.

    A row is NaN if that dot is not visible.
    """
    image = rgb.astype(int)
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    # a pixel belongs to a dot if its colour is bright and clearly saturated (yellow = red + green, no blue)
    masks = {
        "yellow": (red > 120) & (green > 0.7 * red) & (blue < 0.3 * green),
        "green": (green > 100) & (red < 0.3 * green) & (blue < 0.3 * green),
        "blue": (blue > 120) & (red < 0.3 * blue) & (green < 0.3 * blue),
    }
    centres = np.full((3, 2), np.nan)
    for k, colour in enumerate(DOT_COLOURS):
        rows, columns = np.nonzero(masks[colour])
        if len(rows) > 0:
            centres[k] = [columns.mean(), rows.mean()]
    return centres


def _aligned_coordinates(point):
    """swing, pitch that point the instrument axis at a point (used for the target overlay)."""
    u = np.asarray(point) - RCM_POSITION
    u = u / np.linalg.norm(u)
    beta = np.arcsin(u[0])                       # angle from vertical, in the tilted plane
    swing = np.arctan2(-u[1], -u[2])             # rotation of that plane about -x
    beta_home = np.arcsin(SHAFT_DIRECTION_HOME[0])
    return swing, beta - beta_home


def desired_dot_pixel(robot, colour, insertion=0.0):
    """Where the dot of this colour must appear in the image when the instrument points at it.

    Computed for the given insertion j6 [m]. Returns the pixel (u, v) as an array of 2 values.
    """
    k = DOT_COLOURS.index(colour)
    dot = dot_positions()[k]
    swing, pitch = _aligned_coordinates(dot)
    saved = get_joints(robot)
    set_joints(robot, [swing, *parallelogram_joints(pitch), insertion])
    camera_position, camera_rotation = link_pose(robot, "camera_link")
    set_joints(robot, saved)
    return project(dot[None, :], camera_position, camera_rotation)[0]


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
_PLOT_COLOURS = {"yellow": "#ffd000", "green": "#20c020", "blue": "#3060ff"}


def show_image(image, title="", dots=None, targets=None):
    """Show an image, optionally with the dots found and where they should be.

    dots    : (3, 2) array of pixels (u, v) of the yellow, green and blue dots (as from find_dots), drawn as circles
    targets : dict {colour: pixel (u, v)}, drawn as crosses (for example {"blue": desired_dot_pixel(...)})
    """
    if image.ndim == 2:
        fig = px.imshow(image, color_continuous_scale="viridis", title=title)
    else:
        fig = px.imshow(image, title=title)
    if dots is not None:
        for k, colour in enumerate(DOT_COLOURS):
            if not np.any(np.isnan(dots[k])):
                fig.add_trace(go.Scatter(x=[dots[k][0]], y=[dots[k][1]], mode="markers", name=f"{colour} dot",
                                         marker=dict(symbol="circle-open", size=18, color=_PLOT_COLOURS[colour],
                                                     line=dict(width=3, color=_PLOT_COLOURS[colour]))))
    if targets is not None:
        for colour, pixel in targets.items():
            fig.add_trace(go.Scatter(x=[pixel[0]], y=[pixel[1]], mode="markers", name=f"{colour} target",
                                     marker=dict(symbol="x-thin-open", size=18, color="white",
                                                 line=dict(width=3, color=_PLOT_COLOURS[colour]))))
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig.show()


def plot_series(time, series, xlabel="time [s]", ylabel="", title=""):
    """Plot one or more time series. series is a dict {label: list of values}."""
    fig = go.Figure()
    for label, values in series.items():
        fig.add_trace(go.Scatter(x=time, y=values, mode="lines", name=label))
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, title=title,
                      margin=dict(l=10, r=10, t=40, b=10))
    fig.show()
