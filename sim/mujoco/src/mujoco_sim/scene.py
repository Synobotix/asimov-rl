"""Build a MuJoCo scene from a URDF robot file and simulate it."""

import mujoco
import numpy as np


def build_model(
    urdf_path: str,
    control: str = "torque",
    tip_body: str | None = None,
    tip_offset: tuple = (0.0, 0.0, 0.0),
    floor: bool = True,
    kp: float = 50.0,
    kv: float = 8.0,
    max_force: float = 20.0,
) -> mujoco.MjModel:
    """Load a URDF and complete it with what URDF cannot describe.

    URDF has no floor, light, camera or motor, so they are added here:
    a floor, a light, a camera named "front", and one motor per joint.

    Args:
        urdf_path (str): Path to the URDF file.
        control (str): "torque" for direct torque motors (ctrl in [-5, 5]), or "position"
            for position servos (ctrl = target joint angle within the joint range).
        tip_body (str | None): Name of the body that carries the end effector. If given,
            a site named "tip" is added to it.
        tip_offset (tuple): Position of the "tip" site in the frame of `tip_body`.
        floor (bool): Whether to add a floor.
        kp (float): Stiffness of the position servos.
        kv (float): Damping of the position servos.
        max_force (float): Torque limit of the position servos.

    Returns:
        mujoco.MjModel: Compiled model ready for simulation.
    """
    if control not in ("torque", "position"):
        raise ValueError(f"control must be 'torque' or 'position', got {control!r}")
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    if control == "position":
        # The servo damping is stiff for these light links: with the default explicit
        # Euler integrator the joints chatter and never settle. Implicit integration does.
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if floor:
        spec.worldbody.add_geom(
            name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], rgba=[0.8, 0.8, 0.8, 1]
        )
    spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1])
    spec.worldbody.add_camera(name="front", pos=[1.2, 0, 0.5], xyaxes=[0, 1, 0, -0.3, 0, 1])
    for joint in spec.joints:
        motor = spec.add_actuator(name=joint.name, target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        if control == "torque":
            motor.ctrllimited = True
            motor.ctrlrange = [-5, 5]
        else:
            motor.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            motor.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            motor.gainprm[0] = kp
            motor.biasprm[1] = -kp
            motor.biasprm[2] = -kv
            motor.ctrllimited = True
            motor.ctrlrange = list(joint.range)
            motor.forcelimited = True
            motor.forcerange = [-max_force, max_force]
    if tip_body is not None:
        spec.body(tip_body).add_site(name="tip", pos=list(tip_offset), size=[0.01, 0, 0])
    return spec.compile()


def rollout(
    model: mujoco.MjModel,
    n_frames: int = 8,
    steps_per_frame: int = 50,
    size: int = 224,
    render: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Move the joints with a sine torque and record the state (and optionally images).

    Args:
        model (mujoco.MjModel): Model returned by `build_model` (torque control).
        n_frames (int): Number of recorded frames.
        steps_per_frame (int): Physics steps between two frames.
        size (int): Height and width of the rendered images.
        render (bool): If False, only the physics runs (no graphics needed).

    Returns:
        tuple: (frames, qpos). `frames` has shape (n_frames, size, size, 3), dtype uint8,
            or is None when render is False. `qpos` has shape (n_frames, n_joints).
    """
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, size, size) if render else None
    frames, qpos = [], []
    for i in range(n_frames):
        data.ctrl[:] = 3.0 * np.sin(0.8 * i + np.arange(model.nu))
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        qpos.append(data.qpos.copy())
        if renderer is not None:
            renderer.update_scene(data, camera="front")
            frames.append(renderer.render().copy())
    return (np.stack(frames) if render else None), np.stack(qpos)
