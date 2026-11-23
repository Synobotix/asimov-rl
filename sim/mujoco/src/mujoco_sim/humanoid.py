"""Asimov v1 (25-DOF) humanoid: add position actuators and a standing controller."""

from pathlib import Path

import mujoco
import numpy as np

# Where the Asimov v1 25-DOF MJCF is expected (sim/mujoco/models/asimov_1_25dof).
DEFAULT_ASIMOV_XML = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "asimov_1_25dof"
    / "xml"
    / "asimov_1_25dof.xml"
)


def build_asimov(
    xml_path: str,
    kp_leg: float = 150.0,
    kd_leg: float = 6.0,
    kp_upper: float = 40.0,
    kd_upper: float = 2.0,
) -> mujoco.MjModel:
    """Load the Asimov MJCF and add one position servo per joint.

    The MJCF of `sim-model/asimov_1_25dof` has a floor and a floating base but no actuators
    (the Asimov training code adds them). Torque limits come from the joint `actuatorfrcrange`.

    Args:
        xml_path (str): Path to `asimov_1_25dof.xml` (the meshes folder must be next to `xml/`).
        kp_leg (float): Stiffness of hip, knee and ankle servos.
        kd_leg (float): Damping of hip, knee and ankle servos.
        kp_upper (float): Stiffness of waist, neck and arm servos.
        kd_upper (float): Damping of waist, neck and arm servos.

    Returns:
        mujoco.MjModel: Compiled model with `nu` equal to the number of hinge joints.
    """
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for joint in spec.joints:
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        is_leg = any(part in joint.name for part in ("hip", "knee", "ankle"))
        kp, kd = (kp_leg, kd_leg) if is_leg else (kp_upper, kd_upper)
        motor = spec.add_actuator(name=joint.name, target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        motor.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        motor.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        motor.gainprm[0] = kp
        motor.biasprm[1] = -kp
        motor.biasprm[2] = -kd
        motor.ctrllimited = True
        motor.ctrlrange = list(joint.range)
        motor.forcelimited = True
        motor.forcerange = list(joint.actfrcrange)
    return spec.compile()


class StandingController:
    """Hold the zero pose and keep balance by correcting the ankles from the torso pitch.

    Position servos alone let the robot fall backwards after about 2 s: its centre of mass
    sits near the heels. The ankle target is shifted by `kp * pitch + kd * pitch_rate`.
    """

    def __init__(self, model: mujoco.MjModel, kp: float = 1.0, kd: float = 0.1) -> None:
        self.kp, self.kd = kp, kd
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        self.ankles = [(names.index(f"{side}_ankle_pitch_joint"), sign) for side, sign in (("left", 1), ("right", -1))]
        self.ctrl_low = model.actuator_ctrlrange[:, 0]
        self.ctrl_high = model.actuator_ctrlrange[:, 1]

    @staticmethod
    def pitch(data: mujoco.MjData) -> float:
        """Forward/backward tilt of the pelvis in radians (negative = leaning back)."""
        w, x, y, z = data.qpos[3:7]
        return float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))

    def __call__(self, data: mujoco.MjData) -> None:
        ctrl = np.zeros_like(data.ctrl)
        correction = self.kp * self.pitch(data) + self.kd * data.qvel[4]
        for index, sign in self.ankles:
            ctrl[index] = sign * correction
        data.ctrl[:] = np.clip(ctrl, self.ctrl_low, self.ctrl_high)


def place_on_floor(model: mujoco.MjModel, data: mujoco.MjData, margin: float = 0.0005) -> None:
    """Lower the floating base so that the lowest foot geom touches the floor."""
    mujoco.mj_forward(model, data)
    feet = [g for g in range(model.ngeom) if model.geom_contype[g] and "foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    lowest = min(data.geom_xpos[g][2] - model.geom_size[g][0] for g in feet)
    data.qpos[2] -= lowest - margin
    mujoco.mj_forward(model, data)
