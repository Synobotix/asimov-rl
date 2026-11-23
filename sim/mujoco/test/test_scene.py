from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_sim.scene import build_model, rollout

URDF = Path(__file__).parent.parent / "robots" / "arm.urdf"


def make_model():
    return build_model(URDF)


def test_model_structure():
    model = make_model()
    assert model.nq == 2  # two joints
    assert model.nu == 2  # one motor per joint
    assert model.ncam == 1  # camera "front"


def test_rollout_shapes_and_motion():
    _, qpos = rollout(make_model(), n_frames=8, render=False)
    assert qpos.shape == (8, 2)
    assert np.abs(qpos).max() > 0.01  # the joints actually moved


def test_rollout_is_deterministic():
    _, first = rollout(make_model(), render=False)
    _, second = rollout(make_model(), render=False)
    assert np.array_equal(first, second)


def test_no_frames_when_render_disabled():
    frames, _ = rollout(make_model(), render=False)
    assert frames is None


def test_missing_urdf_raises():
    with pytest.raises(Exception):
        build_model("robots/does_not_exist.urdf")


def test_render_frames():
    try:
        frames, _ = rollout(make_model(), n_frames=2, size=64, render=True)
    except Exception as error:  # no graphics backend on this machine
        pytest.skip(f"graphics backend unavailable: {error}")
    assert frames.shape == (2, 64, 64, 3)
    assert frames.dtype == np.uint8


def test_position_servos_reach_their_target():
    model = build_model(URDF, control="position")
    data = mujoco.MjData(model)
    data.ctrl[:] = [0.5, -0.5]
    for _ in range(1500):
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qpos, [0.5, -0.5], atol=0.05)


def test_tip_site_is_added_on_request():
    assert build_model(URDF).nsite == 0
    model = build_model(URDF, tip_body="fore", tip_offset=(0.0, 0.0, 0.25))
    assert model.nsite == 1


def test_floor_is_optional():
    assert build_model(URDF, floor=False).ngeom == build_model(URDF).ngeom - 1


def test_unknown_control_mode_raises():
    with pytest.raises(ValueError):
        build_model(URDF, control="velocity")
