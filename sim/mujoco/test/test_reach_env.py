import numpy as np
import pytest

from mujoco_sim.envs import ArmReachEnv


@pytest.fixture
def env():
    return ArmReachEnv()


def test_reset_shapes(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (8,)
    assert obs.dtype == np.float32
    assert info["distance"] > 0


def test_reset_is_reproducible(env):
    obs1, _ = env.reset(seed=3)
    obs2, _ = env.reset(seed=3)
    np.testing.assert_array_equal(obs1, obs2)


def test_targets_are_reachable_and_above_floor(env):
    for seed in range(20):
        env.reset(seed=seed)
        assert env.target[1] >= env.min_target_height


def test_action_moves_the_tip(env):
    env.reset(seed=0, options={"target": [0.3, 0.3]})
    start = env.tip_position()
    for _ in range(10):
        env.step(np.array([1.0, 1.0]))
    assert np.linalg.norm(env.tip_position() - start) > 0.05


def test_success_terminates_with_bonus(env):
    env.reset(seed=0)
    env.target = env.tip_position()  # target exactly at the tip
    _, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert terminated and not truncated and info["is_success"]
    assert reward > 0.5


def test_truncates_after_max_steps():
    env = ArmReachEnv(max_steps=5)
    env.reset(seed=0, options={"target": [5.0, 5.0]})  # unreachable
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(np.zeros(2))
        assert not (terminated or truncated)
    _, _, terminated, truncated, _ = env.step(np.zeros(2))
    assert truncated and not terminated


def test_forward_kinematics_matches_simulation(env):
    env.reset(seed=0)
    q = np.array([0.4, -0.7])
    expected = env.forward_kinematics(q)
    env.data.qpos[:] = q
    import mujoco

    mujoco.mj_forward(env.model, env.data)
    np.testing.assert_allclose(env.tip_position(), expected)
