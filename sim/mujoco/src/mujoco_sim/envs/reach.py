"""Gymnasium environment: move the tip of a robot arm to a target position."""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from mujoco_sim.scene import build_model

DEFAULT_URDF = Path(__file__).resolve().parents[3] / "robots" / "arm.urdf"


class ArmReachEnv(gym.Env):
    """Reaching task for the two-joint arm of `robots/arm.urdf`.

    The arm moves in the x-z plane. A target is sampled in the reachable workspace
    (the tip position of a random joint configuration), and the agent must bring the
    arm tip within `success_radius` of it.

    Action (2,): target joint angles, normalized to [-1, 1] over each joint range
        (position servos).
    Observation (8,): joint positions normalized to [-1, 1], joint velocities,
        target (x, z) and tip-to-target vector (x, z).
    Reward: minus the distance between tip and target, plus 1 on success.
    The episode ends on success (terminated) or after `max_steps` steps (truncated).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_URDF,
        frame_skip: int = 10,
        max_steps: int = 100,
        success_radius: float = 0.04,
        min_target_height: float = 0.05,
        min_target_distance: float = 0.15,
    ) -> None:
        """
        Args:
            urdf_path: URDF of a robot with bodies "fore" (end link) and revolute joints.
            frame_skip: Physics steps per environment step (control period = frame_skip * 0.002 s).
            max_steps: Episode length.
            success_radius: Distance (m) under which the target counts as reached.
            min_target_height: Targets are sampled above this height (m).
            min_target_distance: Targets are at least this far (m) from the starting tip position.
        """
        super().__init__()
        self.model = build_model(
            str(urdf_path), control="position", tip_body="fore", tip_offset=(0.0, 0.0, 0.25), floor=False
        )
        self.data = mujoco.MjData(self.model)
        self._fk_data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.success_radius = success_radius
        self.min_target_height = min_target_height
        self.min_target_distance = min_target_distance
        self.q_low = self.model.jnt_range[:, 0].copy()
        self.q_high = self.model.jnt_range[:, 1].copy()
        n = self.model.nu
        self.action_space = spaces.Box(-1.0, 1.0, (n,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (2 * n + 4,), np.float32)
        self.target = np.zeros(2)
        self._steps = 0

    def tip_position(self, data: mujoco.MjData | None = None) -> np.ndarray:
        """Tip position (x, z) for the given data (default: the current state)."""
        data = self.data if data is None else data
        return data.site_xpos[0][[0, 2]].copy()

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Tip position (x, z) for joint angles `q`, without disturbing the simulation."""
        self._fk_data.qpos[:] = q
        mujoco.mj_forward(self.model, self._fk_data)
        return self.tip_position(self._fk_data)

    def _sample_target(self, start_tip: np.ndarray) -> np.ndarray:
        for _ in range(100):
            q = self.np_random.uniform(self.q_low, self.q_high)
            tip = self.forward_kinematics(q)
            if tip[1] >= self.min_target_height and np.linalg.norm(tip - start_tip) >= self.min_target_distance:
                return tip
        return tip

    def _observation(self) -> np.ndarray:
        q = self.data.qpos
        q_norm = 2.0 * (q - self.q_low) / (self.q_high - self.q_low) - 1.0
        tip = self.tip_position()
        return np.concatenate([q_norm, self.data.qvel, self.target, self.target - tip]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the arm near its upright pose and sample a target.

        `options={"target": [x, z]}` forces the target (used for tests and demos).
        """
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.np_random.uniform(-0.05, 0.05, self.model.nu)
        self.data.ctrl[:] = self.data.qpos
        mujoco.mj_forward(self.model, self.data)
        if options and "target" in options:
            self.target = np.asarray(options["target"], dtype=np.float64)
        else:
            self.target = self._sample_target(self.tip_position())
        self._steps = 0
        return self._observation(), {"distance": float(np.linalg.norm(self.target - self.tip_position()))}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = self.q_low + (action + 1.0) / 2.0 * (self.q_high - self.q_low)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1
        distance = float(np.linalg.norm(self.target - self.tip_position()))
        success = distance < self.success_radius
        reward = -distance + (1.0 if success else 0.0)
        truncated = self._steps >= self.max_steps and not success
        return self._observation(), reward, success, truncated, {"distance": distance, "is_success": success}
