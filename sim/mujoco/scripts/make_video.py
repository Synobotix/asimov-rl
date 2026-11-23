"""Record videos of what the robot does, so the result can be seen without running anything.

Writes docs/demo_policy.mp4 and docs/demo_policy.gif (the trained PPO agent reaching targets).

Usage (from the project root, after `python scripts/train_ppo.py`):
    python scripts/make_video.py                       # uses output/ppo_reach.zip
    python scripts/make_video.py --model output/ppo_reach --targets 6
On a machine without display (Linux server, Kaggle) set MUJOCO_GL=egl first.
"""
import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO

from mujoco_sim.envs import ArmReachEnv

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="output/ppo_reach")
parser.add_argument("--targets", type=int, default=6, help="number of targets to reach")
parser.add_argument("--out", default="docs/demo_policy")
parser.add_argument("--size", type=int, default=480)
parser.add_argument("--seed", type=int, default=7)
args = parser.parse_args()

env = ArmReachEnv()
policy = PPO.load(args.model, device="cpu")
renderer = mujoco.Renderer(env.model, args.size, args.size)

camera = mujoco.MjvCamera()
camera.lookat[:] = [0.15, 0.0, 0.35]
camera.distance = 1.5
camera.azimuth = 90  # look along the y axis: the arm moves in the x-z plane
camera.elevation = 0


def draw_frame() -> np.ndarray:
    """Render the arm and a green sphere at the target position."""
    renderer.update_scene(env.data, camera=camera)
    scene = renderer.scene
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.025, 0, 0]),
        np.array([env.target[0], 0.0, env.target[1]]),
        np.eye(3).flatten(),
        np.array([0.1, 0.8, 0.2, 1.0]),
    )
    scene.ngeom += 1
    return renderer.render().copy()


frames = []
reached = 0
obs, _ = env.reset(seed=args.seed)
for target_id in range(args.targets):
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        frames.append(draw_frame())
    reached += int(info["is_success"])
    frames.extend([frames[-1]] * 5)  # short pause on the reached target
    obs, _ = env.reset()  # next target; the arm restarts from the upright pose

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(out.with_suffix(".mp4"), frames, fps=20, macro_block_size=1)
imageio.mimsave(out.with_suffix(".gif"), frames[::2], fps=10, loop=0)
print(f"{len(frames)} frames, {reached}/{args.targets} targets reached -> {out}.mp4 and {out}.gif")
