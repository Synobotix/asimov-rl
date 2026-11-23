"""Make Asimov v1 stand and record a video.

Usage (from sim/mujoco/):
    python scripts/asimov_stand.py --push 100
Writes docs/asimov_stand.mp4 and the matching .gif (needs MUJOCO_GL=egl on a machine without display).
"""
import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np

from mujoco_sim.humanoid import (
    DEFAULT_ASIMOV_XML,
    StandingController,
    build_asimov,
    place_on_floor,
)

parser = argparse.ArgumentParser()
parser.add_argument("--xml", default=str(DEFAULT_ASIMOV_XML))
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument("--push", type=float, default=0.0, help="forward push (N) applied to the pelvis at 4 s for 0.1 s")
parser.add_argument("--out", default="docs/asimov_stand.mp4")
args = parser.parse_args()

model = build_asimov(args.xml)
data = mujoco.MjData(model)
place_on_floor(model, data)
controller = StandingController(model)
renderer = mujoco.Renderer(model, 480, 640)
camera = mujoco.MjvCamera()
camera.lookat[:] = [0.0, 0.0, 0.5]
camera.distance = 2.6
camera.azimuth = 150
camera.elevation = -10

frames, step = [], 0
pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
while data.time < args.seconds:
    controller(data)
    data.xfrc_applied[pelvis, 0] = args.push if 4.0 <= data.time < 4.1 else 0.0
    mujoco.mj_step(model, data)
    step += 1
    if step % 8 == 0:  # 40 ms of simulated time per frame -> 25 fps in real time
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(args.out, frames, fps=25, macro_block_size=1)
gif = Path(args.out).with_suffix(".gif")  # small GIF shown in the README
imageio.mimsave(gif, [f[::2, ::2] for f in frames[::3]], fps=8, loop=0)
standing = data.qpos[2] > 0.5
print(f"{len(frames)} frames; pelvis height at the end {data.qpos[2]:.2f} m; {'standing' if standing else 'FELL'} -> {args.out} and {gif}")
