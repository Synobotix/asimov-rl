"""Load a URDF, simulate it and save a video of the camera view.

Usage (from the project root):
    python scripts/run_demo.py                      # robots/arm.urdf, saves output/demo.mp4
    python scripts/run_demo.py --no-render          # physics only, no graphics needed
    python scripts/run_demo.py --urdf robots/other.urdf
"""
import argparse
from pathlib import Path

import imageio
import numpy as np

from mujoco_sim.scene import build_model, rollout

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="robots/arm.urdf")
parser.add_argument("--out", default="output/demo.mp4")
parser.add_argument("--no-render", action="store_true")
args = parser.parse_args()

model = build_model(args.urdf)
print(f"bodies={model.nbody} joints={model.njnt} actuators={model.nu} cameras={model.ncam}")
frames, qpos = rollout(model, n_frames=40, render=not args.no_render)
print("joint positions per frame:\n", np.round(qpos, 3))

if frames is not None:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=10, macro_block_size=1)
    print("video saved:", args.out, f"({len(frames)} frames)")
