"""Open an interactive window showing a robot loaded from a URDF file.

Usage (from the project root, environment active):
    python scripts/view.py                          # robots/arm.urdf
    python scripts/view.py --urdf robots/other.urdf

Mouse: left button rotates, right button moves, wheel zooms. Space pauses the simulation.
"""
import argparse

import mujoco
import mujoco.viewer

from mujoco_sim.scene import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="robots/arm.urdf")
args = parser.parse_args()

model = build_model(args.urdf)
print(f"Loaded {args.urdf}: {model.njnt} joints, {model.nu} motors. Close the window to quit.")
mujoco.viewer.launch(model, mujoco.MjData(model))
