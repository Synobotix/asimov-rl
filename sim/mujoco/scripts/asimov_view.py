"""Watch Asimov v1 standing in a live MuJoCo window.

Usage (from the project root, environment active):
    python scripts/asimov_view.py                   # models/asimov_1_25dof
    python scripts/asimov_view.py --xml path/to/asimov_1_25dof.xml

Mouse: left button rotates, right button moves, wheel zooms. Press P to push the robot
forward (100 N for 0.1 s). Close the window to quit.
"""
import argparse
import time

import mujoco
import mujoco.viewer

from mujoco_sim.humanoid import (
    DEFAULT_ASIMOV_XML,
    StandingController,
    build_asimov,
    place_on_floor,
)

parser = argparse.ArgumentParser()
parser.add_argument("--xml", default=str(DEFAULT_ASIMOV_XML))
args = parser.parse_args()

model = build_asimov(args.xml)
data = mujoco.MjData(model)
place_on_floor(model, data)
controller = StandingController(model)
pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
push_until = [-1.0]


def on_key(keycode: int) -> None:
    if chr(keycode).lower() == "p":
        push_until[0] = data.time + 0.1


with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    while viewer.is_running():
        start = time.time()
        controller(data)
        data.xfrc_applied[pelvis, 0] = 100.0 if data.time < push_until[0] else 0.0
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))
