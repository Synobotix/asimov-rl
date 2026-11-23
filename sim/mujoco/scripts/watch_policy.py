"""Watch a trained PPO agent reach targets in the MuJoCo viewer.

Usage (from the project root, after `python scripts/train_ppo.py`):
    python scripts/watch_policy.py                      # output/ppo_reach.zip
    python scripts/watch_policy.py --model output/ppo_reach --episodes 20

Mouse: left button rotates, right button moves, wheel zooms. Close the window to quit.
"""
import argparse
import time

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from mujoco_sim.envs import ArmReachEnv

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="output/ppo_reach")
parser.add_argument("--episodes", type=int, default=20)
args = parser.parse_args()

env = ArmReachEnv()
policy = PPO.load(args.model, device="cpu")
dt = env.model.opt.timestep * env.frame_skip  # real duration of one environment step

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        while not done and viewer.is_running():
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            viewer.sync()
            time.sleep(dt)
        if not viewer.is_running():
            break
        print(f"episode {episode + 1}: {'reached' if info['is_success'] else 'missed'}, distance {info['distance']:.3f} m")
        time.sleep(0.5)
