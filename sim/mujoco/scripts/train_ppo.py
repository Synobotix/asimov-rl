"""Train a PPO agent on ArmReachEnv and compare it with random and zero-action baselines.

Usage (from the project root):
    python scripts/train_ppo.py                       # 100k steps
    python scripts/train_ppo.py --steps 200000 --out output/ppo_reach
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from mujoco_sim.envs import ArmReachEnv


def evaluate(policy, episodes=200, seed=1000):
    """Return (success rate, mean final distance) of `policy(obs) -> action` on fresh targets."""
    env = ArmReachEnv()
    successes, distances = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
        successes.append(info["is_success"])
        distances.append(info["distance"])
    return float(np.mean(successes)), float(np.mean(distances))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="output/ppo_reach")
    args = parser.parse_args()

    env = make_vec_env(ArmReachEnv, n_envs=4, seed=args.seed)
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=0, device="cpu")
    model.learn(total_timesteps=args.steps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)

    rng = np.random.default_rng(args.seed)
    results = {
        "zero action": evaluate(lambda obs: np.zeros(2)),
        "random": evaluate(lambda obs: rng.uniform(-1, 1, 2)),
        "PPO": evaluate(lambda obs: model.predict(obs, deterministic=True)[0]),
    }
    print(f"{'policy':<12} {'success':>8} {'final distance (m)':>20}")
    for name, (success, distance) in results.items():
        print(f"{name:<12} {success:>8.2f} {distance:>20.3f}")


if __name__ == "__main__":
    main()
