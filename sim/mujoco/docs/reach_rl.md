# Arm reaching with PPO

The two-joint arm of `robots/arm.urdf` must bring its tip within 4 cm of a random target.

- Environment: `mujoco_sim.envs.ArmReachEnv` (Gymnasium).
- Action: target joint angles, normalized to [-1, 1] over each joint range.
- Observation (8 values): joint positions, joint velocities, target position, tip-to-target vector.
- Reward: minus the tip-to-target distance, plus 1 on success.

## Train and evaluate

```bash
pip install -r requirements-rl.txt
python scripts/train_ppo.py --steps 300000
```

The script saves the agent to `output/ppo_reach.zip` and compares it with two baselines on 200 targets never seen during training.

## Results

300 000 steps, CPU, about 3 minutes. Simulation only: nothing here says how a real arm would behave.

| policy | success | final distance |
|---|---|---|
| zero action | 0.00 | 0.445 m |
| random | 0.09 | 0.427 m |
| PPO | 1.00 | 0.030 m |

With 100 000 steps, PPO reaches only 0.41 success (final distance 0.140 m).
