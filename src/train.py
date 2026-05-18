from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import gymnasium as gym

# ── local imports ─────────────────────────────────────────────────────────────
try:
    from simulator import DataCenterEnv, N_SERVERS, FLEET
except ModuleNotFoundError:
    sys.exit("Could not import DataCenterEnv")

from rl_agent import CarbonSchedulerCallback

# ── SB3 ───────────────────────────────────────────────────────────────────────
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure


# =========================================================
# CONFIG
# =========================================================
TOTAL_TIMESTEPS = 300_000
MODEL_SAVE_PATH = "dqn_carbon_scheduler"
CHECKPOINT_DIR = "checkpoints"
EVAL_DIR = "eval_logs"
TENSORBOARD_DIR = "tb_logs/dqn_carbon_scheduler"

# =========================================================
# DQN PARAMS
# =========================================================
DQN_HYPERPARAMS = dict(
    policy="MlpPolicy",
    learning_rate=1e-4,
    buffer_size=200_000,
    learning_starts=5_000,
    batch_size=256,
    gamma=0.995,
    train_freq=2,
    target_update_interval=1000,
    exploration_fraction=0.35,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.04,
    policy_kwargs=dict(net_arch=[256, 256, 128]),
    verbose=0,
)


# =========================================================
# FIXED WRAPPER (IMPORTANT)
# =========================================================
class ImprovedDataCenterWrapper(gym.Wrapper):

    def __init__(self, env: DataCenterEnv, reward_scale: float = 1.0):
        super().__init__(env)
        self.reward_scale = reward_scale
        self._last_completed = 0
        self._last_missed = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_completed = 0
        self._last_missed = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaping = 0.0

        # throughput shaping
        new_completed = info["completed"] - self._last_completed
        new_missed = info["missed_sla"] - self._last_missed

        self._last_completed = info["completed"]
        self._last_missed = info["missed_sla"]

        shaping += 0.15 * new_completed
        shaping -= 0.25 * new_missed

        total_reward = (reward + shaping) / self.reward_scale

        return obs, total_reward, terminated, truncated, info


# =========================================================
# ENV FACTORY
# =========================================================
def make_env(seed=0):
    env = DataCenterEnv(seed=seed)
    env = ImprovedDataCenterWrapper(env)
    env = Monitor(env)
    return env


# =========================================================
# VALIDATION
# =========================================================
def validate_env(seed):
    print("Validating environment...")
    env = DataCenterEnv(seed=seed)
    check_env(env, warn=True)

    for i in range(3):
        obs, _ = env.reset()
        done = False
        total = 0

        while not done:
            action = env.action_space.sample()
            obs, r, t, tr, info = env.step(action)
            total += r
            done = t or tr

        print(f"Episode {i+1}: reward={total:.2f}")

    env.close()
    print("Env OK\n")


# =========================================================
# TRAIN
# =========================================================
def train(seed=42):

    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    Path(EVAL_DIR).mkdir(exist_ok=True)

    env = make_vec_env(lambda: make_env(seed), n_envs=1)
    eval_env = make_vec_env(lambda: make_env(seed + 999), n_envs=1)

    model = DQN(
        env=env,
        seed=seed,
        tensorboard_log=TENSORBOARD_DIR,
        **DQN_HYPERPARAMS,
    )

    logger = configure(TENSORBOARD_DIR, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    callbacks = CallbackList([
        CarbonSchedulerCallback(),
        CheckpointCallback(save_freq=25000, save_path=CHECKPOINT_DIR),
        EvalCallback(eval_env, best_model_save_path=EVAL_DIR, eval_freq=10000)
    ])

    print("Training...")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
    )

    model.save(MODEL_SAVE_PATH)
    print("Model saved!")

    env.close()
    eval_env.close()

    return model


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    validate_env(42)
    train()