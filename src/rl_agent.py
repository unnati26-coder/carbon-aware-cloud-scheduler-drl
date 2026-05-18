"""
rl_agent.py
───────────
Contains:
  • CarbonSchedulerCallback — SB3 BaseCallback that logs reward, CO₂, and SLA
    violation rate to both the console and TensorBoard every `log_freq` steps.
  • DataCenterWrapper        — thin gymnasium.Wrapper that normalises the reward
    signal and optionally adds action-masking info to the observation.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any
import gymnasium as gym

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


# ──────────────────────────────────────────────────────────────────────────────
# Logging callback
# ──────────────────────────────────────────────────────────────────────────────

class CarbonSchedulerCallback(BaseCallback):
    """
    Logs three custom metrics every `log_freq` *environment* steps:

    ┌─────────────────────────────┬───────────────────────────────────────────┐
    │ Key                         │ Description                               │
    ├─────────────────────────────┼───────────────────────────────────────────┤
    │ custom/mean_reward          │ Mean episode reward over last `window`     │
    │ custom/mean_co2_g           │ Mean total CO₂ (g) per completed episode  │
    │ custom/sla_violation_rate   │ Missed SLA tasks / total completed tasks  │
    │ custom/mean_completed_tasks │ Mean tasks finished per episode           │
    │ custom/carbon_intensity     │ Latest grid carbon intensity (gCO₂/kWh)  │
    └─────────────────────────────┴───────────────────────────────────────────┘

    It reads episode-level statistics from the `info` dict emitted by
    DataCenterEnv (keys: total_co2_g, missed_sla, completed, carbon_intensity).

    Parameters
    ----------
    log_freq : int
        How often (in env steps) to write metrics. Default 1 000.
    window : int
        Rolling window size for smoothing. Default 10 episodes.
    verbose : int
        0 = silent, 1 = print to stdout on each log step.
    """

    def __init__(
        self,
        log_freq: int = 1_000,
        window: int = 10,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.log_freq = log_freq
        self.window   = window

        # Rolling buffers — one entry per *completed episode*
        self._ep_rewards:   deque[float] = deque(maxlen=window)
        self._ep_co2:       deque[float] = deque(maxlen=window)
        self._ep_sla_rate:  deque[float] = deque(maxlen=window)
        self._ep_completed: deque[int]   = deque(maxlen=window)

        # Per-episode accumulators (reset at episode start)
        self._current_ep_reward:    float = 0.0
        self._current_ep_steps:     int   = 0
        self._last_log_step:        int   = 0
        self._latest_carbon:        float = 0.0

    # ── SB3 callback hooks ────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        """Called after every env.step(); return False to abort training."""

        # Accumulate reward for current episode
        # `self.locals["rewards"]` is a (n_envs,) array from VecEnv
        reward = float(np.mean(self.locals["rewards"]))
        self._current_ep_reward += reward
        self._current_ep_steps  += 1

        # Grab latest carbon intensity from info (list of dicts, one per env)
        infos: list[dict[str, Any]] = self.locals.get("infos", [{}])
        if infos:
            self._latest_carbon = float(
                infos[0].get("carbon_intensity", self._latest_carbon)
            )

        # Detect episode termination / truncation
        dones     = self.locals.get("dones", [False])
        truncated = self.locals.get("truncateds", [False])  # SB3 ≥ 2.0

        episode_ended = bool(dones[0]) or bool(
            truncated[0] if hasattr(truncated, "__getitem__") else truncated
        )

        if episode_ended and infos:
            info = infos[0]
            # Final-episode statistics from DataCenterEnv._build_info()
            co2_g    = float(info.get("total_co2_g", 0.0))
            missed   = int(  info.get("missed_sla",  0))
            finished = int(  info.get("completed",   1))   # avoid /0

            sla_rate = missed / max(finished, 1)

            self._ep_rewards.append(self._current_ep_reward)
            self._ep_co2.append(co2_g)
            self._ep_sla_rate.append(sla_rate)
            self._ep_completed.append(finished)

            # Reset accumulators
            self._current_ep_reward = 0.0
            self._current_ep_steps  = 0

        # Periodic logging ───────────────────────────────────────────────────
        if (
            self.num_timesteps - self._last_log_step >= self.log_freq
            and len(self._ep_rewards) > 0
        ):
            self._last_log_step = self.num_timesteps
            self._write_logs()

        return True   # keep training

    def _on_training_end(self) -> None:
        """Flush any remaining metrics when training finishes."""
        if len(self._ep_rewards) > 0:
            self._write_logs()

    # ── internal ─────────────────────────────────────────────────────────────

    def _write_logs(self) -> None:
        mean_reward    = float(np.mean(self._ep_rewards))
        mean_co2       = float(np.mean(self._ep_co2))
        mean_sla_rate  = float(np.mean(self._ep_sla_rate))
        mean_completed = float(np.mean(self._ep_completed))

        # Write to SB3 logger (TensorBoard + CSV if configured)
        self.logger.record("custom/mean_reward",          mean_reward)
        self.logger.record("custom/mean_co2_g",           mean_co2)
        self.logger.record("custom/sla_violation_rate",   mean_sla_rate)
        self.logger.record("custom/mean_completed_tasks", mean_completed)
        self.logger.record("custom/carbon_intensity",     self._latest_carbon)
        self.logger.dump(step=self.num_timesteps)

        if self.verbose >= 1:
            print(
                f"[{self.num_timesteps:>7} steps] "
                f"reward={mean_reward:>8.3f} | "
                f"CO₂={mean_co2:>7.1f} g | "
                f"SLA_miss={mean_sla_rate:.2%} | "
                f"done={mean_completed:.1f} tasks | "
                f"CI={self._latest_carbon:.0f} gCO₂/kWh"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Environment wrapper
# ──────────────────────────────────────────────────────────────────────────────

class DataCenterWrapper(gym.Wrapper):
    """
    Thin wrapper around DataCenterEnv that:

    1. Reward scaling — divides raw reward by `reward_scale` so gradients are
       well-conditioned for the neural network.  Default 10.0.

    2. Action masking info — appends a boolean mask vector of length
       N_SERVERS + 1 to `info["action_masks"]` at each step.  Servers that
       cannot fit the current head-of-queue task are masked out.  The mask is
       *not* enforced here (that would require a MaskableDQN), but it is
       available for logging / curriculum experiments.

    Usage
    -----
    >>> from simulator import DataCenterEnv
    >>> from rl_agent import DataCenterWrapper
    >>> env = DataCenterWrapper(DataCenterEnv())
    """

    def __init__(self, env: gym.Env, reward_scale: float = 10.0) -> None:
        super().__init__(env)
        self.reward_scale = reward_scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        scaled_reward = reward / self.reward_scale
        info["action_masks"] = self._compute_masks()
        return obs, scaled_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["action_masks"] = self._compute_masks()
        return obs, info

    # ── helpers ───────────────────────────────────────────────────────────────

    def _compute_masks(self) -> np.ndarray:
        """
        Returns a bool array of shape (N_SERVERS + 1,).
        True  = action is *valid* (server can fit head task, or defer).
        False = action would be infeasible (server cannot fit head task).
        """
        unwrapped = self.env.unwrapped
        n = unwrapped.action_space.n
        masks = np.ones(n, dtype=bool)

        queue = getattr(unwrapped, "_queue", [])
        if not queue:
            # No task pending → all assign actions are vacuously valid
            return masks

        head = queue[0]
        for i in range(n - 1):          # last action = defer, always valid
            if not unwrapped._can_fit(i, head):
                masks[i] = False

        return masks


# Make gym importable at the top of this module without a hard dep on the env
try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is required: pip install gymnasium"
    ) from exc