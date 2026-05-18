"""
baselines.py
────────────
Three deterministic baseline schedulers for DataCenterEnv.

  • FCFSScheduler       — First-Come-First-Served: assign to lowest-index
                          server that has capacity.
  • RoundRobinScheduler — Cycle through servers in order; skip if no headroom.
  • GreedyScheduler     — Always pick the least CPU-loaded server with headroom.

Each scheduler implements a common BaseScheduler interface:

    scheduler.run(seed=0)  →  EpisodeResult(co2_g, energy_wh,
                                            sla_violation_rate,
                                            completed, missed_sla,
                                            total_reward, steps)

Usage
─────
    python baselines.py                  # run all three, print comparison table
    python baselines.py --episodes 20    # average over 20 episodes
"""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from statistics import mean, stdev
from typing import Sequence

import numpy as np

try:
    from simulator import DataCenterEnv, N_SERVERS, FLEET
except ModuleNotFoundError:
    sys.exit(
        "[baselines.py] Cannot import DataCenterEnv from simulator.py.\n"
        "Make sure simulator.py is in the same directory."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """Statistics for one completed episode."""
    co2_g:              float
    energy_wh:          float
    sla_violation_rate: float   # missed / completed  ∈ [0, 1]
    completed:          int
    missed_sla:         int
    total_reward:       float
    steps:              int

    def __str__(self) -> str:
        return (
            f"CO₂={self.co2_g:>8.1f} g  "
            f"energy={self.energy_wh:>8.1f} Wh  "
            f"SLA_miss={self.sla_violation_rate:.2%}  "
            f"done={self.completed:>4}  "
            f"reward={self.total_reward:>9.3f}"
        )


@dataclass
class AggregateResult:
    """Mean ± std over multiple episodes."""
    name:                   str
    n_episodes:             int
    co2_g:                  float;  co2_g_std:              float
    energy_wh:              float;  energy_wh_std:          float
    sla_violation_rate:     float;  sla_violation_rate_std: float
    completed:              float;  completed_std:          float
    total_reward:           float;  total_reward_std:       float


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class BaseScheduler(ABC):
    """
    Common interface for all baseline schedulers.

    Subclasses must implement `select_server(env) -> int`, which receives the
    *unwrapped* DataCenterEnv at the moment an action is needed and returns an
    action in [0, N_SERVERS]:  0…N_SERVERS-1 = assign, N_SERVERS = defer.
    """

    name: str = "BaseScheduler"

    def __init__(self) -> None:
        self._reset_state()

    # subclass contract

    @abstractmethod
    def select_server(self, env: DataCenterEnv) -> int:
        """
        Choose an action for the current head-of-queue task.

        Parameters
        ----------
        env : DataCenterEnv (unwrapped)
            Live environment — read `_queue`, `_cpu_used`, `_mem_used`, etc.

        Returns
        -------
        int : server index (0…N_SERVERS-1) or N_SERVERS (defer)
        """
        ...

    def _reset_state(self) -> None:
        """Override to reset scheduler-internal state between episodes."""
        pass

    # public API

    def run(self, seed: int = 0) -> EpisodeResult:
        """
        Execute one full episode with a freshly seeded environment.

        The scheduler is called every timestep only when the queue is
        non-empty; when the queue is empty the defer action is passed
        automatically (no point burning compute on nothing).

        Returns
        -------
        EpisodeResult
        """
        self._reset_state()
        env = DataCenterEnv(seed=seed)
        obs, info = env.reset(seed=seed)

        total_reward = 0.0
        done = False

        while not done:
            unwrapped = env  # DataCenterEnv has no wrapper here
            if unwrapped._queue:
                action = self.select_server(unwrapped)
            else:
                action = N_SERVERS  # defer — nothing to assign

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        env.close()

        finished = max(info["completed"], 1)
        return EpisodeResult(
            co2_g              = info["total_co2_g"],
            energy_wh          = info["total_energy_wh"],
            sla_violation_rate = info["missed_sla"] / finished,
            completed          = info["completed"],
            missed_sla         = info["missed_sla"],
            total_reward       = total_reward,
            steps              = info["step"],
        )

    def run_episodes(
        self, n: int = 10, start_seed: int = 0
    ) -> AggregateResult:
        """Run `n` episodes with consecutive seeds and aggregate statistics."""
        results = [self.run(seed=start_seed + i) for i in range(n)]
        return _aggregate(self.name, results)


# ──────────────────────────────────────────────────────────────────────────────
# 1. First-Come-First-Served (FCFS)
# ──────────────────────────────────────────────────────────────────────────────

class FCFSScheduler(BaseScheduler):
    """
    Assign the head-of-queue task to the first server (by index) that has
    sufficient CPU *and* memory headroom.  If no server fits, defer.

    This mirrors the simplest real-world queue discipline: tasks are
    processed in arrival order and land on whichever resource becomes
    available first (approximated here by lowest index).
    """

    name = "FCFS"

    def select_server(self, env: DataCenterEnv) -> int:
        task = env._queue[0]
        for i in range(N_SERVERS):
            if env._can_fit(i, task):
                return i
        return N_SERVERS   # no server can fit — defer


# ──────────────────────────────────────────────────────────────────────────────
# 2. Round Robin
# ──────────────────────────────────────────────────────────────────────────────

class RoundRobinScheduler(BaseScheduler):
    """
    Maintain a cursor that advances by one each time a task is assigned.
    Scan up to N_SERVERS candidates starting from the cursor; pick the
    first one with headroom.  If none fit, defer and do *not* advance the
    cursor (we'll retry from the same position next step).

    The cursor wraps around modulo N_SERVERS so every server receives
    roughly equal consideration over time.
    """

    name = "RoundRobin"

    def _reset_state(self) -> None:
        self._cursor: int = 0

    def select_server(self, env: DataCenterEnv) -> int:
        task = env._queue[0]
        for offset in range(N_SERVERS):
            idx = (self._cursor + offset) % N_SERVERS
            if env._can_fit(idx, task):
                self._cursor = (idx + 1) % N_SERVERS   # advance past chosen
                return idx
        return N_SERVERS   # full house — defer (cursor unchanged)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Greedy (Least-Loaded)
# ──────────────────────────────────────────────────────────────────────────────

class GreedyScheduler(BaseScheduler):
    """
    Among all servers that can fit the head task, pick the one with the
    *lowest combined utilisation* (average of CPU% and MEM%).

    Intuition: spreading load evenly maximises future capacity and tends to
    keep more servers at lower, more efficient operating points on the
    idle-to-peak power curve — reducing energy and therefore CO₂.
    """

    name = "Greedy"

    def select_server(self, env: DataCenterEnv) -> int:
        task = env._queue[0]
        best_idx:  int   = -1
        best_util: float = float("inf")

        for i, spec in enumerate(FLEET):
            if not env._can_fit(i, task):
                continue
            cpu_util = env._cpu_used[i] / spec.cpu_capacity
            mem_util = env._mem_used[i] / spec.mem_capacity
            combined = (cpu_util + mem_util) / 2.0
            if combined < best_util:
                best_util = combined
                best_idx  = i

        return best_idx if best_idx >= 0 else N_SERVERS


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate(name: str, results: list[EpisodeResult]) -> AggregateResult:
    def _m(attr: str) -> float:
        return mean(getattr(r, attr) for r in results)
    def _s(attr: str) -> float:
        vals = [getattr(r, attr) for r in results]
        return stdev(vals) if len(vals) > 1 else 0.0

    return AggregateResult(
        name                    = name,
        n_episodes              = len(results),
        co2_g                   = _m("co2_g"),
        co2_g_std               = _s("co2_g"),
        energy_wh               = _m("energy_wh"),
        energy_wh_std           = _s("energy_wh"),
        sla_violation_rate      = _m("sla_violation_rate"),
        sla_violation_rate_std  = _s("sla_violation_rate"),
        completed               = _m("completed"),
        completed_std           = _s("completed"),
        total_reward            = _m("total_reward"),
        total_reward_std        = _s("total_reward"),
    )


def compare(
    schedulers: Sequence[BaseScheduler],
    n_episodes: int = 10,
    start_seed: int = 0,
) -> list[AggregateResult]:
    """
    Run every scheduler for `n_episodes` episodes (same seeds) and return
    a list of AggregateResult objects, one per scheduler.
    """
    aggregates = []
    for sched in schedulers:
        print(f"  Running {sched.name} × {n_episodes} episodes …", end=" ", flush=True)
        agg = sched.run_episodes(n=n_episodes, start_seed=start_seed)
        aggregates.append(agg)
        print(f"done  (mean reward={agg.total_reward:+.3f})")
    return aggregates


def print_table(aggregates: list[AggregateResult]) -> None:
    """Print a formatted comparison table to stdout."""
    col_w = 14
    metrics = [
        ("CO₂ (g)",       "co2_g",              "co2_g_std",              ".1f"),
        ("Energy (Wh)",   "energy_wh",           "energy_wh_std",          ".1f"),
        ("SLA miss %",    "sla_violation_rate",  "sla_violation_rate_std", ".2%"),
        ("Completed",     "completed",           "completed_std",          ".1f"),
        ("Reward",        "total_reward",        "total_reward_std",       ".3f"),
    ]

    names = [a.name for a in aggregates]
    header_pad = 14

    # ── header ────────────────────────────────────────────────────────────────
    sep = "─" * (header_pad + col_w * len(names) + 2)
    print(f"\n{sep}")
    print(f"  {'Metric':<{header_pad}}" +
          "".join(f"{n:>{col_w}}" for n in names))
    print(sep)

    # ── rows ──────────────────────────────────────────────────────────────────
    for label, mean_attr, std_attr, fmt in metrics:
        row = f"  {label:<{header_pad}}"
        for agg in aggregates:
            mean_val = getattr(agg, mean_attr)
            std_val  = getattr(agg, std_attr)
            # Format mean ± std
            if fmt.endswith("%"):
                cell = f"{mean_val:.1%}±{std_val:.1%}"
            else:
                cell = f"{mean_val:{fmt}}±{std_val:{fmt}}"
            row += f"{cell:>{col_w}}"
        print(row)

    print(sep)
    n = aggregates[0].n_episodes if aggregates else 0
    print(f"  (mean ± std over {n} episodes, same seeds for all schedulers)\n")

    # ── winner per metric ────────────────────────────────────────────────────
    print("  Best per metric:")
    best_fns = {
        "CO₂ (g)":    min,
        "Energy (Wh)": min,
        "SLA miss %":  min,
        "Completed":   max,
        "Reward":      max,
    }
    for label, mean_attr, *_ in metrics:
        best_fn = best_fns[label]
        winner  = best_fn(aggregates, key=lambda a: getattr(a, mean_attr))
        print(f"    {label:<14}  →  {winner.name}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline scheduler comparison for DataCenterEnv"
    )
    p.add_argument(
        "--episodes", type=int, default=10,
        help="Number of episodes per scheduler (default 10)"
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Starting seed (episodes use seed, seed+1, … seed+N-1)"
    )
    p.add_argument(
        "--single", action="store_true",
        help="Print per-episode results in addition to the aggregate table"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    schedulers: list[BaseScheduler] = [
        FCFSScheduler(),
        RoundRobinScheduler(),
        GreedyScheduler(),
    ]

    if args.single:
        # Verbose mode: one line per episode
        print("\n── Per-episode results ──────────────────────────────────────")
        for sched in schedulers:
            print(f"\n{sched.name}")
            for ep in range(args.episodes):
                result = sched.run(seed=args.seed + ep)
                print(f"  ep{ep+1:02d}  {result}")

    print("\n── Aggregate comparison ─────────────────────────────────────")
    aggregates = compare(schedulers, n_episodes=args.episodes,
                         start_seed=args.seed)
    print_table(aggregates)


if __name__ == "__main__":
    main()