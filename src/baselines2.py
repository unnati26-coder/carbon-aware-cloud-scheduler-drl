from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean, stdev
import numpy as np

from simulator import DataCenterEnv, N_SERVERS, FLEET


# =========================================================
# RESULT CLASS
# =========================================================
@dataclass
class EpisodeResult:
    co2_g: float
    energy_wh: float
    sla_violation_rate: float
    completed: int
    missed_sla: int
    total_reward: float
    steps: int


@dataclass
class AggregateResult:
    name: str
    n_episodes: int
    co2_g: float; co2_g_std: float
    energy_wh: float; energy_wh_std: float
    sla_violation_rate: float; sla_std: float
    completed: float; completed_std: float
    total_reward: float; reward_std: float


# =========================================================
# BASE CLASS
# =========================================================
class BaseScheduler:
    name = "Base"

    def _reset_state(self):
        pass

    def select_action(self, obs, env):
        raise NotImplementedError

    def run(self, seed=0):
        self._reset_state()
        env = DataCenterEnv(seed=seed)
        obs, _ = env.reset(seed=seed)

        total_reward = 0
        done = False

        while not done:
            action = self.select_action(obs, env)
            obs, reward, t, tr, info = env.step(action)
            total_reward += reward
            done = t or tr

        env.close()

        finished = max(info["completed"], 1)

        return EpisodeResult(
            co2_g=info["total_co2_g"],
            energy_wh=info["total_energy_wh"],
            sla_violation_rate=info["missed_sla"] / finished,
            completed=info["completed"],
            missed_sla=info["missed_sla"],
            total_reward=total_reward,
            steps=info["step"],
        )


# =========================================================
# LACS
# =========================================================
# Partition the fleet into N_LOCS synthetic "locations".
# Each location gets a fixed carbon bias that offsets the shared global
# carbon signal — simulating data centres on grids with different mixes
# (e.g. one region is heavier on renewables, another on fossil fuel).
# LACS then picks the *location* with the lowest effective carbon first,
# and within that location picks the least-loaded server — matching the
# core idea of routing work to the cleanest region.

_N_LOCS = 3
# Relative carbon bias per location (clean → dirty)
_LOC_CARBON_BIAS = np.array([-0.15, 0.0, 0.15])


class LACSScheduler(BaseScheduler):
    name = "LACS"

    def select_action(self, obs, env):
        if not env._queue:
            return N_SERVERS

        task = env._queue[0]
        base_carbon = float(obs[65])

        best_server = -1
        best_score = -1e9

        for loc in range(_N_LOCS):
            # Effective carbon intensity for this location
            loc_carbon = float(np.clip(base_carbon + _LOC_CARBON_BIAS[loc], 0.0, 1.0))

            # Servers assigned to this location by round-robin partitioning
            loc_servers = [i for i in range(N_SERVERS) if i % _N_LOCS == loc]

            for i in loc_servers:
                if not env._can_fit(i, task):
                    continue

                spec = FLEET[i]
                load = (env._cpu_used[i] / spec.cpu_capacity +
                        env._mem_used[i] / spec.mem_capacity) / 2

                # Carbon is the primary axis; load is a small tie-breaker
                score = -loc_carbon + 0.1 * (1.0 - load)

                if score > best_score:
                    best_score, best_server = score, i

        return best_server if best_server != -1 else N_SERVERS


# =========================================================
# CarbonClipper
# =========================================================
# When carbon is above the adaptive threshold, CarbonClipper does not
# simply refuse to schedule.  Instead it *clips* the eligible server pool
# to only servers that are running below a utilisation ceiling — mimicking
# a power-cap that limits total draw without grinding to a halt.
# When carbon is normal, all servers are eligible and the least-loaded
# one wins as usual.

_CLIP_LOAD_CEILING = 0.60   # servers above this load are excluded during high-carbon


class CarbonClipperScheduler(BaseScheduler):
    name = "CarbonClipper"

    def _reset_state(self):
        self.history = []

    def select_action(self, obs, env):
        if not env._queue:
            return N_SERVERS

        carbon = float(obs[65])
        self.history.append(carbon)

        if len(self.history) > 50:
            self.history.pop(0)

        threshold = np.mean(self.history) + np.std(self.history)
        carbon_high = carbon > threshold

        task = env._queue[0]
        best, best_load = -1, 1e9

        for i, spec in enumerate(FLEET):
            if not env._can_fit(i, task):
                continue

            load = (env._cpu_used[i] / spec.cpu_capacity +
                    env._mem_used[i] / spec.mem_capacity) / 2

            # Power-cap clip: during high-carbon periods exclude heavily
            # loaded servers so overall fleet power draw is reduced.
            if carbon_high and load > _CLIP_LOAD_CEILING:
                continue

            if load < best_load:
                best_load, best = load, i

        # If the clipped pool is empty fall back to the full pool so we
        # never permanently stall — mirrors a "soft clip" policy.
        if best == -1:
            for i, spec in enumerate(FLEET):
                if not env._can_fit(i, task):
                    continue
                load = (env._cpu_used[i] / spec.cpu_capacity +
                        env._mem_used[i] / spec.mem_capacity) / 2
                if load < best_load:
                    best_load, best = load, i

        return best if best != -1 else N_SERVERS


# =========================================================
# TACS
# =========================================================
# True temporal smoothing: use a linear regression over the recent history
# window to *predict* carbon N steps ahead.  A task is deferred only when
# the current carbon is above the rolling mean AND the short-term forecast
# also stays above that mean — i.e. no clean window is imminent.
# This replaces the original hardcoded 0.6 threshold, making the deferral
# decision fully data-driven and adaptive to the observed signal.

_TACS_WINDOW   = 24   # steps of history to keep
_TACS_HORIZON  = 6    # steps ahead to forecast


class TACSScheduler(BaseScheduler):
    name = "TACS"

    def _reset_state(self):
        self.history = []

    def _forecast(self):
        """Linear-extrapolation forecast _TACS_HORIZON steps ahead."""
        if len(self.history) < 3:
            return self.history[-1]
        x = np.arange(len(self.history), dtype=float)
        slope, intercept = np.polyfit(x, self.history, 1)
        return float(np.clip(intercept + slope * (len(self.history) + _TACS_HORIZON),
                             0.0, 1.0))

    def select_action(self, obs, env):
        if not env._queue:
            return N_SERVERS

        carbon = float(obs[65])
        self.history.append(carbon)

        if len(self.history) > _TACS_WINDOW:
            self.history.pop(0)

        rolling_mean  = float(np.mean(self.history))
        predicted     = self._forecast()

        # Defer only when carbon is above average NOW *and* the forecast
        # says it will remain so — a cleaner window is not yet here.
        if carbon > rolling_mean and predicted > rolling_mean:
            return N_SERVERS

        task = env._queue[0]
        best, best_load = -1, 1e9

        for i, spec in enumerate(FLEET):
            if not env._can_fit(i, task):
                continue

            load = (env._cpu_used[i] / spec.cpu_capacity +
                    env._mem_used[i] / spec.mem_capacity) / 2

            if load < best_load:
                best_load, best = load, i

        return best if best != -1 else N_SERVERS


# =========================================================
# EVALUATION
# =========================================================
def aggregate(name, results):
    def m(attr): return mean(getattr(r, attr) for r in results)
    def s(attr): return stdev([getattr(r, attr) for r in results])

    return AggregateResult(
        name, len(results),
        m("co2_g"), s("co2_g"),
        m("energy_wh"), s("energy_wh"),
        m("sla_violation_rate"), s("sla_violation_rate"),
        m("completed"), s("completed"),
        m("total_reward"), s("total_reward")
    )


def compare(schedulers, episodes=10):
    aggs = []

    for s in schedulers:
        print(f"  Running {s.name} × {episodes} episodes …", end=" ")
        res = [s.run(seed=i) for i in range(episodes)]
        agg = aggregate(s.name, res)
        aggs.append(agg)
        print(f"done  (mean reward={agg.total_reward:.3f})")

    return aggs


def print_table(aggs):
    print("\n──────────────────────────────────────────────────────────")
    print("  Metric", end="")

    for a in aggs:
        print(f"{a.name:>15}", end="")
    print()

    print("──────────────────────────────────────────────────────────")

    def row(label, attr, std):
        print(f"  {label:<12}", end="")
        for a in aggs:
            print(f"{getattr(a, attr):.2f}±{getattr(a, std):.2f}".rjust(15), end="")
        print()

    row("CO₂ (g)", "co2_g", "co2_g_std")
    row("Energy", "energy_wh", "energy_wh_std")
    row("SLA %", "sla_violation_rate", "sla_std")
    row("Completed", "completed", "completed_std")
    row("Reward", "total_reward", "reward_std")

    print("──────────────────────────────────────────────────────────")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    schedulers = [
        LACSScheduler(),
        CarbonClipperScheduler(),
        TACSScheduler(),
    ]

    print("\n── Aggregate comparison ─────────────────────────────────────")

    results = compare(schedulers, episodes=10)
    print_table(results)