"""
carbon_profile.py
─────────────────
generate_carbon_profile(days=30)

Returns an (N,) numpy array of hourly carbon intensity values (g CO₂/kWh)
that mimic a real grid with:
  • Low overnight  (~1–5 am)  — low demand, wind-heavy hours
  • Morning ramp   (~6–9 am)  — demand rises, gas peakers spin up
  • Solar dip      (~11–14)   — midday solar suppresses carbon intensity
  • Evening peak   (~17–20)   — solar drops, demand peaks → dirtiest hours
  • Slow overnight decay back to base

A 7-day weekly pattern is also layered in: weekends run ~15 % cleaner
because heavy industrial load is off.

Parameters
----------
days       : int   — number of days to generate  (default 30)
base       : float — median carbon intensity, gCO₂/kWh (default 280)
amplitude  : float — peak-to-trough swing          (default 120)
noise_std  : float — Gaussian noise std dev         (default 18)
seed       : int | None — RNG seed for reproducibility

Returns
-------
np.ndarray, shape (days * 24,), dtype float32
"""

import numpy as np


def generate_carbon_profile(
    days: int = 30,
    base: float = 280.0,
    amplitude: float = 120.0,
    noise_std: float = 18.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate hourly carbon intensity for `days` days.

    Grid behaviour encoded
    ──────────────────────
    Hour  0–5   : overnight valley  — low demand, more wind/nuclear on-margin
    Hour  6–9   : morning ramp      — demand climbs, gas peakers come online
    Hour 10–14  : solar dip         — utility-scale solar suppresses CI
    Hour 15–16  : solar fade        — output tails off, CI starts climbing
    Hour 17–21  : evening peak      — highest CI of the day
    Hour 22–23  : late-night decay  — demand falls, CI drifts back down
    """
    rng = np.random.default_rng(seed)
    n_hours = days * 24
    hours   = np.arange(n_hours)
    h_of_day = hours % 24                 # 0–23
    day_idx  = hours // 24               # 0 … days-1

    # ── 1. Intra-day shape  ───────────────────────────────────────────────────
    # Built from two overlapping components:
    #
    #  (a) A broad "demand" sine that peaks in the evening (~19:00)
    #  (b) A narrower negative "solar" Gaussian centred on noon
    #
    # Both are expressed as offsets in gCO₂/kWh from `base`.

    # (a) Demand component — cosine shifted so peak lands at hour 19
    #     cos has its maximum at 0 → shift phase so max is at 19/24 × 2π
    peak_hour   = 19.0
    demand_wave = amplitude * 0.6 * (
        -np.cos(2 * np.pi * (h_of_day - peak_hour) / 24)
    )

    # (b) Solar dip — Gaussian centred on 12:30, width ~2.5 h, magnitude ~80
    solar_centre = 12.5
    solar_sigma  = 2.5
    solar_dip = -amplitude * 0.65 * np.exp(
        -0.5 * ((h_of_day - solar_centre) / solar_sigma) ** 2
    )

    # (c) Secondary morning shoulder (~8 am) — small positive bump
    morning_centre = 8.0
    morning_sigma  = 1.5
    morning_bump = amplitude * 0.20 * np.exp(
        -0.5 * ((h_of_day - morning_centre) / morning_sigma) ** 2
    )

    intraday = demand_wave + solar_dip + morning_bump

    # ── 2. Weekly pattern ────────────────────────────────────────────────────
    # Weekend (day % 7 in {5, 6}) → industry offline → ~15 % lower CI
    is_weekend = ((day_idx % 7) >= 5).astype(float)
    weekly_offset = -amplitude * 0.15 * is_weekend

    # ── 3. Slow multi-day drift (e.g. high-pressure weather → less wind) ─────
    # A gentle sine over the full period, period ~5–7 days
    drift_period = rng.uniform(5, 8) * 24   # hours
    drift_phase  = rng.uniform(0, 2 * np.pi)
    multi_day_drift = amplitude * 0.18 * np.sin(
        2 * np.pi * hours / drift_period + drift_phase
    )

    # ── 4. Gaussian noise ────────────────────────────────────────────────────
    noise = rng.normal(0.0, noise_std, size=n_hours)

    # ── 5. Assemble & clip ───────────────────────────────────────────────────
    ci = base + intraday + weekly_offset + multi_day_drift + noise
    ci = np.clip(ci, 30.0, 600.0).astype(np.float32)

    return ci


# ── quick visual check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    profile = generate_carbon_profile(days=7, seed=0)

    print(f"Shape  : {profile.shape}")
    print(f"Min    : {profile.min():.1f} gCO₂/kWh")
    print(f"Max    : {profile.max():.1f} gCO₂/kWh")
    print(f"Mean   : {profile.mean():.1f} gCO₂/kWh")
    print(f"Std    : {profile.std():.1f} gCO₂/kWh")

    # ASCII sparkline — one character per hour, first 3 days
    buckets = "▁▂▃▄▅▆▇█"
    lo, hi  = profile.min(), profile.max()
    spark   = "".join(
        buckets[int((v - lo) / (hi - lo) * (len(buckets) - 1))]
        for v in profile[:72]
    )
    print(f"\n3-day sparkline (each char = 1 hour):")
    print(spark)
    print("        night→morn→solar-dip→eve-peak (×3 days)")