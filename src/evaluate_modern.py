from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import csv

from simulator import DataCenterEnv
from baselines2 import (
    LACSScheduler,
    CarbonClipperScheduler,
    TACSScheduler,
    EpisodeResult
)
from stable_baselines3 import DQN


# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "dqn_carbon_scheduler.zip"
EPISODES = 10
SEED = 42

METHODS = {
    "DQN": None,
    "LACS": LACSScheduler(),
    "CarbonClipper": CarbonClipperScheduler(),
    "TACS": TACSScheduler(),
}

COLORS = {
    "DQN": "#2ECC71",
    "LACS": "#E74C3C",
    "CarbonClipper": "#F39C12",
    "TACS": "#3498DB",
}

# DQN is baseline for % improvement calculation
BASELINE = "DQN"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =========================================================
# RUN DQN
# =========================================================
def run_dqn(model, seed):
    env = DataCenterEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, t, tr, info = env.step(int(action))
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
# RUN ALL
# =========================================================
def run_all():
    results = {}

    print("Loading DQN...")
    model = DQN.load(MODEL_PATH, device="cpu")

    print("Running DQN...")
    results["DQN"] = [run_dqn(model, SEED + i) for i in range(EPISODES)]

    for name, sched in METHODS.items():
        if name == "DQN":
            continue
        print(f"Running {name}...")
        results[name] = [sched.run(seed=SEED + i) for i in range(EPISODES)]

    return results


# =========================================================
# STATS
# =========================================================
def compute_stats(results):
    stats = {}
    for name, res in results.items():
        stats[name] = {
            "co2":       np.mean([r.co2_g for r in res]),
            "energy":    np.mean([r.energy_wh for r in res]),
            "sla":       np.mean([r.sla_violation_rate for r in res]),
            "reward":    np.mean([r.total_reward for r in res]),
            "completed": np.mean([r.completed for r in res]),
        }
    return stats


# =========================================================
# HORIZONTAL BAR CHART — one per metric, saved separately
# =========================================================
def plot_metric(results, stats, key, title, xlabel, filename, lower_better=True):
    attr_map = {
        "co2":       "co2_g",
        "energy":    "energy_wh",
        "sla":       "sla_violation_rate",
        "reward":    "total_reward",
        "completed": "completed",
    }

    methods = list(stats.keys())
    attr    = attr_map[key]

    means = np.array([stats[m][key] for m in methods])
    stds  = np.array([np.std([getattr(r, attr) for r in results[m]]) for m in methods])

    # ranks  (1 = best)
    order = np.argsort(means) if lower_better else np.argsort(means)[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(methods) + 1)

    # sort bars best-to-worst (top to bottom)
    sorted_idx = order if lower_better else order[::-1]
    methods_s  = [methods[i] for i in sorted_idx]
    means_s    = means[sorted_idx]
    stds_s     = stds[sorted_idx]
    ranks_s    = ranks[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 5))

    y     = np.arange(len(methods_s))
    bars  = ax.barh(
        y, means_s,
        xerr=stds_s, capsize=5,
        color=[COLORS[m] for m in methods_s],
        edgecolor="white", linewidth=1.5,
        alpha=0.88, zorder=3,
        error_kw=dict(elinewidth=1.5, ecolor="gray", capthick=1.5),
    )

    # scatter individual episode points
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods_s):
        vals   = [getattr(r, attr) for r in results[m]]
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(
            vals, y[i] + jitter,
            color=COLORS[m], edgecolors="white",
            s=30, zorder=5, alpha=0.75, linewidths=0.6,
        )

    # bold border on rank-1 bar
    bars[0].set_edgecolor("#222")
    bars[0].set_linewidth(2.5)

    # rank label + value at end of each bar
    x_max = (means_s + stds_s).max()
    for i, (v, std, rank) in enumerate(zip(means_s, stds_s, ranks_s)):
        rank_label = f"#{rank}"
        ax.text(
            v + std + x_max * 0.01, i,
            f"{rank_label}  {v:.2f}",
            va="center", fontsize=9,
            fontweight="bold" if rank == 1 else "normal",
            color="#111",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(methods_s, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(0, x_max * 1.28)
    ax.xaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()   # best on top

    # colour legend
    patches = [mpatches.Patch(color=COLORS[m], label=m) for m in methods_s]
    ep_patch = mpatches.Patch(color="gray", alpha=0.6,
                               label=f"Episode dots (n={EPISODES})")
    ax.legend(handles=patches + [ep_patch], fontsize=8,
              loc="lower right", framealpha=0.8)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


# =========================================================
# EVALUATION MATRIX — with ★ best + % improvement vs baseline
# =========================================================
def plot_matrix(stats):
    methods      = list(stats.keys())
    data         = np.array([
        [stats[m]["co2"]    for m in methods],
        [stats[m]["energy"] for m in methods],
        [stats[m]["sla"]    for m in methods],
        [stats[m]["reward"] for m in methods],
    ])
    labels       = ["CO2 (g) ↓", "Energy (Wh) ↓", "SLA Violation ↓", "Reward ↑"]
    lower_better = [True, True, True, False]

    # normalise → higher score always means better
    norm = (data - data.min(axis=1, keepdims=True)) / (
        data.max(axis=1, keepdims=True) - data.min(axis=1, keepdims=True) + 1e-8
    )
    for i, lb in enumerate(lower_better):
        if lb:
            norm[i] = 1 - norm[i]

    cmap = LinearSegmentedColormap.from_list("rg", ["#d73027", "#ffffbf", "#1a9850"])

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)

    baseline_col = methods.index(BASELINE) if BASELINE in methods else None

    for i in range(len(labels)):
        best_j = int(np.argmin(data[i]) if lower_better[i] else np.argmax(data[i]))

        for j in range(len(methods)):
            val     = data[i, j]
            score   = norm[i, j]
            is_best = (j == best_j)

            # thick border on best cell
            if is_best:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=2.8, edgecolor="#111",
                    facecolor="none", zorder=3,
                )
                ax.add_patch(rect)

            text_color = "white" if score < 0.30 else "black"
            star = " ★" if is_best else ""

            # % improvement vs baseline (skip baseline column itself)
            pct_line = ""
            if baseline_col is not None and j != baseline_col:
                base_val = data[i, baseline_col]
                if abs(base_val) > 1e-8:
                    pct = (base_val - val) / abs(base_val) * 100
                    if not lower_better[i]:
                        pct = -pct   # flip sign: positive = better
                    sign = "+" if pct >= 0 else ""
                    pct_line = f"\n{sign}{pct:.1f}% vs {BASELINE}"

            ax.text(
                j, i,
                f"{val:.2f}{star}{pct_line}",
                ha="center", va="center",
                fontsize=8.5 if pct_line else 10,
                fontweight="bold" if is_best else "normal",
                color=text_color, zorder=4,
                linespacing=1.4,
            )

    ax.set_title(
        f"Evaluation Matrix  (★ = best per metric,  % = improvement vs {BASELINE})",
        fontsize=12, fontweight="bold", pad=12,
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.12)
    cbar.set_label("Normalised Score (higher = better)", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Worst", "Mid", "Best"])

    plt.tight_layout()
    plt.savefig("evaluation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved evaluation_matrix.png")


# =========================================================
# CSV
# =========================================================
def save_csv(results):
    with open("evaluation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "co2", "energy", "sla", "reward", "completed"])
        for name, res in results.items():
            for r in res:
                writer.writerow([name, r.co2_g, r.energy_wh,
                                  r.sla_violation_rate, r.total_reward, r.completed])


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    results = run_all()
    stats   = compute_stats(results)

    print("\nGenerating separate metric plots (horizontal bars)...")
    plot_metric(results, stats, "co2",    "CO2 Emissions",      "grams",   "co2.png",    True)
    plot_metric(results, stats, "energy", "Energy Consumption", "Wh",      "energy.png", True)
    plot_metric(results, stats, "sla",    "SLA Violation Rate", "rate",    "sla.png",    True)
    plot_metric(results, stats, "reward", "Total Reward",       "reward",  "reward.png", False)

    print("\nGenerating evaluation matrix...")
    plot_matrix(stats)

    save_csv(results)

    print("\n✅ Generated files:")
    print("   co2.png")
    print("   energy.png")
    print("   sla.png")
    print("   reward.png")
    print("   evaluation_matrix.png")
    print("   evaluation_results.csv")
