"""
Sanity-check plots for cache_sim.

Generates:
  1. Tier occupancy over time
  2. TTFT distribution by cache hit tier
  3. Cache hit rate breakdown (pie)
  4. L2 capacity sensitivity (TTFT p95 vs L2 size)
  5. Savings class distribution
  6. Queue depth over time
"""
from __future__ import annotations
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.config import SimConfig
from sim.engine import SimEngine

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "default.json")
OUT_DIR = Path(__file__).resolve().parent.parent / "plots"
OUT_DIR.mkdir(exist_ok=True)


def run_sim(config: SimConfig):
    engine = SimEngine(config)
    return engine.run()


def short_config() -> SimConfig:
    """60s sim, 5s warmup — fast enough for plots."""
    cfg = SimConfig.from_json(CONFIG_PATH)
    cfg.sim_duration_s = 60.0
    cfg.warmup_s = 5.0
    cfg.epoch_report_interval_s = 5.0
    return cfg


# ── Plot 1: Tier occupancy over time ──

def plot_tier_occupancy(metrics, cfg):
    fig, ax = plt.subplots(figsize=(10, 5))
    epoch_s = cfg.epoch_report_interval_s
    for tier in ["L1", "L2", "L3A"]:
        data = metrics.tier_occupancy_pct.get(tier, [])
        if data:
            t = [epoch_s * (i + 1) for i in range(len(data))]
            ax.plot(t, data, label=tier, linewidth=2)
    ax.set_xlabel("Sim time (s)")
    ax.set_ylabel("Occupancy (%)")
    ax.set_title("Tier Occupancy Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tier_occupancy.png", dpi=150)
    plt.close(fig)
    print("  ✓ tier_occupancy.png")


# ── Plot 2: TTFT distribution by hit tier ──

def plot_ttft_distribution(metrics):
    fig, ax = plt.subplots(figsize=(10, 5))
    sources = ["L1_hit", "L2_hit", "L3A_hit", "cold_miss"]
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
    labels_used = []
    data_lists = []
    color_list = []
    for src, color in zip(sources, colors):
        data = metrics.ttft_us.get(src, [])
        if data:
            data_ms = [d / 1000.0 for d in data]
            data_lists.append(data_ms)
            labels_used.append(src)
            color_list.append(color)

    if data_lists:
        parts = ax.violinplot(data_lists, showmedians=True, showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(color_list[i])
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels_used) + 1))
        ax.set_xticklabels(labels_used)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT Distribution by Cache Hit Source")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ttft_distribution.png", dpi=150)
    plt.close(fig)
    print("  ✓ ttft_distribution.png")


# ── Plot 3: Cache hit rate pie chart ──

def plot_hit_rate_pie(metrics):
    fig, ax = plt.subplots(figsize=(7, 7))
    labels = []
    sizes = []
    colors_map = {
        "CACHE_HIT_L1": "#2ecc71",
        "CACHE_HIT_L2_WORTHWHILE": "#3498db",
        "CACHE_HIT_L3A_WORTHWHILE": "#e67e22",
        "CACHE_HIT_L3A_BREAK_EVEN": "#f39c12",
        "COLD_MISS": "#e74c3c",
    }
    colors = []
    for cls in colors_map:
        count = metrics.savings_events.get(cls, 0)
        if count > 0:
            labels.append(cls.replace("CACHE_HIT_", "").replace("_", " "))
            sizes.append(count)
            colors.append(colors_map[cls])

    if sizes:
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 10})
    ax.set_title("Savings Class Distribution")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hit_rate_pie.png", dpi=150)
    plt.close(fig)
    print("  ✓ hit_rate_pie.png")


# ── Plot 4: L2 capacity sensitivity ──

def plot_l2_sensitivity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    l2_sizes_gb = [2, 5, 10, 20, 50]
    p50s, p95s, l2_hit_pcts = [], [], []

    for gb in l2_sizes_gb:
        cfg = stressed_config()
        cfg.tiers[1].capacity_bytes = int(gb * 1024**3)
        cfg.seed = 42
        m = run_sim(cfg)
        # Use L2_hit TTFT (main hit source in stressed config)
        pcts = m.ttft_percentiles("L2_hit", [50, 95])
        p50s.append(pcts[50] / 1000.0)
        p95s.append(pcts[95] / 1000.0)
        total = max(1, sum(m.savings_events.values()))
        l2_pct = m.savings_events.get("CACHE_HIT_L2_WORTHWHILE", 0) / total * 100
        l2_hit_pcts.append(l2_pct)
        print(f"    L2={gb}GB done")

    ax1.plot(l2_sizes_gb, p50s, "o-", label="p50", linewidth=2)
    ax1.plot(l2_sizes_gb, p95s, "s-", label="p95", linewidth=2)
    ax1.set_xlabel("L2 Capacity (GB)")
    ax1.set_ylabel("L2 Hit TTFT (ms)")
    ax1.set_title("TTFT vs L2 Capacity (stressed)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(l2_sizes_gb, l2_hit_pcts, "o-", color="#3498db", linewidth=2)
    ax2.set_xlabel("L2 Capacity (GB)")
    ax2.set_ylabel("L2 Hit Rate (%)")
    ax2.set_title("L2 Hit Rate vs L2 Capacity")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "l2_sensitivity.png", dpi=150)
    plt.close(fig)
    print("  ✓ l2_sensitivity.png")


# ── Plot 5: Queue depth over time ──

def plot_queue_depth(metrics, cfg):
    fig, ax = plt.subplots(figsize=(10, 5))
    epoch_s = cfg.epoch_report_interval_s
    if metrics.prefill_queue_depth:
        t = [epoch_s * (i + 1) for i in range(len(metrics.prefill_queue_depth))]
        ax.plot(t, metrics.prefill_queue_depth, label="Prefill queue", linewidth=2)
    if metrics.decode_queue_depth:
        t = [epoch_s * (i + 1) for i in range(len(metrics.decode_queue_depth))]
        ax.plot(t, metrics.decode_queue_depth, label="Decode queue", linewidth=2)
    ax.set_xlabel("Sim time (s)")
    ax.set_ylabel("Queue depth")
    ax.set_title("GPU Queue Depth Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "queue_depth.png", dpi=150)
    plt.close(fig)
    print("  ✓ queue_depth.png")


# ── Plot 6: Recompute fraction histogram ──

def plot_recompute_fraction(metrics):
    fig, ax = plt.subplots(figsize=(10, 5))
    if metrics.recompute_fraction:
        ax.hist(metrics.recompute_fraction, bins=30, color="#3498db",
                edgecolor="white", alpha=0.8)
    ax.set_xlabel("Recompute Fraction (0=full hit, 1=full miss)")
    ax.set_ylabel("Request Count")
    ax.set_title("Distribution of Recompute Fraction")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recompute_fraction.png", dpi=150)
    plt.close(fig)
    print("  ✓ recompute_fraction.png")


def stressed_config() -> SimConfig:
    """
    Constrained scenario: L1 too small to hold most KV objects,
    forcing placement directly into L2. Short TTLs drive L3A hibernation.
    """
    cfg = short_config()
    cfg.run_id = "stressed"
    cfg.tiers[0].capacity_bytes = 500 * 1024**2        # 500 MB L1 — can't hold a 2048-token KV (~670MB)
    cfg.tiers[1].capacity_bytes = 10 * 1024**3          # 10 GB L2
    cfg.tiers[2].capacity_bytes = 50 * 1024**3          # 50 GB L3A
    cfg.ttl_l2_s = 20.0
    cfg.ttl_l3a_s = 120.0
    cfg.eviction_hbm_threshold = 0.6
    cfg.eviction_ram_threshold = 0.8
    return cfg


# ── Plot 7: L1 size sensitivity (hit rate breakdown) ──

def plot_l1_sensitivity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    l1_sizes_gb = [0.25, 0.5, 1, 2, 4, 10, 40, 80]
    l1_hits, l2_hits, l3a_hits, misses = [], [], [], []
    pressure_eviction_rates = []
    ttl_migration_rates = []

    for gb in l1_sizes_gb:
        cfg = stressed_config()
        cfg.tiers[0].capacity_bytes = int(gb * 1024**3)
        cfg.seed = 42
        m = run_sim(cfg)
        total = max(1, sum(m.savings_events.values()))
        l1_hits.append(m.savings_events.get("CACHE_HIT_L1", 0) / total * 100)
        l2_hits.append(m.savings_events.get("CACHE_HIT_L2_WORTHWHILE", 0) / total * 100)
        l3a_w = m.savings_events.get("CACHE_HIT_L3A_WORTHWHILE", 0)
        l3a_b = m.savings_events.get("CACHE_HIT_L3A_BREAK_EVEN", 0)
        l3a_hits.append((l3a_w + l3a_b) / total * 100)
        misses.append(m.savings_events.get("COLD_MISS", 0) / total * 100)
        eff_s = max(1, m.effective_sim_us) / 1e6
        pressure_eviction_rates.append(m.l1_to_l2_evictions / eff_s)
        ttl_migration_rates.append(m.l1_to_l2_ttl_migrations / eff_s)
        print(f"    L1={gb}GB done")

    # Stacked bar for hit rates
    x = np.arange(len(l1_sizes_gb))
    w = 0.6
    ax1.bar(x, l1_hits, w, label="L1 hit", color="#2ecc71")
    ax1.bar(x, l2_hits, w, bottom=l1_hits, label="L2 hit", color="#3498db")
    bottoms = [a + b for a, b in zip(l1_hits, l2_hits)]
    ax1.bar(x, l3a_hits, w, bottom=bottoms, label="L3A hit", color="#e67e22")
    bottoms2 = [a + b for a, b in zip(bottoms, l3a_hits)]
    ax1.bar(x, misses, w, bottom=bottoms2, label="Miss", color="#e74c3c")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{g}" for g in l1_sizes_gb])
    ax1.set_xlabel("L1 Capacity (GB)")
    ax1.set_ylabel("Fraction (%)")
    ax1.set_title("Cache Hit Breakdown vs L1 Size")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.plot(l1_sizes_gb, pressure_eviction_rates, "o-", color="#e74c3c", linewidth=2,
             label="Pressure evictions")
    ax2.plot(l1_sizes_gb, ttl_migration_rates, "s--", color="#3498db", linewidth=2,
             label="TTL migrations")
    ax2.set_xlabel("L1 Capacity (GB)")
    ax2.set_ylabel("L1->L2 Events/s")
    ax2.set_title("L1->L2 Movement Rate vs L1 Size")
    ax2.set_xscale("log", base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "l1_sensitivity.png", dpi=150)
    plt.close(fig)
    print("  ✓ l1_sensitivity.png")


def main():
    # ── Stressed scenario (small L1 forces multi-tier activity) ──
    print("Running stressed simulation (2 GB L1, short TTLs)...")
    cfg = stressed_config()
    metrics = run_sim(cfg)

    report = metrics.report()
    print(f"\nStressed report summary:")
    print(f"  Sharing factor:  {report['sharing_factor']}")
    print(f"  Hit rates:       {report['cache_hit_rate']}")
    print(f"  Tier saturation: {report['tier_saturation_pct']}")
    print(f"  L1->L2 pressure:   {report['eviction_rate_per_s']['L1_to_L2_pressure']}/s")
    print(f"  L1->L2 TTL:        {report['eviction_rate_per_s']['L1_to_L2_ttl']}/s")
    print(f"  Evictions L2->L3A: {report['eviction_rate_per_s']['L2_to_L3A']}/s")
    print(f"  Cold evictions:    {report['session_cold_evictions']}")
    print()

    print("Generating plots from stressed scenario...")
    plot_tier_occupancy(metrics, cfg)
    plot_ttft_distribution(metrics)
    plot_hit_rate_pie(metrics)
    plot_queue_depth(metrics, cfg)
    plot_recompute_fraction(metrics)

    print("\nRunning L1 capacity sweep (8 points)...")
    plot_l1_sensitivity()

    print("\nRunning L2 capacity sweep (5 points)...")
    plot_l2_sensitivity()

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
