"""
Heavy coding workload analysis — all plots at 20-min simulation.
Generates plots for the heavy_coding_report.md.
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
from sim.analysis import find_sustaining_qps

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "heavy_coding.json")
OUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "heavy_coding"
OUT_DIR.mkdir(exist_ok=True, parents=True)

GPUS_PER_WORKER = 8
SIM_DURATION = 1200.0  # 20 min
WARMUP = 10.0
EPOCH_INTERVAL = 30.0


def base_config() -> SimConfig:
    cfg = SimConfig.from_json(CONFIG_PATH)
    cfg.sim_duration_s = SIM_DURATION
    cfg.warmup_s = WARMUP
    cfg.epoch_report_interval_s = EPOCH_INTERVAL
    cfg.sim_start_time_s = 36000.0
    return cfg


def run_sim(config: SimConfig):
    return SimEngine(config).run()


# ─── Plot 1: Global vs Local hit rate over time (4 workers) ───

def plot_global_vs_local_timeline():
    """Hit rate and occupancy comparison at 4 workers over 20 min."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for shared, label, color in [(True, "Global L3A", "#2ecc71"), (False, "Local L3A", "#e67e22")]:
        cfg = base_config()
        cfg.service.n_prefill_nodes = 32
        cfg.service.n_gpus_per_worker = GPUS_PER_WORKER
        cfg.service.l3a_shared = shared
        cfg.service.l3a_remote_latency_us = 50_000 if shared else 0
        m = run_sim(cfg)

        epochs = [(WARMUP + EPOCH_INTERVAL * (i + 1)) / 60 for i in range(len(m.tier_occupancy_pct.get("L1", [])))]

        # L2 occupancy
        axes[0, 0].plot(epochs, m.tier_occupancy_pct.get("L2", []), linewidth=2, color=color, label=label)
        # L3A occupancy
        axes[0, 1].plot(epochs, m.tier_occupancy_pct.get("L3A", []), linewidth=2, color=color, label=label)
        # Queue depth
        axes[1, 0].plot(epochs, m.prefill_queue_depth, linewidth=2, color=color, label=label)
        # Cold evictions per epoch
        axes[1, 1].plot(epochs[:len(m.cold_evictions_per_epoch)], m.cold_evictions_per_epoch, linewidth=2, color=color, label=label)

        total = max(1, sum(m.savings_events.values()))
        miss = m.savings_events.get("COLD_MISS", 0)
        print(f"  {label}: hit={1-miss/total:.2%} miss={miss} cold_evict={m.session_cold_evictions}")

    for ax, title, ylabel in [
        (axes[0, 0], "L2 Occupancy Over Time", "Occupancy (%)"),
        (axes[0, 1], "L3A Occupancy Over Time", "Occupancy (%)"),
        (axes[1, 0], "Prefill Queue Depth", "Queue Depth"),
        (axes[1, 1], "Cold Evictions Per Epoch", "Evictions"),
    ]:
        ax.set_xlabel("Sim Time (min)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Global vs Local L3A — 4 Workers × 8 GPUs, 20 Min", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "global_vs_local_timeline.png", dpi=150)
    plt.close(fig)
    print("  ✓ global_vs_local_timeline.png")


# ─── Plot 2: Single worker deep dive (20 min) ───

def plot_single_worker_deep_dive():
    """Single worker (8 GPUs) at 20 min — tier occupancy, TTFT, queue, hit rates."""
    cfg = base_config()
    cfg.service.n_prefill_nodes = 8
    cfg.service.n_gpus_per_worker = GPUS_PER_WORKER
    m = run_sim(cfg)
    r = m.report()

    epochs = [(WARMUP + EPOCH_INTERVAL * (i + 1)) / 60 for i in range(len(m.tier_occupancy_pct.get("L1", [])))]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Tier occupancy over time
    ax = axes[0, 0]
    for tier in ["L1", "L2", "L3A"]:
        data = m.tier_occupancy_pct.get(tier, [])
        if data:
            ax.plot(epochs[:len(data)], data, linewidth=2, label=tier)
    ax.set_xlabel("Sim Time (min)")
    ax.set_ylabel("Occupancy (%)")
    ax.set_title("Tier Occupancy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # TTFT distribution
    ax = axes[0, 1]
    sources = ["L1_hit", "L2_hit", "L3A_hit", "cold_miss"]
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
    data_lists, labels_used, color_list = [], [], []
    for src, c in zip(sources, colors):
        data = m.ttft_us.get(src, [])
        if data:
            data_lists.append([d / 1000.0 for d in data])
            labels_used.append(src)
            color_list.append(c)
    if data_lists:
        parts = ax.violinplot(data_lists, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(color_list[i])
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels_used) + 1))
        ax.set_xticklabels(labels_used, rotation=20)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    # Queue depth over time
    ax = axes[0, 2]
    ax.plot(epochs[:len(m.prefill_queue_depth)], m.prefill_queue_depth, linewidth=2, color="#e74c3c")
    ax.set_xlabel("Sim Time (min)")
    ax.set_ylabel("Queue Depth")
    ax.set_title("Prefill Queue Depth")
    ax.grid(True, alpha=0.3)

    # Hit rate pie
    ax = axes[1, 0]
    colors_map = {
        "CACHE_HIT_L1": "#2ecc71", "CACHE_HIT_L2_WORTHWHILE": "#3498db",
        "CACHE_HIT_L3A_WORTHWHILE": "#e67e22", "CACHE_HIT_L3A_BREAK_EVEN": "#f39c12",
        "COLD_MISS": "#e74c3c",
    }
    labels, sizes, pie_colors = [], [], []
    for cls, c in colors_map.items():
        count = m.savings_events.get(cls, 0)
        if count > 0:
            labels.append(cls.replace("CACHE_HIT_", "").replace("_", " "))
            sizes.append(count)
            pie_colors.append(c)
    if sizes:
        ax.pie(sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("Cache Hit Breakdown")

    # Recompute fraction
    ax = axes[1, 1]
    if m.recompute_fraction:
        ax.hist(m.recompute_fraction, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Recompute Fraction")
    ax.set_ylabel("Request Count")
    ax.set_title("Recompute Fraction Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    # Slot utilization
    ax = axes[1, 2]
    ax.plot(epochs[:len(m.slot_utilization_pct)], m.slot_utilization_pct, linewidth=2, color="#9b59b6")
    ax.set_xlabel("Sim Time (min)")
    ax.set_ylabel("Utilization (%)")
    ax.set_title("Prefill Slot Utilization")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    total = max(1, sum(m.savings_events.values()))
    miss = m.savings_events.get("COLD_MISS", 0)
    fig.suptitle(f"Single Worker (8 GPUs) — 20 Min (hit={1-miss/total:.1%}, {total} events)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "single_worker_20min.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ single_worker_20min.png (hit={1-miss/total:.1%})")


# ─── Plot 3: 4-worker deep dive (20 min, global vs local) ───

def plot_multi_worker_deep_dive():
    """4 workers — global vs local at 20 min."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for shared, label, color, ls in [
        (True, "Global", "#2ecc71", "-"),
        (False, "Local", "#e67e22", "--"),
    ]:
        cfg = base_config()
        cfg.service.n_prefill_nodes = 32
        cfg.service.n_gpus_per_worker = GPUS_PER_WORKER
        cfg.service.l3a_shared = shared
        cfg.service.l3a_remote_latency_us = 50_000 if shared else 0
        m = run_sim(cfg)

        epochs = [(WARMUP + EPOCH_INTERVAL * (i + 1)) / 60 for i in range(len(m.tier_occupancy_pct.get("L1", [])))]
        total = max(1, sum(m.savings_events.values()))
        miss = m.savings_events.get("COLD_MISS", 0)

        # L1 occupancy
        axes[0, 0].plot(epochs, m.tier_occupancy_pct.get("L1", []), linestyle=ls, linewidth=2, color=color, label=f"{label} ({1-miss/total:.0%} hit)")
        # L2 occupancy
        axes[0, 1].plot(epochs, m.tier_occupancy_pct.get("L2", []), linestyle=ls, linewidth=2, color=color, label=label)
        # L3A occupancy
        axes[0, 2].plot(epochs, m.tier_occupancy_pct.get("L3A", []), linestyle=ls, linewidth=2, color=color, label=label)
        # Queue depth
        axes[1, 0].plot(epochs[:len(m.prefill_queue_depth)], m.prefill_queue_depth, linestyle=ls, linewidth=2, color=color, label=label)
        # Slot utilization
        axes[1, 1].plot(epochs[:len(m.slot_utilization_pct)], m.slot_utilization_pct, linestyle=ls, linewidth=2, color=color, label=label)
        # Cold evictions per epoch
        axes[1, 2].plot(epochs[:len(m.cold_evictions_per_epoch)], m.cold_evictions_per_epoch, linestyle=ls, linewidth=2, color=color, label=label)

        print(f"  {label}: hit={1-miss/total:.2%} miss={miss} cold={m.session_cold_evictions} aff={m.affinity_dispatches} non_aff={m.non_affinity_dispatches}")

    titles = ["L1 Occupancy", "L2 Occupancy", "L3A Occupancy", "Queue Depth", "Slot Utilization (%)", "Cold Evictions/Epoch"]
    ylabels = ["Occupancy (%)", "Occupancy (%)", "Occupancy (%)", "Depth", "Utilization (%)", "Evictions"]
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_xlabel("Sim Time (min)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("4 Workers × 8 GPUs — Global vs Local L3A, 20 Min", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "multi_worker_20min.png", dpi=150)
    plt.close(fig)
    print("  ✓ multi_worker_20min.png")


# ─── Plot 4: Node scaling at 20 min ───

def plot_node_scaling_20min():
    """Worker sweep at 20 min — global vs local."""
    worker_counts = [1, 2, 4]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    for shared, label, color, marker in [
        (True, "Global L3A", "#2ecc71", "o-"),
        (False, "Local L3A", "#e67e22", "s--"),
    ]:
        hits, misses, qw_p95s = [], [], []
        gpu_counts = []
        for nw in worker_counts:
            n_gpus = nw * GPUS_PER_WORKER
            gpu_counts.append(n_gpus)
            cfg = base_config()
            cfg.service.n_prefill_nodes = n_gpus
            cfg.service.n_gpus_per_worker = GPUS_PER_WORKER
            cfg.service.l3a_shared = shared
            cfg.service.l3a_remote_latency_us = 50_000 if shared else 0
            m = run_sim(cfg)
            total = max(1, sum(m.savings_events.values()))
            miss = m.savings_events.get("COLD_MISS", 0)
            hits.append((1 - miss / total) * 100)
            misses.append(miss)
            qw = float(np.percentile(m.queue_wait_us, 95)) / 1000.0 if m.queue_wait_us else 0.0
            qw_p95s.append(qw)
            print(f"  {label} {nw}W: hit={1-miss/total:.1%} miss={miss} qw_p95={qw:.0f}ms")

        ax1.plot(gpu_counts, hits, marker, linewidth=2, color=color, label=label)
        ax2.plot(gpu_counts, misses, marker, linewidth=2, color=color, label=label)
        ax3.plot(gpu_counts, qw_p95s, marker, linewidth=2, color=color, label=label)

    for ax, title, ylabel in [
        (ax1, "Cache Hit Rate", "Hit Rate (%)"),
        (ax2, "Total Cold Misses", "Misses"),
        (ax3, "Queue Wait p95", "ms"),
    ]:
        ax.set_xlabel(f"Total GPUs ({GPUS_PER_WORKER}/worker)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Node Scaling — 20 Min Simulation", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "node_scaling_20min.png", dpi=150)
    plt.close(fig)
    print("  ✓ node_scaling_20min.png")


# ─── Plot 5: TTL sensitivity at 20 min ───

def plot_ttl_sensitivity_20min():
    """TTL sweep at 20 min, 4 workers."""
    ttl_values = [10, 30, 60, 120, 300]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for shared, label, color, marker in [
        (True, "Global L3A", "#2ecc71", "o-"),
        (False, "Local L3A", "#e67e22", "s--"),
    ]:
        hits, qw_p95s = [], []
        for ttl in ttl_values:
            cfg = base_config()
            cfg.service.n_prefill_nodes = 32
            cfg.service.n_gpus_per_worker = GPUS_PER_WORKER
            cfg.service.l3a_shared = shared
            cfg.service.l3a_remote_latency_us = 50_000 if shared else 0
            cfg.ttl_l2_s = ttl
            m = run_sim(cfg)
            total = max(1, sum(m.savings_events.values()))
            miss = m.savings_events.get("COLD_MISS", 0)
            hits.append((1 - miss / total) * 100)
            qw = float(np.percentile(m.queue_wait_us, 95)) / 1000.0 if m.queue_wait_us else 0.0
            qw_p95s.append(qw)
            print(f"  {label} TTL={ttl}s: hit={1-miss/total:.1%} qw_p95={qw:.0f}ms")

        ax1.plot(ttl_values, hits, marker, linewidth=2, color=color, label=label)
        ax2.plot(ttl_values, qw_p95s, marker, linewidth=2, color=color, label=label)

    ax1.set_xlabel("L2 TTL (seconds)")
    ax1.set_ylabel("Hit Rate (%)")
    ax1.set_title("Hit Rate vs L2 TTL")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("L2 TTL (seconds)")
    ax2.set_ylabel("Queue Wait p95 (ms)")
    ax2.set_title("Queue Wait vs L2 TTL")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("TTL Sensitivity — 4 Workers, 20 Min", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "ttl_sensitivity_20min.png", dpi=150)
    plt.close(fig)
    print("  ✓ ttl_sensitivity_20min.png")


def main():
    print("=== Heavy Coding Analysis (20 min sims) ===\n")

    print("1. Global vs Local timeline (4 workers)...")
    plot_global_vs_local_timeline()
    print()

    print("2. Single worker deep dive...")
    plot_single_worker_deep_dive()
    print()

    print("3. Multi-worker deep dive (4 workers)...")
    plot_multi_worker_deep_dive()
    print()

    print("4. Node scaling sweep...")
    plot_node_scaling_20min()
    print()

    print("5. TTL sensitivity sweep...")
    plot_ttl_sensitivity_20min()
    print()

    print(f"All plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
