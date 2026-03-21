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
from sim.analysis import find_sustaining_qps

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


# ── Plot 8: Node-scaling — Global vs Local L3A ──

def _collect_node_scaling_data(node_counts, l3a_shared, label):
    """Run sims for each node count and collect metrics."""
    results = []
    for n in node_counts:
        cfg = stressed_config()
        cfg.service.n_prefill_nodes = n
        cfg.service.dispatch_algorithm = "push"
        cfg.service.l3a_shared = l3a_shared
        cfg.service.l3a_remote_latency_us = 50_000 if l3a_shared else 0
        cfg.seed = 42
        m = run_sim(cfg)

        total_events = max(1, sum(m.savings_events.values()))
        eff_s = max(1, m.effective_sim_us) / 1e6

        # Queue utilization %
        q = m.prefill_queue_depth
        total_queue_max = n * cfg.service.prefill_queue_max
        queue_util = (np.mean(q) / total_queue_max * 100) if q and total_queue_max > 0 else 0.0

        # Hit rate breakdown
        l1_hit = m.savings_events.get("CACHE_HIT_L1", 0) / total_events
        l2_hit = m.savings_events.get("CACHE_HIT_L2_WORTHWHILE", 0) / total_events
        l3a_hit = (m.savings_events.get("CACHE_HIT_L3A_WORTHWHILE", 0) +
                   m.savings_events.get("CACHE_HIT_L3A_BREAK_EVEN", 0)) / total_events
        miss_rate = m.savings_events.get("COLD_MISS", 0) / total_events

        # Queue wait p95
        qw_p95 = float(np.percentile(m.queue_wait_us, 95)) / 1000.0 if m.queue_wait_us else 0.0

        # Eviction rate
        evict_rate = (m.l1_to_l2_evictions + m.l2_to_l3a_evictions) / eff_s

        # TTFT p95 decomposed: queue_wait + compute
        all_ttft = []
        for src in ["L1_hit", "L2_hit", "L3A_hit", "cold_miss"]:
            all_ttft.extend(m.ttft_us.get(src, []))
        ttft_p95 = float(np.percentile(all_ttft, 95)) / 1000.0 if all_ttft else 0.0

        results.append({
            "n": n,
            "queue_util_pct": queue_util,
            "hit_rate": 1.0 - miss_rate,
            "l1_hit": l1_hit, "l2_hit": l2_hit, "l3a_hit": l3a_hit, "miss": miss_rate,
            "qw_p95_ms": qw_p95,
            "ttft_p95_ms": ttft_p95,
            "evict_rate": evict_rate,
        })
        print(f"    {label} N={n}: hit={1-miss_rate:.1%} qw_p95={qw_p95:.0f}ms ttft_p95={ttft_p95:.0f}ms")
    return results


def plot_node_scaling():
    node_counts = [1, 2, 4, 8]

    print("  Collecting global L3A data...")
    global_data = _collect_node_scaling_data(node_counts, l3a_shared=True, label="Global")
    print("  Collecting local L3A data...")
    local_data = _collect_node_scaling_data(node_counts, l3a_shared=False, label="Local")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: Cache hit rate breakdown (global vs local)
    ax = axes[0, 0]
    g_hit = [d["hit_rate"] * 100 for d in global_data]
    l_hit = [d["hit_rate"] * 100 for d in local_data]
    ax.plot(node_counts, g_hit, "o-", linewidth=2, color="#2ecc71", label="Global L3A")
    ax.plot(node_counts, l_hit, "s--", linewidth=2, color="#e67e22", label="Local L3A")
    ax.set_xlabel("Prefill Nodes")
    ax.set_ylabel("Cache Hit Rate (%)")
    ax.set_title("(b) Cache Hit Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Per-node eviction rate
    ax = axes[0, 1]
    g_evict = [d["evict_rate"] / d["n"] for d in global_data]
    l_evict = [d["evict_rate"] / d["n"] for d in local_data]
    ax.plot(node_counts, g_evict, "o-", linewidth=2, color="#2ecc71", label="Global L3A")
    ax.plot(node_counts, l_evict, "s--", linewidth=2, color="#e67e22", label="Local L3A")
    ax.set_xlabel("Prefill Nodes")
    ax.set_ylabel("Evictions/s per Node")
    ax.set_title("(a) Per-Node Eviction Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Queue wait p95
    ax = axes[0, 2]
    g_qw = [d["qw_p95_ms"] for d in global_data]
    l_qw = [d["qw_p95_ms"] for d in local_data]
    ax.plot(node_counts, g_qw, "o-", linewidth=2, color="#2ecc71", label="Global L3A")
    ax.plot(node_counts, l_qw, "s--", linewidth=2, color="#e67e22", label="Local L3A")
    ax.set_xlabel("Prefill Nodes")
    ax.set_ylabel("Queue Wait p95 (ms)")
    ax.set_title("(d) Queue Wait p95")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Queue utilization %
    ax = axes[1, 0]
    g_qu = [d["queue_util_pct"] for d in global_data]
    l_qu = [d["queue_util_pct"] for d in local_data]
    ax.plot(node_counts, g_qu, "o-", linewidth=2, color="#2ecc71", label="Global L3A")
    ax.plot(node_counts, l_qu, "s--", linewidth=2, color="#e67e22", label="Local L3A")
    ax.set_xlabel("Prefill Nodes")
    ax.set_ylabel("Queue Utilization (%)")
    ax.set_title("Queue Utilization vs Node Count")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 5: TTFT p95 (total = queue wait + compute)
    ax = axes[1, 1]
    g_ttft = [d["ttft_p95_ms"] for d in global_data]
    l_ttft = [d["ttft_p95_ms"] for d in local_data]
    ax.plot(node_counts, g_ttft, "o-", linewidth=2, color="#2ecc71", label="Global L3A")
    ax.plot(node_counts, l_ttft, "s--", linewidth=2, color="#e67e22", label="Local L3A")
    ax.set_xlabel("Prefill Nodes")
    ax.set_ylabel("TTFT p95 (ms)")
    ax.set_title("(d) TTFT p95 vs Node Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 6: L3A latency sensitivity (fixed 4 nodes)
    ax = axes[1, 2]
    latencies_ms = [0, 10, 25, 50, 75, 100]
    ttft_at_lat = []
    hit_at_lat = []
    for lat_ms in latencies_ms:
        cfg = stressed_config()
        cfg.service.n_prefill_nodes = 4
        cfg.service.l3a_shared = True
        cfg.service.l3a_remote_latency_us = lat_ms * 1000
        cfg.seed = 42
        m = run_sim(cfg)
        all_ttft = []
        for src in ["L1_hit", "L2_hit", "L3A_hit", "cold_miss"]:
            all_ttft.extend(m.ttft_us.get(src, []))
        p95 = float(np.percentile(all_ttft, 95)) / 1000.0 if all_ttft else 0.0
        ttft_at_lat.append(p95)
        total_ev = max(1, sum(m.savings_events.values()))
        miss = m.savings_events.get("COLD_MISS", 0) / total_ev
        hit_at_lat.append((1 - miss) * 100)
        print(f"    L3A latency={lat_ms}ms: ttft_p95={p95:.0f}ms hit={1-miss:.1%}")

    ax.plot(latencies_ms, ttft_at_lat, "o-", linewidth=2, color="#3498db", label="TTFT p95")
    ax.set_xlabel("Global L3A Remote Latency (ms)")
    ax.set_ylabel("TTFT p95 (ms)")
    ax.set_title("(f) L3A Latency Sensitivity (4 nodes)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Multi-Node Scaling: Global vs Local L3A", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "node_scaling.png", dpi=150)
    plt.close(fig)
    print("  ✓ node_scaling.png")


# ── Plot 9: TTL sensitivity — Global vs Local L3A ──

def plot_ttl_sensitivity():
    ttl_l2_values = [1, 5, 10, 20, 60, 120]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for l3a_shared, label, color, marker in [
        (True, "Global L3A", "#2ecc71", "o-"),
        (False, "Local L3A", "#e67e22", "s--"),
    ]:
        hit_rates = []
        qw_p95s = []
        for ttl in ttl_l2_values:
            cfg = stressed_config()
            cfg.service.n_prefill_nodes = 4
            cfg.service.l3a_shared = l3a_shared
            cfg.service.l3a_remote_latency_us = 50_000 if l3a_shared else 0
            cfg.ttl_l2_s = ttl
            cfg.seed = 42
            m = run_sim(cfg)
            total = max(1, sum(m.savings_events.values()))
            miss = m.savings_events.get("COLD_MISS", 0) / total
            hit_rates.append((1 - miss) * 100)
            qw = float(np.percentile(m.queue_wait_us, 95)) / 1000.0 if m.queue_wait_us else 0.0
            qw_p95s.append(qw)
            print(f"    {label} TTL_L2={ttl}s: hit={1-miss:.1%} qw_p95={qw:.0f}ms")

        ax1.plot(ttl_l2_values, hit_rates, marker, linewidth=2, color=color, label=label)
        ax2.plot(ttl_l2_values, qw_p95s, marker, linewidth=2, color=color, label=label)

    ax1.set_xlabel("L2 TTL (seconds)")
    ax1.set_ylabel("Cache Hit Rate (%)")
    ax1.set_title("(c) Hit Rate vs L2 TTL")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("L2 TTL (seconds)")
    ax2.set_ylabel("Queue Wait p95 (ms)")
    ax2.set_title("(c) Queue Wait vs L2 TTL")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("TTL Sensitivity: Global vs Local L3A (4 nodes)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "ttl_sensitivity.png", dpi=150)
    plt.close(fig)
    print("  ✓ ttl_sensitivity.png")


# ── Plot 10: Sustaining QPS at SLA ──

def plot_sustaining_qps():
    node_counts = [1, 2, 4, 8]
    sla_ms = 500.0  # p95 queue wait < 500ms

    fig, ax = plt.subplots(figsize=(10, 6))

    for l3a_shared, label, color, marker in [
        (True, "Global L3A", "#2ecc71", "o-"),
        (False, "Local L3A", "#e67e22", "s--"),
    ]:
        qps_values = []
        for n in node_counts:
            cfg = stressed_config()
            cfg.service.n_prefill_nodes = n
            cfg.service.l3a_shared = l3a_shared
            cfg.service.l3a_remote_latency_us = 50_000 if l3a_shared else 0
            cfg.seed = 42

            mult, report = find_sustaining_qps(
                cfg, sla_queue_wait_p95_ms=sla_ms,
                rate_range=(0.1, 5.0), max_iterations=8,
            )
            # Compute approximate QPS from multiplier and base rates
            base_qps = sum(
                p.arrival_rate_peak * cfg.profile_mix.get(p.name, 0)
                for p in cfg.profiles
            )
            sustaining = mult * base_qps
            qps_values.append(sustaining)
            actual_qw = report.get("_queue_wait_p95_ms", 0)
            print(f"    {label} N={n}: sustaining={sustaining:.0f} qps (mult={mult:.2f}, qw_p95={actual_qw:.0f}ms)")

        ax.plot(node_counts, qps_values, marker, linewidth=2, color=color, label=label)

    ax.set_xlabel("Prefill Nodes")
    ax.set_ylabel(f"Sustaining QPS (p95 queue wait < {sla_ms:.0f}ms)")
    ax.set_title("(e) Sustaining QPS at SLA vs Node Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add ideal linear scaling line
    if qps_values:
        ax.plot(node_counts, [node_counts[0] * qps_values[0] / node_counts[0] * n for n in node_counts],
                "k:", alpha=0.3, label="Ideal linear")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sustaining_qps.png", dpi=150)
    plt.close(fig)
    print("  ✓ sustaining_qps.png")


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

    print("\nRunning node-scaling sweep (4 points)...")
    plot_node_scaling()

    print("\nRunning TTL sensitivity sweep...")
    plot_ttl_sensitivity()

    print("\nRunning sustaining QPS sweep...")
    plot_sustaining_qps()

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
