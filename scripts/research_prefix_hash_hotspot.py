#!/usr/bin/env python3
"""Quantify the hotspot effect of prefix-hash routing (OpenAI-style).

Compare 3 dispatch algorithms on the same heavy coding workload:
- push: session affinity, commit at arrival
- pull: self-selection from global queue
- prefix_hash: route by prefix content hash (creates hotspots)

Measure per-node load distribution, hit rates, TTFT, and overflow.
"""
import sys, json, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine
from agentsim.core.des.dispatch import PrefixHashDispatcher

OUT = Path(__file__).resolve().parent.parent / "results" / "prefix_hash"
OUT.mkdir(parents=True, exist_ok=True)

results = {}

for peak in [15, 50]:
    for algo in ["push", "pull", "prefix_hash"]:
        cfg = SimConfig.from_json("configs/heavy_coding.json")
        cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
        cfg.service.n_prefill_nodes = 32; cfg.service.n_gpus_per_worker = 8
        cfg.service.l3a_shared = True; cfg.service.dispatch_algorithm = algo
        for p in cfg.profiles:
            p.arrival_rate_peak = peak

        start = time.monotonic()
        engine = SimEngine(cfg)
        m = engine.run()
        elapsed = time.monotonic() - start
        r = m.report()

        hit = r["cache_hit_rate"]
        all_ttft = []
        for v in m.ttft_us.values():
            all_ttft.extend(v)

        # Per-node load distribution
        node_counts = list(m.per_node_prefill_count.values()) if m.per_node_prefill_count else [0]
        load_std = float(np.std(node_counts)) if node_counts else 0
        load_max = max(node_counts) if node_counts else 0
        load_min = min(node_counts) if node_counts else 0
        load_imbalance = load_max / max(1, load_min) if load_min > 0 else float('inf')

        overflow = 0
        overflow_pct = 0
        if isinstance(engine.dispatcher, PrefixHashDispatcher):
            overflow = engine.dispatcher.overflow_count
            overflow_pct = overflow / max(1, engine.dispatcher.total_dispatches) * 100

        key = f"peak{peak}_{algo}"
        results[key] = {
            "algorithm": algo,
            "peak": peak,
            "completed": sum(m.savings_events.values()),
            "hit_rate": round(1.0 - hit.get("miss", 1.0), 4),
            "L1_hit": round(hit.get("L1", 0), 4),
            "ttft_p50_s": round(np.percentile(all_ttft, 50) / 1e6, 2) if all_ttft else 0,
            "ttft_p95_s": round(np.percentile(all_ttft, 95) / 1e6, 2) if all_ttft else 0,
            "load_std": round(load_std, 1),
            "load_imbalance": round(load_imbalance, 1),
            "load_max": load_max,
            "load_min": load_min,
            "overflow": overflow,
            "overflow_pct": round(overflow_pct, 1),
            "elapsed_s": round(elapsed, 1),
        }
        print(f"  {key}: hit={results[key]['hit_rate']:.3f} L1={results[key]['L1_hit']:.3f} "
              f"TTFT_p50={results[key]['ttft_p50_s']}s imbalance={load_imbalance:.1f}x "
              f"overflow={overflow_pct:.0f}% ({elapsed:.1f}s)")

print("\n" + "=" * 100)
print("SUMMARY: Prefix-Hash Hotspot Analysis")
print("=" * 100)
print(f"{'Config':<25} {'Hit%':>5} {'L1%':>5} {'TTFT p50':>9} {'Imbalance':>10} {'Overflow':>9}")
print("-" * 100)
for key in sorted(results.keys()):
    r = results[key]
    print(f"{key:<25} {r['hit_rate']*100:>4.1f}% {r['L1_hit']*100:>4.1f}% "
          f"{r['ttft_p50_s']:>8.2f}s {r['load_imbalance']:>9.1f}x {r['overflow_pct']:>8.1f}%")

with open(OUT / "prefix_hash_hotspot.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT / 'prefix_hash_hotspot.json'}")
