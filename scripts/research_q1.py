#!/usr/bin/env python3
"""Q1: At what worker count does global L3A become essential?

Node scaling: 1/2/4/8 workers, global vs local L3A, peak=100, 5 min.
Find where local hit rate drops below 90%.
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine

OUT = Path(__file__).resolve().parent.parent / "results" / "phase1-baseline"
OUT.mkdir(parents=True, exist_ok=True)

results = {}
for nw in [1, 2, 4, 8]:
    for shared, label in [(True, "global"), (False, "local")]:
        cfg = SimConfig.from_json("configs/heavy_coding.json")
        cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
        cfg.service.n_prefill_nodes = nw * 8; cfg.service.n_gpus_per_worker = 8
        cfg.service.l3a_shared = shared; cfg.service.dispatch_algorithm = "pull"
        m = SimEngine(cfg).run()
        hit = 1.0 - m.report()["cache_hit_rate"].get("miss", 1.0)
        key = f"{nw}w_{label}"
        results[key] = round(hit, 4)
        print(f"  {key}: hit={hit:.3f}")

print("\n=== Q1 Answer ===")
crossover = None
for nw in [1, 2, 4, 8]:
    g, l = results[f"{nw}w_global"], results[f"{nw}w_local"]
    below = l < 0.90
    print(f"  {nw}W: global={g:.3f} local={l:.3f} delta={100*(g-l):.1f}pt {'< 90%' if below else '>= 90%'}")
    if below and crossover is None:
        crossover = nw

results["crossover_workers"] = crossover
results["answer"] = f"Local L3A drops below 90% at {crossover} workers" if crossover else "Local stays >= 90% at all tested scales"

with open(OUT / "q1_node_scaling.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT / 'q1_node_scaling.json'}")
