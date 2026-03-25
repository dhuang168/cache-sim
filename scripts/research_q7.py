#!/usr/bin/env python3
"""Q7: DDR tier break-even point on custom NPU.

At what context length does DDR (L2) access become worthwhile vs full recompute?
Use SavingsEvent classification — find where HIT_L2 events appear.
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine

OUT = Path(__file__).resolve().parent.parent / "results" / "phase3-baseline"
OUT.mkdir(parents=True, exist_ok=True)

cfg = SimConfig.from_json("configs/custom_npu_global_l3a.json")
cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
cfg.service.n_prefill_nodes = 4; cfg.service.n_gpus_per_worker = 4
cfg.service.dispatch_algorithm = "pull"
for p in cfg.profiles: p.arrival_rate_peak = 15

m = SimEngine(cfg).run()
r = m.report()

savings = dict(m.savings_events)
hit_rate = r["cache_hit_rate"]
tier_sat = r.get("tier_saturation_pct", {})

results = {
    "savings_events": savings,
    "cache_hit_rate": hit_rate,
    "tier_saturation_pct": tier_sat,
    "l2_hit_rate": hit_rate.get("L2", 0),
    "l2_events": savings.get("CACHE_HIT_L2_WORTHWHILE", 0),
    "answer": f"L2 hit rate = {hit_rate.get('L2', 0)*100:.1f}% with {savings.get('CACHE_HIT_L2_WORTHWHILE', 0)} worthwhile L2 hits",
}

print("=== Q7 Answer: DDR Break-Even ===")
print(f"  L2 hit rate: {hit_rate.get('L2', 0)*100:.1f}%")
print(f"  L2 worthwhile events: {savings.get('CACHE_HIT_L2_WORTHWHILE', 0)}")
print(f"  L3A hit rate: {hit_rate.get('L3A', 0)*100:.1f}%")
print(f"  Tier saturation: L1={tier_sat.get('L1',0):.1f}% L2={tier_sat.get('L2',0):.1f}%")

with open(OUT / "q7_ddr_breakeven.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT / 'q7_ddr_breakeven.json'}")
