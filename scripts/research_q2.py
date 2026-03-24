#!/usr/bin/env python3
"""Q2: Steady-state cache hit rate distribution under heavy agentic coding.

Single worker, heavy_coding, 5 min, peak=100.
Record SavingsEvent breakdown: HIT_L1 / HIT_L2_WIN / HIT_L3_WIN / MISS.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine

OUT = Path(__file__).resolve().parent.parent / "results" / "phase1-baseline"
OUT.mkdir(parents=True, exist_ok=True)

cfg = SimConfig.from_json("configs/heavy_coding.json")
cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
cfg.service.n_prefill_nodes = 8; cfg.service.n_gpus_per_worker = 8
cfg.service.dispatch_algorithm = "pull"

m = SimEngine(cfg).run()
r = m.report()
completed = sum(m.savings_events.values())

results = {
    "completed": completed,
    "savings_events": dict(m.savings_events),
    "cache_hit_rate": r["cache_hit_rate"],
    "savings_class_distribution": r.get("savings_class_distribution", {}),
    "tier_saturation_pct": r.get("tier_saturation_pct", {}),
    "recompute_fraction_mean": round(float(sum(m.recompute_fraction) / max(1, len(m.recompute_fraction))), 4) if m.recompute_fraction else 0,
}

print("=== Q2 Answer: SavingsEvent Breakdown ===")
for cls, count in sorted(m.savings_events.items()):
    pct = count / max(1, completed) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")
print(f"  Total: {completed:,}")

with open(OUT / "q2_savings_breakdown.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT / 'q2_savings_breakdown.json'}")
