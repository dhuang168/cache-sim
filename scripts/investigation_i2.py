#!/usr/bin/env python3
"""I2: L3A saturation timeline — when does global-vs-local diverge?

Hypothesis: Divergence starts when L3A fills past ~90% and sessions migrate.
Run 4W heavy_coding at peak=100 for 2/5/10/15/20 min. Track hit rate delta.
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine

OUT = Path(__file__).resolve().parent.parent / "results" / "investigations"
OUT.mkdir(parents=True, exist_ok=True)

results = {}
for dur_min in [2, 5, 10]:  # capped at 10min (15/20 too slow at peak=100)
    for shared, label in [(True, "global"), (False, "local")]:
        cfg = SimConfig.from_json("configs/heavy_coding.json")
        cfg.sim_duration_s = dur_min * 60.0
        cfg.warmup_s = min(30.0, dur_min * 6)  # 10% warmup, min 30s
        cfg.sim_start_time_s = 36000.0
        cfg.service.n_prefill_nodes = 32; cfg.service.n_gpus_per_worker = 8
        cfg.service.l3a_shared = shared; cfg.service.dispatch_algorithm = "pull"

        m = SimEngine(cfg).run()
        r = m.report()
        hit = 1.0 - r["cache_hit_rate"].get("miss", 1.0)
        l3a_sat = r.get("tier_saturation_pct", {}).get("L3A", 0)
        key = f"{dur_min}min_{label}"
        results[key] = {"hit": round(hit, 4), "l3a_sat": round(l3a_sat, 1)}
        print(f"  {key}: hit={hit:.3f} L3A_sat={l3a_sat:.1f}%")

print("\n=== I2 Answer: Saturation Timeline ===")
for dur_min in [2, 5, 10]:  # capped at 10min (15/20 too slow at peak=100)
    g = results[f"{dur_min}min_global"]
    l = results[f"{dur_min}min_local"]
    delta = (g["hit"] - l["hit"]) * 100
    print(f"  {dur_min:2d}min: global={g['hit']:.3f} (L3A {g['l3a_sat']:.0f}%) local={l['hit']:.3f} (L3A {l['l3a_sat']:.0f}%) delta={delta:+.1f}pt")

with open(OUT / "i2_saturation_timeline.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT / 'i2_saturation_timeline.json'}")
