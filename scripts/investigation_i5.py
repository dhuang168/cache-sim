#!/usr/bin/env python3
"""I5: Dispatch affinity — 4 vs 8 GPUs per worker.

Hypothesis: 8 GPUs/worker gives better intra-worker L2 sharing,
improving global L3A affinity. 4 GPUs/worker has less sharing.
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine

OUT = Path(__file__).resolve().parent.parent / "results" / "investigations"
OUT.mkdir(parents=True, exist_ok=True)

results = {}
# Same total GPUs (32), different GPUs per worker
for gpw in [4, 8]:
    n_workers = 32 // gpw
    cfg = SimConfig.from_json("configs/heavy_coding.json")
    cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
    cfg.service.n_prefill_nodes = 32; cfg.service.n_gpus_per_worker = gpw
    cfg.service.l3a_shared = True; cfg.service.dispatch_algorithm = "pull"
    for p in cfg.profiles: p.arrival_rate_peak = 15

    m = SimEngine(cfg).run()
    r = m.report()
    hit = r["cache_hit_rate"]
    all_ttft = []
    for v in m.ttft_us.values(): all_ttft.extend(v)

    key = f"{gpw}gpw_{n_workers}w"
    results[key] = {
        "gpus_per_worker": gpw,
        "n_workers": n_workers,
        "L1": round(hit.get("L1", 0), 4),
        "L2": round(hit.get("L2", 0), 4),
        "L3A": round(hit.get("L3A", 0), 4),
        "miss": round(hit.get("miss", 0), 4),
        "ttft_p50_s": round(np.percentile(all_ttft, 50) / 1e6, 2) if all_ttft else 0,
        "affinity_dispatches": m.affinity_dispatches,
        "non_affinity_dispatches": m.non_affinity_dispatches,
        "tier_sat": r.get("tier_saturation_pct", {}),
    }
    aff_rate = m.affinity_dispatches / max(1, m.affinity_dispatches + m.non_affinity_dispatches)
    print(f"  {key}: L1={hit['L1']:.3f} L2={hit['L2']:.3f} affinity={aff_rate:.1%} TTFT_p50={results[key]['ttft_p50_s']}s")

print(f"\n=== I5 Answer ===")
for k, v in results.items():
    aff = v["affinity_dispatches"] / max(1, v["affinity_dispatches"] + v["non_affinity_dispatches"])
    print(f"  {k}: L1={v['L1']*100:.1f}% L2={v['L2']*100:.1f}% affinity={aff:.1%} TTFT={v['ttft_p50_s']}s")

with open(OUT / "i5_dispatch_affinity.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT / 'i5_dispatch_affinity.json'}")
