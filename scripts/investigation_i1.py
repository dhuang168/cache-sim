#!/usr/bin/env python3
"""I1: L2 utilization — TTL vs LRU eviction on A100.

Hypothesis: TTL migration pushes objects through L2 too fast to be re-accessed.
LRU eviction (no TTL) should increase L2 hit rate.
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine

OUT = Path(__file__).resolve().parent.parent / "results" / "investigations"
OUT.mkdir(parents=True, exist_ok=True)

results = {}
for policy in ["ttl", "lru"]:
    cfg = SimConfig.from_json("configs/heavy_coding.json")
    cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
    cfg.service.n_prefill_nodes = 8; cfg.service.n_gpus_per_worker = 8
    cfg.service.dispatch_algorithm = "pull"
    cfg.cache.eviction_policy = policy
    m = SimEngine(cfg).run()
    r = m.report()
    hit = r["cache_hit_rate"]
    results[policy] = {
        "L1": hit.get("L1", 0), "L2": hit.get("L2", 0),
        "L3A": hit.get("L3A", 0), "miss": hit.get("miss", 0),
        "tier_sat": r.get("tier_saturation_pct", {}),
        "l1_to_l2_ttl": m.l1_to_l2_ttl_migrations,
        "l1_to_l2_pressure": m.l1_to_l2_evictions,
    }
    print(f"  {policy}: L1={hit['L1']:.3f} L2={hit['L2']:.3f} L3A={hit['L3A']:.3f} miss={hit['miss']:.4f} ttl_mig={m.l1_to_l2_ttl_migrations} pressure={m.l1_to_l2_evictions}")

print(f"\n=== I1 Answer ===")
print(f"  TTL L2 hit: {results['ttl']['L2']*100:.1f}%  LRU L2 hit: {results['lru']['L2']*100:.1f}%")
results["hypothesis_confirmed"] = results["lru"]["L2"] > results["ttl"]["L2"]
results["answer"] = f"LRU L2 hit rate ({results['lru']['L2']*100:.1f}%) vs TTL ({results['ttl']['L2']*100:.1f}%)"

with open(OUT / "i1_l2_ttl_vs_lru.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT / 'i1_l2_ttl_vs_lru.json'}")
