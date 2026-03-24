#!/usr/bin/env python3
"""I7: Disaggregated P/D + stressed cache interaction.

Hypothesis: Under stressed L1 (500MB), disagg's 14% TTFT improvement
compounds because more requests hit long-tail prefill times.
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine

OUT = Path(__file__).resolve().parent.parent / "results" / "investigations"
OUT.mkdir(parents=True, exist_ok=True)

results = {}
for label, stressed in [("default_l1", False), ("stressed_l1", True)]:
    for mode in ["legacy", "disagg"]:
        cfg = SimConfig.from_json("configs/heavy_coding.json")
        cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
        cfg.service.n_gpus_per_worker = 8; cfg.service.dispatch_algorithm = "pull"
        for p in cfg.profiles: p.arrival_rate_peak = 15

        if stressed:
            cfg.tiers[0].capacity_bytes = 500 * 1024**2  # 500 MB L1
            cfg.tiers[1].capacity_bytes = 2 * 1024**3
            cfg.ttl_l2_s = 20.0

        if mode == "disagg":
            cfg.service.n_prefill_nodes = 24
            cfg.service.disaggregated = True
            cfg.service.n_decode_nodes = 8
            cfg.service.kv_transfer_bandwidth_bytes_per_s = 50_000_000_000
            cfg.service.kv_transfer_latency_floor_us = 2000
            cfg.service.prefill_latency_multiplier = 0.85
        else:
            cfg.service.n_prefill_nodes = 32
            cfg.service.disaggregated = False

        m = SimEngine(cfg).run()
        r = m.report()
        hit = r["cache_hit_rate"]
        all_ttft = []
        for v in m.ttft_us.values(): all_ttft.extend(v)

        key = f"{label}_{mode}"
        results[key] = {
            "hit_rate": round(1.0 - hit.get("miss", 1.0), 4),
            "miss_pct": round(hit.get("miss", 0) * 100, 1),
            "ttft_p50_s": round(np.percentile(all_ttft, 50) / 1e6, 2) if all_ttft else 0,
            "ttft_p95_s": round(np.percentile(all_ttft, 95) / 1e6, 2) if all_ttft else 0,
            "prefill_mean_s": round(np.mean(m.prefill_duration_us) / 1e6, 2) if m.prefill_duration_us else 0,
        }
        print(f"  {key}: hit={results[key]['hit_rate']:.3f} miss={results[key]['miss_pct']}% TTFT_p50={results[key]['ttft_p50_s']}s")

print(f"\n=== I7 Answer ===")
for stress_label in ["default_l1", "stressed_l1"]:
    leg = results[f"{stress_label}_legacy"]["ttft_p50_s"]
    dis = results[f"{stress_label}_disagg"]["ttft_p50_s"]
    imp = (leg - dis) / max(0.01, leg) * 100
    print(f"  {stress_label}: legacy={leg}s disagg={dis}s improvement={imp:+.1f}%")

results["answer"] = "Disagg improvement under stressed vs default L1"

with open(OUT / "i7_disagg_stressed.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT / 'i7_disagg_stressed.json'}")
