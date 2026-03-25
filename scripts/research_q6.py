#!/usr/bin/env python3
"""Q6: Custom NPU vs A100 on agentic coding TTFT.

Run agentic_coding on both NPU (global L3A) and A100 configs.
Compare TTFT p50/p95. NPU is ANALYTICAL_ONLY.
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine

OUT = Path(__file__).resolve().parent.parent / "results" / "phase3-baseline"
OUT.mkdir(parents=True, exist_ok=True)

def run_config(path, label, peak=15):
    cfg = SimConfig.from_json(path)
    cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
    cfg.service.dispatch_algorithm = "pull"
    for p in cfg.profiles: p.arrival_rate_peak = peak
    m = SimEngine(cfg).run()
    r = m.report()
    all_ttft = []
    for v in m.ttft_us.values(): all_ttft.extend(v)
    return {
        "label": label,
        "completed": sum(m.savings_events.values()),
        "hit_rate": round(1.0 - r["cache_hit_rate"].get("miss", 1.0), 4),
        "ttft_p50_s": round(np.percentile(all_ttft, 50) / 1e6, 2) if all_ttft else 0,
        "ttft_p95_s": round(np.percentile(all_ttft, 95) / 1e6, 2) if all_ttft else 0,
    }

print("Running A100 (CALIBRATED)...")
a100 = run_config("configs/heavy_coding.json", "A100 (calibrated)")
print(f"  A100: TTFT p50={a100['ttft_p50_s']}s p95={a100['ttft_p95_s']}s")

print("Running NPU global (ANALYTICAL-ONLY)...")
npu = run_config("configs/custom_npu_global_l3a.json", "NPU global (analytical-only)")
print(f"  NPU:  TTFT p50={npu['ttft_p50_s']}s p95={npu['ttft_p95_s']}s")

results = {"a100": a100, "npu_global": npu}
results["answer"] = f"A100 TTFT p50={a100['ttft_p50_s']}s vs NPU p50={npu['ttft_p50_s']}s (NPU is ANALYTICAL-ONLY)"

print(f"\n=== Q6 Answer ===\n  {results['answer']}")

with open(OUT / "q6_npu_vs_a100.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT / 'q6_npu_vs_a100.json'}")
