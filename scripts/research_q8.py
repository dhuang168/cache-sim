#!/usr/bin/env python3
"""Q8: Does global L3A benefit NPU the same as GPU?

Compare NPU local vs global L3A at 4 workers, then compare the delta
against the GPU result from Phase 0.5/1.
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine

OUT = Path(__file__).resolve().parent.parent / "results" / "phase3-baseline"
OUT.mkdir(parents=True, exist_ok=True)

def run(path, label, n_workers=4, peak=15):
    cfg = SimConfig.from_json(path)
    cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
    cfg.service.n_prefill_nodes = n_workers * cfg.service.n_gpus_per_worker
    cfg.service.dispatch_algorithm = "pull"
    for p in cfg.profiles: p.arrival_rate_peak = peak
    m = SimEngine(cfg).run()
    r = m.report()
    all_ttft = []
    for v in m.ttft_us.values(): all_ttft.extend(v)
    return {
        "label": label,
        "hit_rate": round(1.0 - r["cache_hit_rate"].get("miss", 1.0), 4),
        "ttft_p50_s": round(np.percentile(all_ttft, 50) / 1e6, 2) if all_ttft else 0,
        "qw_mean_s": round(np.mean(m.queue_wait_us) / 1e6, 2) if m.queue_wait_us else 0,
    }

print("NPU local L3A...")
npu_local = run("configs/custom_npu_local_l3a.json", "NPU local")
print(f"  {npu_local}")

print("NPU global L3A...")
npu_global = run("configs/custom_npu_global_l3a.json", "NPU global")
print(f"  {npu_global}")

print("GPU global L3A...")
gpu_global = run("configs/heavy_coding.json", "GPU global")
print(f"  {gpu_global}")

print("GPU local L3A...")
cfg_local = SimConfig.from_json("configs/heavy_coding.json")
cfg_local.service.l3a_shared = False
cfg_local.service.l3a_remote_latency_us = 0
cfg_local.sim_duration_s = 300.0; cfg_local.warmup_s = 30.0; cfg_local.sim_start_time_s = 36000.0
cfg_local.service.n_prefill_nodes = 32; cfg_local.service.n_gpus_per_worker = 8
cfg_local.service.dispatch_algorithm = "pull"
for p in cfg_local.profiles: p.arrival_rate_peak = 15
m = SimEngine(cfg_local).run()
r = m.report()
all_ttft = []
for v in m.ttft_us.values(): all_ttft.extend(v)
gpu_local = {
    "label": "GPU local",
    "hit_rate": round(1.0 - r["cache_hit_rate"].get("miss", 1.0), 4),
    "ttft_p50_s": round(np.percentile(all_ttft, 50) / 1e6, 2) if all_ttft else 0,
    "qw_mean_s": round(np.mean(m.queue_wait_us) / 1e6, 2) if m.queue_wait_us else 0,
}
print(f"  {gpu_local}")

npu_delta = npu_global["hit_rate"] - npu_local["hit_rate"]
gpu_delta = gpu_global["hit_rate"] - gpu_local["hit_rate"]

results = {
    "npu_local": npu_local, "npu_global": npu_global,
    "gpu_local": gpu_local, "gpu_global": gpu_global,
    "npu_hit_delta_pct": round(npu_delta * 100, 1),
    "gpu_hit_delta_pct": round(gpu_delta * 100, 1),
    "answer": f"NPU global-local delta: {npu_delta*100:.1f}pt, GPU delta: {gpu_delta*100:.1f}pt",
}

print(f"\n=== Q8 Answer ===\n  {results['answer']}")

with open(OUT / "q8_npu_vs_gpu_l3a.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT / 'q8_npu_vs_gpu_l3a.json'}")
