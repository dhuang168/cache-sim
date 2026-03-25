#!/usr/bin/env python3
"""Q9: Does LMCache CPU offload reduce TTFT p95?

Compare vLLM baseline (L1 only, large) vs vLLM + LMCache (L1 + L2 CPU offload).
Heavy agentic_coding, 4 workers.
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.core.des.config import SimConfig, ModelConfig
from agentsim.core.des.engine import SimEngine
from agentsim.integration.adapters.vllm import VLLMConfigAdapter
from agentsim.integration.adapters.lmcache import LMCacheConfigAdapter

OUT = Path(__file__).resolve().parent.parent / "results" / "phase4-baseline"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = ModelConfig(model_id="llama3-70b", n_layers=80, n_kv_heads=8, head_dim=128, bytes_per_element=2)

def run_and_measure(cfg, label):
    cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
    cfg.service.n_prefill_nodes = 32; cfg.service.n_gpus_per_worker = 8
    cfg.service.dispatch_algorithm = "pull"
    for p in cfg.profiles: p.arrival_rate_peak = 15
    m = SimEngine(cfg).run()
    r = m.report()
    all_ttft = []
    for v in m.ttft_us.values(): all_ttft.extend(v)
    result = {
        "label": label,
        "completed": sum(m.savings_events.values()),
        "hit_rate": round(1.0 - r["cache_hit_rate"].get("miss", 1.0), 4),
        "ttft_p50_s": round(np.percentile(all_ttft, 50) / 1e6, 2) if all_ttft else 0,
        "ttft_p95_s": round(np.percentile(all_ttft, 95) / 1e6, 2) if all_ttft else 0,
    }
    print(f"  {label}: hit={result['hit_rate']:.3f} TTFT p50={result['ttft_p50_s']}s p95={result['ttft_p95_s']}s")
    return result

# vLLM baseline (standard config)
print("vLLM baseline (no CPU offload)...")
base = SimConfig.from_json("configs/heavy_coding.json")
adapter_v = VLLMConfigAdapter()
cfg_vllm = adapter_v.from_vllm_config(
    {"block_size": 16, "max_num_seqs": 256, "enable_prefix_caching": True},
    model=MODEL, base_config=base,
)
vllm_result = run_and_measure(cfg_vllm, "vLLM baseline")

# vLLM + LMCache (CPU offload)
print("vLLM + LMCache (CPU offload)...")
base2 = SimConfig.from_json("configs/heavy_coding.json")
adapter_l = LMCacheConfigAdapter()
cfg_lmc = adapter_l.from_lmcache_config(
    {"chunk_size": 256, "local_cpu": True, "max_local_cpu_size": 64.0},
    model=MODEL, base_config=base2,
)
lmc_result = run_and_measure(cfg_lmc, "vLLM + LMCache")

improvement = (vllm_result["ttft_p95_s"] - lmc_result["ttft_p95_s"]) / max(0.01, vllm_result["ttft_p95_s"]) * 100

results = {
    "vllm_baseline": vllm_result,
    "vllm_lmcache": lmc_result,
    "ttft_p95_improvement_pct": round(improvement, 1),
    "answer": f"LMCache CPU offload changes TTFT p95 by {improvement:+.1f}%",
}

print(f"\n=== Q9 Answer ===\n  {results['answer']}")

with open(OUT / "q9_lmcache_offload.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT / 'q9_lmcache_offload.json'}")
