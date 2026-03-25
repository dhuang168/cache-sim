#!/usr/bin/env python3
"""Q10: Does chunked prefill scheduling reduce chat TTFT p99?

Compare Orca vs Sarathi scheduling on 50/50 coding+chat mixed workload.
Note: schema-level comparison — not actual chunked prefill execution.
Sarathi modeled as smaller prefill slots (fewer max_num_seqs → less interference).
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.core.des.config import SimConfig, ModelConfig
from agentsim.core.des.engine import SimEngine
from agentsim.integration.adapters.vllm import VLLMConfigAdapter

OUT = Path(__file__).resolve().parent.parent / "results" / "phase4-baseline"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = ModelConfig(model_id="llama3-70b", n_layers=80, n_kv_heads=8, head_dim=128, bytes_per_element=2)

def run_mixed(cfg, label):
    cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
    # Single worker, 50/50 coding+chat
    cfg.service.n_prefill_nodes = 8; cfg.service.n_gpus_per_worker = 8
    cfg.profile_mix = {"coding": 0.50, "chat": 0.50}
    cfg.service.dispatch_algorithm = "pull"
    for p in cfg.profiles: p.arrival_rate_peak = 15
    m = SimEngine(cfg).run()
    r = m.report()
    all_ttft = []
    for v in m.ttft_us.values(): all_ttft.extend(v)
    result = {
        "label": label,
        "completed": sum(m.savings_events.values()),
        "ttft_p50_s": round(np.percentile(all_ttft, 50) / 1e6, 2) if all_ttft else 0,
        "ttft_p95_s": round(np.percentile(all_ttft, 95) / 1e6, 2) if all_ttft else 0,
        "ttft_p99_s": round(np.percentile(all_ttft, 99) / 1e6, 2) if all_ttft else 0,
    }
    print(f"  {label}: p50={result['ttft_p50_s']}s p95={result['ttft_p95_s']}s p99={result['ttft_p99_s']}s")
    return result

adapter = VLLMConfigAdapter()

# Orca scheduling (standard — large batch, prefill-prioritizing)
print("Orca scheduling (standard)...")
base1 = SimConfig.from_json("configs/heavy_coding.json")
cfg_orca = adapter.from_vllm_config(
    {"block_size": 16, "max_num_seqs": 256, "enable_prefix_caching": True},
    model=MODEL, base_config=base1,
)
orca = run_mixed(cfg_orca, "Orca")

# Sarathi scheduling (chunked prefill — modeled as smaller prefill slots)
print("Sarathi scheduling (chunked prefill)...")
base2 = SimConfig.from_json("configs/heavy_coding.json")
cfg_sarathi = adapter.from_vllm_config(
    {"block_size": 16, "max_num_seqs": 64, "enable_prefix_caching": True,
     "enable_chunked_prefill": True},
    model=MODEL, base_config=base2,
)
sarathi = run_mixed(cfg_sarathi, "Sarathi")

p99_change = (orca["ttft_p99_s"] - sarathi["ttft_p99_s"]) / max(0.01, orca["ttft_p99_s"]) * 100

results = {
    "orca": orca, "sarathi": sarathi,
    "ttft_p99_change_pct": round(p99_change, 1),
    "answer": f"Sarathi changes chat TTFT p99 by {p99_change:+.1f}% vs Orca",
    "caveat": "Schema-level comparison — Sarathi modeled as fewer max_num_seqs, not actual chunked prefill",
}

print(f"\n=== Q10 Answer ===\n  {results['answer']}")

with open(OUT / "q10_sarathi_vs_orca.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT / 'q10_sarathi_vs_orca.json'}")
