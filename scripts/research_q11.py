#!/usr/bin/env python3
"""Q11: SGLang RadixAttention vs vLLM LRU on cache hit rate.

Compare SGLang (prefix-aware tail_first eviction) vs vLLM (standard LRU)
on agentic_coding profile. Expected: SGLang preserves shared prefixes longer.
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.core.des.config import SimConfig, ModelConfig
from agentsim.core.des.engine import SimEngine
from agentsim.integration.adapters.vllm import VLLMConfigAdapter
from agentsim.integration.adapters.sglang import SGLangConfigAdapter

OUT = Path(__file__).resolve().parent.parent / "results" / "phase4-baseline"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = ModelConfig(model_id="llama3-70b", n_layers=80, n_kv_heads=8, head_dim=128, bytes_per_element=2)

def run_and_measure(cfg, label):
    cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
    cfg.service.n_prefill_nodes = 8; cfg.service.n_gpus_per_worker = 8
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
        "cache_hit_rate": r["cache_hit_rate"],
        "ttft_p50_s": round(np.percentile(all_ttft, 50) / 1e6, 2) if all_ttft else 0,
        "dedup_ratio": r.get("chunk_dedup", {}).get("dedup_ratio"),
        "recompute_mean": round(np.mean(m.recompute_fraction), 4) if m.recompute_fraction else 0,
    }
    print(f"  {label}: hit={result['hit_rate']:.3f} recompute={result['recompute_mean']:.3f} dedup={result.get('dedup_ratio', 'N/A')}")
    return result

# vLLM with standard LRU
print("vLLM (standard LRU)...")
base1 = SimConfig.from_json("configs/heavy_coding.json")
adapter_v = VLLMConfigAdapter()
cfg_vllm = adapter_v.from_vllm_config(
    {"block_size": 16, "max_num_seqs": 256, "enable_prefix_caching": True},
    model=MODEL, base_config=base1,
)
vllm_result = run_and_measure(cfg_vllm, "vLLM LRU")

# SGLang with RadixAttention (tail_first chunk eviction)
print("SGLang (RadixAttention / tail_first)...")
base2 = SimConfig.from_json("configs/heavy_coding.json")
adapter_s = SGLangConfigAdapter()
cfg_sglang = adapter_s.from_sglang_config(
    {"radix_attention": True, "max_running_requests": 256},
    model=MODEL, base_config=base2,
)
sglang_result = run_and_measure(cfg_sglang, "SGLang RadixAttention")

hit_delta = sglang_result["hit_rate"] - vllm_result["hit_rate"]

results = {
    "vllm": vllm_result,
    "sglang": sglang_result,
    "hit_rate_delta_pct": round(hit_delta * 100, 1),
    "answer": f"SGLang hit rate {sglang_result['hit_rate']:.3f} vs vLLM {vllm_result['hit_rate']:.3f} (delta={hit_delta*100:+.1f}pt)",
    "caveat": "SGLang RadixAttention modeled as tail_first chunk eviction, not actual radix tree",
}

print(f"\n=== Q11 Answer ===\n  {results['answer']}")

with open(OUT / "q11_sglang_vs_vllm.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT / 'q11_sglang_vs_vllm.json'}")
