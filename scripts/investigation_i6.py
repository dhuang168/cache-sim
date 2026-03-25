#!/usr/bin/env python3
"""I6: Non-consecutive chunk reuse (CacheBlend model).

Hypothesis: Counting ALL cached chunks (not just consecutive from 0)
will dramatically improve chunk-mode hit rate.

Approach: Run chunk mode and measure:
1. Current consecutive hit count
2. Total cached chunks (including after gaps)
3. Theoretical recompute if we could reuse non-consecutive chunks
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine
from agentsim.core.contracts import ObserverBase, DESEvent, DESEventKind

OUT = Path(__file__).resolve().parent.parent / "results" / "investigations"
OUT.mkdir(parents=True, exist_ok=True)

# Run both object mode and chunk mode (tail_first) and compare
results = {}

# Object mode baseline
cfg_obj = SimConfig.from_json("configs/heavy_coding.json")
cfg_obj.sim_duration_s = 300.0; cfg_obj.warmup_s = 30.0; cfg_obj.sim_start_time_s = 36000.0
cfg_obj.service.n_prefill_nodes = 8; cfg_obj.service.n_gpus_per_worker = 8
cfg_obj.service.dispatch_algorithm = "pull"
for p in cfg_obj.profiles: p.arrival_rate_peak = 15
m_obj = SimEngine(cfg_obj).run()
r_obj = m_obj.report()
results["object_mode"] = {
    "hit_rate": round(1.0 - r_obj["cache_hit_rate"].get("miss", 1.0), 4),
    "recompute_mean": round(np.mean(m_obj.recompute_fraction), 4) if m_obj.recompute_fraction else 0,
}
print(f"Object mode: hit={results['object_mode']['hit_rate']:.3f} recompute={results['object_mode']['recompute_mean']:.3f}")

# Chunk mode with tail_first
cfg_chunk = SimConfig.from_json("configs/heavy_coding.json")
cfg_chunk.sim_duration_s = 300.0; cfg_chunk.warmup_s = 30.0; cfg_chunk.sim_start_time_s = 36000.0
cfg_chunk.service.n_prefill_nodes = 8; cfg_chunk.service.n_gpus_per_worker = 8
cfg_chunk.service.dispatch_algorithm = "pull"
cfg_chunk.cache.block_size_tokens = 256
cfg_chunk.cache.eviction_policy = "lru"
cfg_chunk.cache.deduplication = "chunk"
cfg_chunk.cache.tier_migration = "demand_pull"
cfg_chunk.cache.chunk_eviction = "tail_first"
for p in cfg_chunk.profiles: p.arrival_rate_peak = 15
m_chunk = SimEngine(cfg_chunk).run()
r_chunk = m_chunk.report()
dedup = r_chunk.get("chunk_dedup", {})
results["chunk_tail_first"] = {
    "hit_rate": round(1.0 - r_chunk["cache_hit_rate"].get("miss", 1.0), 4),
    "recompute_mean": round(np.mean(m_chunk.recompute_fraction), 4) if m_chunk.recompute_fraction else 0,
    "dedup_ratio": dedup.get("dedup_ratio", 0),
    "novel_chunks": dedup.get("novel_chunks", 0),
    "dedup_hits": dedup.get("dedup_hits", 0),
}
print(f"Chunk tail_first: hit={results['chunk_tail_first']['hit_rate']:.3f} recompute={results['chunk_tail_first']['recompute_mean']:.3f} dedup={results['chunk_tail_first']['dedup_ratio']:.3f}")

# Analysis: what would CacheBlend achieve?
# If we could reuse all cached chunks (not just consecutive), recompute = gap_fraction
# The gap fraction is: 1 - (total_cached_chunks / total_chunks)
# We can estimate this from dedup_ratio: the fraction that are already cached = dedup_ratio
chunk_hit = results["chunk_tail_first"]["hit_rate"]
chunk_recomp = results["chunk_tail_first"]["recompute_mean"]
dedup_r = results["chunk_tail_first"]["dedup_ratio"]

# CacheBlend recompute model: only recompute ~15% of gap tokens + missing chunks
# Effective recompute ≈ (1 - dedup_ratio) * 0.15 (for blend) + (1 - dedup_ratio) * 0.85 (already computed)
# Actually simpler: if dedup_ratio = 0.5, half of chunks are shared/cached.
# CacheBlend would recompute only the missing chunks, not everything after the first gap.
# Estimated CacheBlend recompute = 1 - hit_rate (fraction of missing chunks)
# + 0.15 * hit_rate (selective recompute for blending)
cacheblend_recompute = (1 - chunk_hit) + 0.15 * chunk_hit
results["cacheblend_estimate"] = {
    "estimated_recompute": round(cacheblend_recompute, 4),
    "improvement_vs_consecutive": round((chunk_recomp - cacheblend_recompute) / max(0.01, chunk_recomp) * 100, 1),
    "note": "CacheBlend would recompute only missing chunks + 15% selective blend, not everything after first gap",
}

print(f"\n=== I6 Answer ===")
print(f"  Object mode:   hit={results['object_mode']['hit_rate']:.3f} recompute={results['object_mode']['recompute_mean']:.3f}")
print(f"  Chunk consec:  hit={results['chunk_tail_first']['hit_rate']:.3f} recompute={results['chunk_tail_first']['recompute_mean']:.3f}")
print(f"  CacheBlend est: recompute≈{cacheblend_recompute:.3f} ({results['cacheblend_estimate']['improvement_vs_consecutive']}% better than consecutive)")

with open(OUT / "i6_cacheblend_estimate.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT / 'i6_cacheblend_estimate.json'}")
