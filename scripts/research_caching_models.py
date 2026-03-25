#!/usr/bin/env python3
"""Compare three caching models: OpenAI prefix, Anthropic segment, LMCache chunk.

Simulates the same heavy coding workload under each model's caching semantics:

1. OpenAI Prefix: Single contiguous prefix per session, 1024-token minimum,
   128-token increments, cross-session sharing of identical prefixes.
   → Our object model with sharing enabled + block_size=128.

2. Anthropic Segment: Up to 4 breakpoints per session. System prompt (shared),
   conversation history (per-session), recent context (per-session).
   → Our object model with multi-tier sharing enabled.

3. LMCache Chunk: 256-token chunks, hash-deduped, tail-first eviction.
   → Our chunk model with tail_first.

4. Baseline: Per-session objects, no cross-session sharing.
   → Our default object model.
"""
import sys, json, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.core.des.config import SimConfig, SharingConfig, SharingTier, CacheConfig
from agentsim.core.des.engine import SimEngine

OUT = Path(__file__).resolve().parent.parent / "results" / "caching_models"
OUT.mkdir(parents=True, exist_ok=True)

PEAK = 15
SIM_DUR = 300.0
WARMUP = 30.0


def run_config(cfg, label):
    cfg.sim_duration_s = SIM_DUR
    cfg.warmup_s = WARMUP
    cfg.sim_start_time_s = 36000.0
    cfg.service.n_prefill_nodes = 32
    cfg.service.n_gpus_per_worker = 8
    cfg.service.dispatch_algorithm = "pull"
    for p in cfg.profiles:
        p.arrival_rate_peak = PEAK

    start = time.monotonic()
    m = SimEngine(cfg).run()
    elapsed = time.monotonic() - start
    r = m.report()

    completed = sum(m.savings_events.values())
    hit = r.get("cache_hit_rate", {})
    combined_hit = 1.0 - hit.get("miss", 1.0)
    all_ttft = []
    for v in m.ttft_us.values():
        all_ttft.extend(v)
    tier_sat = r.get("tier_saturation_pct", {})
    sharing = r.get("sharing_factor", 1.0)

    result = {
        "label": label,
        "completed": completed,
        "combined_hit_rate": round(combined_hit, 4),
        "hit_rate": {k: round(v, 4) for k, v in hit.items()},
        "ttft_p50_s": round(np.percentile(all_ttft, 50) / 1e6, 2) if all_ttft else 0,
        "ttft_p95_s": round(np.percentile(all_ttft, 95) / 1e6, 2) if all_ttft else 0,
        "recompute_mean": round(np.mean(m.recompute_fraction), 4) if m.recompute_fraction else 0,
        "tier_saturation": tier_sat,
        "sharing_factor": sharing,
        "elapsed_s": round(elapsed, 1),
    }
    # Chunk-specific
    if m.chunk_total_logical > 0:
        result["dedup_ratio"] = round(m.chunk_dedup_hits / max(1, m.chunk_total_logical), 4)

    print(f"  {label}: hit={combined_hit:.3f} recomp={result['recompute_mean']:.3f} "
          f"TTFT_p50={result['ttft_p50_s']}s sharing={sharing:.2f} ({elapsed:.1f}s)")
    return result


results = {}

# ─── Model 1: Baseline (per-session objects, no sharing) ───
print("Model 1: Baseline (per-session, no sharing)")
cfg1 = SimConfig.from_json("configs/heavy_coding.json")
results["baseline"] = run_config(cfg1, "Baseline (per-session)")

# ─── Model 2: OpenAI Prefix (object + sharing + 128-token blocks) ───
print("\nModel 2: OpenAI Prefix (shared prefix + 128-token blocks)")
cfg2 = SimConfig.from_json("configs/heavy_coding.json")
cfg2.cache.block_size_tokens = 128  # OpenAI's 128-token increment
# Enable shared system prefix (already in profiles via shared_system_prefix_tokens)
# The existing engine handles this via the shared prefix trie
results["openai_prefix"] = run_config(cfg2, "OpenAI Prefix (128-tok blocks)")

# ─── Model 3: Anthropic Segment (multi-tier sharing) ───
print("\nModel 3: Anthropic Segment (multi-tier sharing, 3 breakpoints)")
cfg3 = SimConfig.from_json("configs/heavy_coding.json")
cfg3.cache.sharing = SharingConfig(
    enabled=True,
    tiers=[
        # Breakpoint 1: System prompt (shared across all users of same profile)
        SharingTier(name="system", tokens=20000, sharing_group_size=1000),
        # Breakpoint 2: Common workspace context (shared within team of 10)
        SharingTier(name="workspace", tokens=5000, sharing_group_size=10),
        # Breakpoint 3: Session-unique context (no sharing)
        SharingTier(name="session", tokens=0, sharing_group_size=1),
    ],
)
results["anthropic_segment"] = run_config(cfg3, "Anthropic Segment (3 breakpoints)")

# ─── Model 4: LMCache Chunk (256-token, tail-first) ───
print("\nModel 4: LMCache Chunk (256-tok, tail-first, demand-pull)")
cfg4 = SimConfig.from_json("configs/heavy_coding.json")
cfg4.cache.block_size_tokens = 256
cfg4.cache.eviction_policy = "lru"
cfg4.cache.deduplication = "chunk"
cfg4.cache.tier_migration = "demand_pull"
cfg4.cache.chunk_eviction = "tail_first"
results["lmcache_chunk"] = run_config(cfg4, "LMCache Chunk (256-tok tail-first)")

# ─── Summary ───
print("\n" + "=" * 90)
print("SUMMARY: Caching Model Comparison (4W, peak=15, 5min, heavy_coding)")
print("=" * 90)
print(f"{'Model':<40} {'Hit%':>6} {'Recomp':>7} {'TTFT p50':>9} {'Sharing':>8}")
print("-" * 90)
for key in ["baseline", "openai_prefix", "anthropic_segment", "lmcache_chunk"]:
    r = results[key]
    dedup = f" dedup={r.get('dedup_ratio', 'N/A')}" if "dedup_ratio" in r else ""
    print(f"{r['label']:<40} {r['combined_hit_rate']*100:>5.1f}% {r['recompute_mean']*100:>6.1f}% "
          f"{r['ttft_p50_s']:>8.2f}s {r['sharing_factor']:>7.2f}{dedup}")

with open(OUT / "caching_models_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT / 'caching_models_comparison.json'}")
