#!/usr/bin/env python3
"""I4: Auto-calibrated OpenAI baseline.

Hypothesis: Using cold-miss p95 TTFT as baseline will reduce the observability gap.
Re-run Q5 with baseline auto-detected from engine's cold-miss TTFT distribution.
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine
from agentsim.core.observation.events import EventStream, EventKind, CacheHitType
from agentsim.core.observation.anthropic import AnthropicProtocolObserver
from agentsim.core.observation.openai_chat import OpenAIProtocolObserver

OUT = Path(__file__).resolve().parent.parent / "results" / "investigations"
OUT.mkdir(parents=True, exist_ok=True)

cfg = SimConfig.from_json("configs/heavy_coding.json")
cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
cfg.service.n_prefill_nodes = 8; cfg.service.n_gpus_per_worker = 8
cfg.service.dispatch_algorithm = "pull"

# First run: get cold-miss TTFT distribution for auto-calibration
stream_a = EventStream()
obs_a = AnthropicProtocolObserver(stream_a)
m = SimEngine(cfg, observers=[obs_a]).run()

# Extract cold-miss TTFTs from engine
cold_ttfts = m.ttft_us.get("cold_miss", [])
if cold_ttfts:
    baseline_us = int(np.percentile(cold_ttfts, 95))
else:
    # Fallback: use overall p95
    all_ttft = []
    for v in m.ttft_us.values(): all_ttft.extend(v)
    baseline_us = int(np.percentile(all_ttft, 95)) if all_ttft else 50_000_000

print(f"Auto-calibrated baseline: {baseline_us/1e6:.1f}s (cold-miss p95)")

# Compare different baselines
results = {}
for label, baseline in [("50s_fixed", 50_000_000), ("auto_p95", baseline_us), ("auto_p50", int(np.percentile(cold_ttfts, 50)) if cold_ttfts else 25_000_000)]:
    stream_a2 = EventStream()
    stream_o = EventStream()
    obs_a2 = AnthropicProtocolObserver(stream_a2)
    obs_o = OpenAIProtocolObserver(stream_o, baseline_ttft_us=baseline)
    SimEngine(cfg, observers=[obs_a2, obs_o]).run()

    a_dec = stream_a2.filter(kind=EventKind.CACHE_DECISION)
    o_dec = stream_o.filter(kind=EventKind.CACHE_DECISION)
    a_hits = sum(1 for e in a_dec if e.cache_hit_type != CacheHitType.NONE)
    o_hits = sum(1 for e in o_dec if e.cache_hit_type != CacheHitType.NONE)
    total = len(a_dec)
    gap = abs(a_hits - o_hits) / max(1, total) * 100

    results[label] = {
        "baseline_s": round(baseline / 1e6, 1),
        "anthropic_hits": a_hits,
        "openai_hits": o_hits,
        "total": total,
        "gap_pct": round(gap, 1),
    }
    print(f"  {label} (baseline={baseline/1e6:.1f}s): gap={gap:.1f}% (A:{a_hits} O:{o_hits})")

print(f"\n=== I4 Answer ===")
for k, v in results.items():
    print(f"  {k}: baseline={v['baseline_s']}s gap={v['gap_pct']}%")

with open(OUT / "i4_openai_baseline.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT / 'i4_openai_baseline.json'}")
