#!/usr/bin/env python3
"""Q5: What fraction of misses invisible to OpenAI vs Anthropic client?

Run both observers on same sim. Compare miss detection rates.
Anthropic has explicit cache tokens; OpenAI infers from TTFT.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine
from agentsim.core.observation.events import EventStream, EventKind, CacheMissType, CacheHitType
from agentsim.core.observation.anthropic import AnthropicProtocolObserver
from agentsim.core.observation.openai_chat import OpenAIProtocolObserver

OUT = Path(__file__).resolve().parent.parent / "results" / "phase2-baseline"
OUT.mkdir(parents=True, exist_ok=True)

cfg = SimConfig.from_json("configs/heavy_coding.json")
cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
cfg.service.n_prefill_nodes = 8; cfg.service.n_gpus_per_worker = 8
cfg.service.dispatch_algorithm = "pull"

stream_a = EventStream()
stream_o = EventStream()
obs_a = AnthropicProtocolObserver(stream_a)
obs_o = OpenAIProtocolObserver(stream_o, baseline_ttft_us=50_000_000)  # 50s baseline

SimEngine(cfg, observers=[obs_a, obs_o]).run()

# Anthropic decisions
a_decisions = stream_a.filter(kind=EventKind.CACHE_DECISION)
a_hits = sum(1 for e in a_decisions if e.cache_hit_type != CacheHitType.NONE)
a_misses = sum(1 for e in a_decisions if e.cache_hit_type == CacheHitType.NONE)
a_expiry = sum(1 for e in a_decisions if e.cache_miss_type == CacheMissType.EXPIRY)

# OpenAI decisions
o_decisions = stream_o.filter(kind=EventKind.CACHE_DECISION)
o_hits = sum(1 for e in o_decisions if e.cache_hit_type != CacheHitType.NONE)
o_misses = sum(1 for e in o_decisions if e.cache_hit_type == CacheHitType.NONE)

total = len(a_decisions)
invisible = abs(a_misses - o_misses)

results = {
    "total_decisions": total,
    "anthropic_hits": a_hits,
    "anthropic_misses": a_misses,
    "anthropic_expiry": a_expiry,
    "openai_hits": o_hits,
    "openai_misses": o_misses,
    "observability_gap": invisible,
    "gap_pct": round(invisible / max(1, total) * 100, 1),
    "answer": f"OpenAI misclassifies {invisible} decisions ({invisible/max(1,total)*100:.1f}%) vs Anthropic ground truth",
}

print("=== Q5 Answer: Observability Gap ===")
print(f"  Anthropic: {a_hits} hits, {a_misses} misses (expiry={a_expiry})")
print(f"  OpenAI:    {o_hits} hits, {o_misses} misses")
print(f"  Gap: {invisible} decisions differ ({invisible/max(1,total)*100:.1f}%)")

with open(OUT / "q5_observability_gap.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT / 'q5_observability_gap.json'}")
