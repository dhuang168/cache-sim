#!/usr/bin/env python3
"""Q4: Does Anthropic miss taxonomy correctly classify all miss types?

Run heavy_coding through AnthropicProtocolObserver, verify COLD/EXPIRY/EVICTION
breakdown is consistent with SavingsEvent numbers from the engine.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine
from agentsim.core.observation.events import EventStream, EventKind, CacheMissType, CacheHitType
from agentsim.core.observation.anthropic import AnthropicProtocolObserver

OUT = Path(__file__).resolve().parent.parent / "results" / "phase2-baseline"
OUT.mkdir(parents=True, exist_ok=True)

cfg = SimConfig.from_json("configs/heavy_coding.json")
cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
cfg.service.n_prefill_nodes = 8; cfg.service.n_gpus_per_worker = 8
cfg.service.dispatch_algorithm = "pull"

stream = EventStream()
observer = AnthropicProtocolObserver(stream)
m = SimEngine(cfg, observers=[observer]).run()

# Engine-side metrics
engine_completed = sum(m.savings_events.values())
engine_hit_rate = m.report()["cache_hit_rate"]

# Observer-side metrics
decisions = stream.filter(kind=EventKind.CACHE_DECISION)
obs_total = len(decisions)
obs_hits = sum(1 for e in decisions if e.cache_hit_type != CacheHitType.NONE)
obs_cold = sum(1 for e in decisions if e.cache_miss_type == CacheMissType.COLD)
obs_expiry = sum(1 for e in decisions if e.cache_miss_type == CacheMissType.EXPIRY)
obs_eviction = sum(1 for e in decisions if e.cache_miss_type == CacheMissType.EVICTION)

results = {
    "engine_completed": engine_completed,
    "engine_hit_rate": engine_hit_rate,
    "observer_total_decisions": obs_total,
    "observer_hits": obs_hits,
    "observer_cold": obs_cold,
    "observer_expiry": obs_expiry,
    "observer_eviction": obs_eviction,
    "observer_hit_rate": round(obs_hits / max(1, obs_total), 4),
    "consistency_check": "PASS" if abs(obs_hits/max(1,obs_total) - (1 - engine_hit_rate.get("miss", 0))) < 0.05 else "DIVERGENT",
}

print("=== Q4 Answer: Anthropic Miss Taxonomy Validation ===")
print(f"  Engine: {engine_completed} completed, miss={engine_hit_rate.get('miss', 0)*100:.1f}%")
print(f"  Observer: {obs_total} decisions, hit={obs_hits/max(1,obs_total)*100:.1f}%")
print(f"    COLD: {obs_cold} ({obs_cold/max(1,obs_total)*100:.1f}%)")
print(f"    EXPIRY: {obs_expiry} ({obs_expiry/max(1,obs_total)*100:.1f}%)")
print(f"    EVICTION: {obs_eviction} ({obs_eviction/max(1,obs_total)*100:.1f}%)")
print(f"  Consistency: {results['consistency_check']}")

with open(OUT / "q4_anthropic_taxonomy.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT / 'q4_anthropic_taxonomy.json'}")
