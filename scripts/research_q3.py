#!/usr/bin/env python3
"""Q3: Fraction of sessions experiencing TTL expiry miss.

Run agentic_coding profile (heavy), 5 min, with Anthropic observer.
Record expiry miss rate from the 5-min TTL tracker.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine
from agentsim.core.observation.events import EventStream, EventKind, CacheMissType
from agentsim.core.observation.anthropic import AnthropicProtocolObserver

OUT = Path(__file__).resolve().parent.parent / "results" / "phase1-baseline"
OUT.mkdir(parents=True, exist_ok=True)

cfg = SimConfig.from_json("configs/heavy_coding.json")
cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
cfg.service.n_prefill_nodes = 8; cfg.service.n_gpus_per_worker = 8
cfg.service.dispatch_algorithm = "pull"

stream = EventStream()
observer = AnthropicProtocolObserver(stream)
m = SimEngine(cfg, observers=[observer]).run()

summary = stream.miss_summary()
decisions = stream.filter(kind=EventKind.CACHE_DECISION)
expiry_count = sum(1 for e in decisions if e.cache_miss_type == CacheMissType.EXPIRY)
total_decisions = len(decisions)

results = {
    "total_cache_decisions": total_decisions,
    "expiry_misses": expiry_count,
    "expiry_rate": round(expiry_count / max(1, total_decisions), 4),
    "miss_summary": summary,
    "answer": f"{expiry_count} expiry misses out of {total_decisions} decisions ({expiry_count/max(1,total_decisions)*100:.1f}%)",
}

print("=== Q3 Answer: TTL Expiry Miss Rate ===")
print(f"  Total decisions: {total_decisions}")
print(f"  Expiry misses: {expiry_count} ({expiry_count/max(1,total_decisions)*100:.1f}%)")
print(f"  Cold misses: {summary.get('cold_miss_rate', 0)*100:.1f}%")
print(f"  Eviction misses: {summary.get('eviction_miss_rate', 0)*100:.1f}%")
print(f"  Full hit rate: {summary.get('full_hit_rate', 0)*100:.1f}%")

with open(OUT / "q3_ttl_expiry.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT / 'q3_ttl_expiry.json'}")
