#!/usr/bin/env python3
"""I3: DESEvent emission coverage audit.

Hypothesis: Not all prefill completions emit DESEvents — some code paths skip.
Count engine prefill completions vs DESEvents emitted.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine
from agentsim.core.contracts import ObserverBase, DESEvent, DESEventKind

OUT = Path(__file__).resolve().parent.parent / "results" / "investigations"
OUT.mkdir(parents=True, exist_ok=True)

class EventCounter(ObserverBase):
    def __init__(self):
        self.counts = {}
    def on_event(self, event: DESEvent):
        k = event.kind.value
        self.counts[k] = self.counts.get(k, 0) + 1

cfg = SimConfig.from_json("configs/heavy_coding.json")
cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
cfg.service.n_prefill_nodes = 8; cfg.service.n_gpus_per_worker = 8
cfg.service.dispatch_algorithm = "pull"

counter = EventCounter()
m = SimEngine(cfg, observers=[counter]).run()

engine_completed = sum(m.savings_events.values())
engine_prefill_dur = len(m.prefill_duration_us)
des_prefill = counter.counts.get("prefill_complete", 0)
des_decode = counter.counts.get("decode_complete", 0)

# The warmup filter: engine collects metrics only after warmup,
# but DESEvent emission happens for ALL events (no warmup filter on _emit)
warmup_us = int(cfg.warmup_s * 1e6)
sim_us = int(cfg.sim_duration_s * 1e6)
collection_fraction = (sim_us - warmup_us) / sim_us

results = {
    "engine_savings_events": engine_completed,
    "engine_prefill_duration_samples": engine_prefill_dur,
    "des_prefill_complete": des_prefill,
    "des_decode_complete": des_decode,
    "des_all_events": counter.counts,
    "warmup_s": cfg.warmup_s,
    "sim_duration_s": cfg.sim_duration_s,
    "collection_fraction": round(collection_fraction, 2),
}

print("=== I3 Answer: DESEvent Emission Coverage ===")
print(f"  Engine savings events (post-warmup): {engine_completed}")
print(f"  Engine prefill duration samples: {engine_prefill_dur}")
print(f"  DES PREFILL_COMPLETE events: {des_prefill}")
print(f"  DES DECODE_COMPLETE events: {des_decode}")
print(f"  DES all events: {counter.counts}")
print(f"  Collection window: {collection_fraction*100:.0f}% of sim")

if des_prefill > 0:
    ratio = engine_prefill_dur / des_prefill
    print(f"  Ratio engine_prefill_samples / DES_prefill: {ratio:.2f}")
    results["ratio_engine_to_des"] = round(ratio, 2)

    if abs(ratio - 1.0) < 0.05:
        results["diagnosis"] = "1:1 match — emission covers all post-warmup prefills. Q4 gap is because observer was counting total decisions (including warmup-filtered cache decisions) differently."
    else:
        results["diagnosis"] = f"Mismatch: ratio={ratio:.2f}. Investigate which code paths skip _emit."
else:
    results["diagnosis"] = "No DES events — _emit may not be called"

print(f"  Diagnosis: {results['diagnosis']}")

with open(OUT / "i3_emission_coverage.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT / 'i3_emission_coverage.json'}")
