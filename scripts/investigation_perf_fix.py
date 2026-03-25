#!/usr/bin/env python3
"""Verify session index perf fix produces identical results + benchmark speedup.

Compares old engine (sim/) vs new engine (agentsim/) on the same config.
The new engine has O(1) has_session_cached via session index set.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent.parent / "results" / "investigations"
OUT.mkdir(parents=True, exist_ok=True)

from sim.config import SimConfig as OldConfig
from sim.engine import SimEngine as OldEngine
from agentsim.core.des.config import SimConfig as NewConfig
from agentsim.core.des.engine import SimEngine as NewEngine

results = {}

# Test 1: Small sim — verify identical results
print("=== Test 1: Identical results (2min, 4W local, peak=100) ===")
for label, CfgClass, EngClass in [("old", OldConfig, OldEngine), ("new", NewConfig, NewEngine)]:
    cfg = CfgClass.from_json("configs/heavy_coding.json")
    cfg.sim_duration_s = 120.0; cfg.warmup_s = 12.0; cfg.sim_start_time_s = 36000.0
    cfg.service.n_prefill_nodes = 32; cfg.service.n_gpus_per_worker = 8
    cfg.service.l3a_shared = False; cfg.service.dispatch_algorithm = "pull"

    start = time.monotonic()
    m = EngClass(cfg).run()
    elapsed = time.monotonic() - start

    completed = sum(m.savings_events.values())
    hit = m.report()["cache_hit_rate"]
    results[label] = {"time_s": round(elapsed, 1), "completed": completed, "hit": hit}
    print(f"  {label}: {elapsed:.1f}s completed={completed} hit={hit}")

match = results["old"]["completed"] == results["new"]["completed"]
hit_match = results["old"]["hit"] == results["new"]["hit"]
speedup = results["old"]["time_s"] / max(0.1, results["new"]["time_s"])

print(f"\n  Completed match: {match}")
print(f"  Hit rate match: {hit_match}")
print(f"  Speedup: {speedup:.1f}x ({results['old']['time_s']}s → {results['new']['time_s']}s)")

# Test 2: Larger sim — measure scaling
print("\n=== Test 2: Scaling (5min, 4W local, peak=100) — new engine only ===")
cfg = NewConfig.from_json("configs/heavy_coding.json")
cfg.sim_duration_s = 300.0; cfg.warmup_s = 30.0; cfg.sim_start_time_s = 36000.0
cfg.service.n_prefill_nodes = 32; cfg.service.n_gpus_per_worker = 8
cfg.service.l3a_shared = False; cfg.service.dispatch_algorithm = "pull"

start = time.monotonic()
m = NewEngine(cfg).run()
elapsed = time.monotonic() - start
print(f"  New engine 5min: {elapsed:.1f}s (was 2668s = 44min before fix)")
results["5min_new"] = {"time_s": round(elapsed, 1), "completed": sum(m.savings_events.values())}

results["identical"] = match and hit_match
results["speedup_2min"] = round(speedup, 1)

with open(OUT / "perf_fix_verification.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT / 'perf_fix_verification.json'}")
