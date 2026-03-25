# Spec Consistency Gap Analysis
# Original artifacts vs. agentsim_final_plan.md

---

## Summary: 4 files need updates, 1 file is superseded, 1 is fine

| Artifact | Status | Gap Severity | Action |
|----------|--------|--------------|--------|
| `core/hardware_model.py` | Inconsistent | HIGH | Split + rewrite |
| `sim/request_sim.py` | Inconsistent | HIGH | Move + constrain |
| `core/session_model.py` | Inconsistent | MEDIUM | Demote + annotate |
| `core/events.py` | Mostly consistent | LOW | Move + re-position |
| `agentsim_design.md` | Superseded | — | Archive, point to final plan |
| `simulator_architecture_v2.md` | Superseded | — | Archive |

---

## File-by-File Gaps

### 1. `core/hardware_model.py` → HIGH severity

Three specific violations of final plan principles:

**Gap A: `page_size_tokens` is token-based (violates byte-based accounting)**
```python
# CURRENT — wrong
page_size_tokens: int   # "4K page requirement"

# FINAL PLAN requires
# "all tier capacities are bytes, tokens are metadata"
# page size expressed as block_size_bytes in TierSpec
```

**Gap B: `HardwareModel` roofline is positioned as the core oracle**
The final plan demotes roofline to `ConfidenceLabel.ANALYTICAL_ONLY` fallback.
The primary oracle must be piecewise-linear from benchmark tables.
`HardwareModel` as written has no confidence label and no table-lookup path.

**Gap C: Wrong file location and wrong responsibility boundary**
Final plan splits this file into two:
- `integration/chips/profiles.py` → ChipProfile + ModelConfig (no latency math)
- `core/des/oracle.py` → CacheOracleBase with piecewise + roofline fallback

**Fix**: Split the file. ChipProfile and ModelConfig move to
`integration/chips/profiles.py`. HardwareModel becomes a roofline
fallback oracle in `core/des/oracle.py` wrapped with confidence label.
The `page_size_tokens` field is removed from ChipProfile entirely —
page size is now `block_size_bytes` on `TierSpec`.

---

### 2. `sim/request_sim.py` → HIGH severity

Two specific violations:

**Gap A: `KVCacheManager` is owned by the SimPy layer**
The final plan states explicitly:
> "SimPy should never own cache state logic independently.
>  It can call simplified estimators, but eviction, tier accounting,
>  and reuse semantics should be defined once in the Core DES model."

`KVCacheManager` in `request_sim.py` directly tracks `hbm_used_tokens`,
`ddr_used_tokens`, fires `EVICTION_TRIGGER` events, and manages per-session
state. This is exactly the prohibited pattern.

**Gap B: Wrong file location**
Final plan puts SimPy in `sweep/runner.py`, not `sim/request_sim.py`.
The `sim/` directory is reserved for the Core DES layer (ported from cache-sim).

**Fix**: Move to `sweep/request_sweep.py`. Strip out `KVCacheManager`
entirely. Replace with a call to `SweepEstimator.estimate_turn_latency_ms()`
which takes cache_hit and cache_tier as inputs (pre-computed from a
simplified lookup, not tracked state). Add "sweep-estimate" label to all outputs.

---

### 3. `core/session_model.py` → MEDIUM severity

**Gap A: Wrong location**
Final plan: session model is demoted, replaced by cache-sim's NHPP workload
profiles in `core/des/workload.py`. The current `session_model.py` in `core/`
implies it is a core dependency. It should live in `sweep/` as the
session generator for the sweep tool only.

**Gap B: `AgenticSessionGenerator` uses simple Poisson arrivals**
Final plan adopts NHPP with diurnal pattern from cache-sim.
Current code uses `rng.expovariate(arrival_rate_qps)` — simple Poisson,
no time-of-day variation. The steady-state behavior cache-sim studies
(L2 saturation at ~5 min, session migration at ~5-20 min) depends on
realistic sustained load that the diurnal NHPP models properly.

**Gap C: `TokenProfileDistribution` is not calibrated to cache-sim profiles**
Final plan adopts cache-sim's validated profiles directly:
- coding: 20k system prefix, 2-10k context growth/turn, 1-hour sessions
- agentic_coding: 30k prefix, 5-20k growth/turn, 30-min sessions

Current `TokenProfileDistribution` has `system_prompt_tokens=8000` and
`max_turns=25` — not matching the validated profiles.

**Fix**: Annotate file clearly as "sweep tool only, not authoritative".
Mark the class as `SweepSessionGenerator` to make the role explicit.
Keep the code as-is for now — it gets replaced properly in Phase 1 when
`core/des/workload.py` is ported from cache-sim.

---

### 4. `core/events.py` → LOW severity

**Gap A: File location**
Final plan puts this in `core/observation/events.py`, not `core/events.py`.
The `core/` top level is reserved for contracts and DES layer.
Observation layer lives in `core/observation/`.

**Gap B: `AnthropicEventMapper` drives events rather than consuming them**
The mapper currently receives explicit method calls:
```python
mapper.on_message_start(session_id, turn_id, sim_time_ms, ...)
mapper.on_first_content_delta(...)
```
Final plan: mappers implement `ObserverBase.on_event(DESEvent)`.
They are called by the DES engine, not driven by external callers.

The content of `events.py` is otherwise good — `CheckpointEvent`,
`CacheMissType`, `CacheHitType`, `EventStream`, `miss_summary()` are
all correctly aligned with the final plan. The miss taxonomy
(COLD/EXPIRY/EVICTION/TRANSFER) is correct. The issue is purely
positioning and the call-direction of the mappers.

**Fix**: Move file. Add `ObserverBase` ABC. Refactor
`AnthropicEventMapper` to implement `on_event(DESEvent)` and derive
checkpoint events from DES payloads instead of receiving explicit calls.
The existing logic moves inside `on_event` dispatch.

---

### 5. `agentsim_design.md` → Superseded

This was the locked design document from before cache-sim analysis.
It describes the SimPy request-level engine as primary, uses a 2-tier
HBM+DDR model, and has a Phase 1-4 plan that's been fully replaced.

The Goal 1/2/3 framing and the protocol event schema are still correct
and preserved in the final plan. Everything else is superseded.

**Fix**: Add a header note: "SUPERSEDED — see agentsim_final_plan.md".
Retain for historical reference.

---

### 6. `simulator_architecture.md` and `simulator_architecture_v2.md` → Superseded

Both are pre-final-plan architecture sketches. The v2 was the "right
direction" that the final plan refines. Both are superseded.

**Fix**: Add header notes. Retain for historical reference.

---

## The One Thing That Is Correct and Needs No Change

`ModelConfig` in `hardware_model.py` — the fields (`num_layers`,
`num_kv_heads`, `head_dim`, `dtype_bytes`) and the `bytes_per_token_kv`
and `total_bytes_per_token_kv` properties are correct and align with
the byte-based accounting principle. This moves cleanly to
`integration/chips/profiles.py` unchanged.
