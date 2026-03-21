# Workload Mix Fix + Latency Impact Metrics — Plan #2

Date: 2026-03-21

## Problem (a): Workload mix doesn't match profile_mix

### Current behavior
`profile_mix` only gates whether a profile stream is active (weight > 0). The actual session proportion is determined by `arrival_rate_peak` alone:
- batch: 0.20 × 200 peak = 40 eff. sessions/s → 67% of sessions
- chat: 0.30 × 100 peak = 30 → 16%
- coding: 0.20 × 50 peak = 10 → 8%
- agent: 0.15 × 30 peak = 4.5 → 5%
- agentic_coding: 0.15 × 20 peak = 3 → 3%

### Fix
Scale effective arrival rate by `profile_mix` weight in `WorkloadSynthesizer`. The effective peak rate becomes `arrival_rate_peak * profile_mix[name]`. This makes `profile_mix` actually control proportions:
- batch: 200 × 0.20 = 40/s → 26%
- chat: 100 × 0.30 = 30/s → 19%
- coding: 50 × 0.20 = 10/s → 6.5%
- agent: 30 × 0.15 = 4.5/s → 2.9%
- agentic_coding: 20 × 0.15 = 3/s → 1.9%

Hmm, batch still dominates because its `arrival_rate_peak` is highest. The real fix is to either:
1. Make `arrival_rate_peak` represent the desired absolute rate (already does), and set values to match desired proportions
2. Or add a `total_arrival_rate_peak` config field and derive per-profile rates from `total × profile_mix[name]`

**Chosen approach**: Option 1 — adjust `arrival_rate_peak` values so proportions match `profile_mix`. Set all profiles to the same `arrival_rate_peak` and let `profile_mix` control proportions via the rate scaling.

New arrival rates (targeting ~100 total sessions/s at peak, proportional to mix):
- chat: 100 (0.30 × 100 = 30/s)
- coding: 100 (0.20 × 100 = 20/s)
- batch: 100 (0.20 × 100 = 20/s)
- agent: 100 (0.15 × 100 = 15/s)
- agentic_coding: 100 (0.15 × 100 = 15/s)

This way, with `profile_mix` scaling, actual rates are 30:20:20:15:15, matching the configured mix.

### Code change
In `WorkloadSynthesizer`, scale the rate computation by `profile_mix[name]`:
- `diurnal_rate()`: multiply by mix weight
- `sample_next_arrival_time()`: use `rate_max * mix_weight` as the thinning bound

## Problem (b): Metrics to validate 50ms latency → eviction hypothesis

### Hypothesis
Global L3A's 50ms remote latency → longer prefills → slots occupied longer → more concurrent cache objects → more L3A pressure → more cold evictions → lower hit rate.

### Metrics needed to validate
1. **`prefill_duration_us: list[int]`** — actual prefill compute time per request. Should be higher with global L3A due to remote latency.
2. **`slot_utilization_pct` time series** — fraction of slots busy at each epoch. Should be higher with global L3A.
3. **`l3a_object_count` time series** — number of objects in L3A at each epoch. Should show more pressure with global L3A.
4. **`cold_evictions_per_epoch` time series** — when are cold evictions happening? Should correlate with high L3A occupancy.

### Comparison test
Run global vs local L3A side by side, compare:
- Mean prefill duration (global should be ~50ms higher for L3A hits)
- Slot utilization (global should be higher)
- L3A object count trajectory (global should be more crowded)
- Cold eviction timing (global should evict more under pressure)

## Phase 1: Fix workload mix
- Update `WorkloadSynthesizer.diurnal_rate()` and `sample_next_arrival_time()` to scale by `profile_mix` weight
- Update `arrival_rate_peak` values in `default.json` to uniform 100
- Verify actual session proportions match `profile_mix`

## Phase 2: Add latency impact metrics
- Add `prefill_duration_us`, `slot_utilization_pct`, `l3a_object_count`, `cold_evictions_per_epoch` to MetricsCollector
- Record in engine's `_on_prefill_complete` and `_on_epoch_report`

## Phase 3: Tests, traffic mix verification, plots, docs, crosscheck, commit
