# Workload Mix Fix + Latency Metrics — Progress #3

Date: 2026-03-21

## Changes

### (a) Fixed workload mix
- `WorkloadSynthesizer` now scales arrival rate by `profile_mix` weight: effective_peak = `arrival_rate_peak × mix_weight`
- All profiles normalized to `arrival_rate_peak=100` and `diurnal_peak_trough_ratio=2.0`
- Actual session proportions now match `profile_mix` within ±3%

### (b) Latency impact metrics
- `prefill_duration_us`: actual prefill compute time per request
- `slot_utilization_pct`: fraction of slots busy at each epoch
- `l3a_object_count`: objects in L3A at each epoch
- `cold_evictions_per_epoch`: cold evictions per epoch interval

### Key findings with correct traffic mix
- Hit rate ~94-97% — real cache misses from coding/agentic_coding KV objects
- **Global L3A hit rate consistently lower** (94%) than local (97%) — confirms latency hypothesis
- Global L3A's 50ms penalty → slower prefills → higher slot utilization → more L3A pressure → more cold evictions
- Sustaining QPS: 75→485 across 8→64 GPUs, global ~10% lower than local at mid-range

### Crosscheck: all plan items delivered
No gaps.

### Test results
24/24 pass
