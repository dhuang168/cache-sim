# Realistic Coding Workload — Progress #2: Diurnal Bug Fix

Date: 2026-03-21

## Bug Found and Fixed

### Root cause
Short sims (60s) started at t=0 (midnight). The NHPP diurnal model has zero arrival rate at midnight for profiles with `diurnal_peak_trough_ratio=4.0`. Only `batch` (ratio=1.5) generated traffic. All v1 and v2 results for non-batch profiles were **silently empty**.

### Fix
Added `sim_start_time_s` field to `SimConfig` (default: 0). Short sim configs now set `sim_start_time_s=36000` (10 AM). Applied to: `sanity_plots.short_config()`, `test_invariants._short_config()`, `test_multinode._quick_config()`.

## Results After Fix

### Traffic mix now correct
| Profile | Sessions | KV Objects | Mean Tokens | Mean KV Size |
|---------|----------|------------|-------------|-------------|
| batch | 8,456 | 8,322 | 2,097 | 655 MB |
| chat | 2,067 | 269 | 2,591 | 809 MB |
| coding | 1,008 | 89 | 26,377 | 8.2 GB |
| agent | 610 | 137 | 4,538 | 1.4 GB |
| agentic_coding | 430 | 63 | 49,885 | 15.6 GB |

### Global vs Local L3A gap now visible
| Nodes | Global Hit Rate | Local Hit Rate | Gap |
|-------|----------------|----------------|-----|
| 1 | 100% | 100% | 0% |
| 2 | 100% | 90.4% | 9.6% |
| 4 | 100% | 60.5% | **39.5%** |
| 8 | 99.8% | 100% | -0.2% |

### TTL sensitivity now shows real differences
- Global L3A: ~100% hit rate across all L2 TTLs (except TTL=1s: 97.9%)
- Local L3A: 59-65% hit rate regardless of TTL — capacity is the bottleneck, not TTL

### Crosscheck
| Plan Item | Status |
|-----------|--------|
| Update coding profile | Done |
| Add agentic_coding profile | Done |
| Fix diurnal bug | Done (new) |
| Verify traffic mix | Done |
| 22/22 tests pass | Done |
| Regenerate plots | Done |
| Update docs (user/session model, bug history) | Done |
| Commit + push | Pending |
