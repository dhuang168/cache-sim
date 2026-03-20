# Example Sanity-Check Report

Results from running the full test suite and sanity-check plots against the simulator. This report validates that the core simulation logic is correct before using it for exploratory analysis.

## Test Suite Results

**13/13 tests passed** (pytest, Python 3.11, ~10 min total)

### Unit Tests

| Test | Module | Result | Description |
|------|--------|--------|-------------|
| `test_kv_size_1000_tokens` | `test_kv_size` | PASS | `kv_size_bytes(1000, 70B_fp16) == 327,680,000` |
| `test_kv_size_scales_linearly` | `test_kv_size` | PASS | 2x tokens = 2x bytes |
| `test_allocated_blocks_ceiling` | `test_kv_size` | PASS | Ceiling division correctness |
| `test_block_waste_ratio` | `test_kv_size` | PASS | Internal fragmentation calculation |
| `test_prefill_oracle_exact_points` | `test_oracle` | PASS | Exact table lookups (512 tok = 45ms, 1024 tok = 90ms) |
| `test_prefill_oracle_interpolation` | `test_oracle` | PASS | Linear interpolation between breakpoints |
| `test_prefill_oracle_clamp` | `test_oracle` | PASS | Clamping at table bounds |
| `test_transfer_time` | `test_oracle` | PASS | `transfer_time_us(64MB, L2) == 1010 us` |
| `test_decode_oracle` | `test_oracle` | PASS | Decode latency increases with batch size |

### Invariant Tests (boundary conditions)

These tests exercise extreme configurations to verify the simulator respects physical constraints.

| Test | Result | What It Proves |
|------|--------|----------------|
| `test_zero_ttl_collapses_l2` | PASS | TTL=0 forces all L2 objects to hibernate immediately; L2 occupancy stays ~0% |
| `test_infinite_l1_no_evictions` | PASS | L1 capacity = 2^50 bytes produces exactly 0 L1-to-L2 evictions |
| `test_no_shared_prefix_sharing_factor_one` | PASS | `shared_system_prefix_tokens=0` for all profiles yields `sharing_factor ~= 1.0` |
| `test_zero_bandwidth_penalty_prefers_restore` | PASS | Infinite bandwidth + zero latency floor produces 0 BREAK_EVEN events (restore is always worthwhile) |

## Sanity-Check Plots (Stressed Scenario)

The sanity plots use a **stressed configuration**: 500 MB L1 (too small for most KV objects), 10 GB L2, 50 GB L3A, TTL_L2=20s, TTL_L3A=120s. This forces multi-tier activity.

### Stressed Scenario Report

```
Sharing factor:  2.5
Hit rates:       L1=0.0%  L2=80.03%  L3A=19.97%  miss=0.0%
Tier saturation: L1=0.0%  L2=98.23%  L3A=99.04%
Evictions L1->L2:   0.0/s
Evictions L2->L3A:  0.5/s
Cold evictions:     10,891
```

**Key observations:**
- **L1 hit rate is 0%** because L1 is only 500 MB — a single 70B KV object at 2048 tokens is ~670 MB, so nothing fits in L1. All objects are placed directly into L2.
- **L2 hit rate is 80%** — the primary cache tier under this configuration. L2 saturation at 98% shows it is capacity-constrained.
- **L3A hit rate is 20%** with 99% saturation — hibernated objects are still useful but at higher latency cost.
- **Sharing factor of 2.5** — the 2048-token shared system prefix across sessions means each unique prefix token is served ~2.5 times on average.
- **10,891 cold evictions** from L3A — capacity pressure forces permanent loss of some cached KV state.

### Generated Plots

All plots saved to `plots/`:

| Plot | File | What It Shows |
|------|------|---------------|
| Tier occupancy over time | `tier_occupancy.png` | L2 and L3A saturate quickly; L1 stays empty (stressed config) |
| TTFT distribution | `ttft_distribution.png` | Violin plots by hit source — L2 hits are ~1ms, L3A hits are ~10ms, cold misses are ~50-200ms |
| Hit rate pie | `hit_rate_pie.png` | L2 WORTHWHILE dominates (~80%) |
| Queue depth | `queue_depth.png` | Prefill and decode queue utilization over time |
| Recompute fraction | `recompute_fraction.png` | Distribution of how much of each request must be recomputed |
| L1 sensitivity | `l1_sensitivity.png` | Stacked bar of hit rates + eviction rate vs L1 capacity (0.25-80 GB) |
| L2 sensitivity | `l2_sensitivity.png` | TTFT p50/p95 and L2 hit rate vs L2 capacity (2-50 GB) |

### Sensitivity Sweep Highlights

**L1 capacity sweep** (8 points: 0.25 GB to 80 GB):
- Below ~1 GB: L1 is useless, all traffic goes to L2/L3A
- At 4 GB: L1 starts absorbing small KV objects, eviction rate drops
- At 40-80 GB: L1 hit rate climbs significantly, eviction rate approaches zero

**L2 capacity sweep** (5 points: 2 GB to 50 GB):
- L2 hit rate increases with capacity (more objects survive before TTL expiry)
- TTFT p95 for L2 hits is relatively stable (dominated by transfer bandwidth, not capacity)

## How to Reproduce

```bash
# Run tests
pytest tests/ -v

# Generate plots
python scripts/sanity_plots.py

# Plots appear in plots/
```
