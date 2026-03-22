# Example Sanity-Check Report

Results from running the full test suite and sanity-check plots against the simulator. This report validates that the core simulation logic is correct before using it for exploratory analysis.

## Test Suite Results

**27/27 tests passed** (pytest, Python 3.11, ~4s total)

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
| `test_infinite_l1_no_evictions` | PASS | L1 capacity = 2^50 bytes produces exactly 0 L1-to-L2 pressure evictions |
| `test_no_shared_prefix_sharing_factor_one` | PASS | `shared_system_prefix_tokens=0` for all profiles yields `sharing_factor ~= 1.0` |
| `test_zero_bandwidth_penalty_prefers_restore` | PASS | Infinite bandwidth + zero latency floor produces 0 BREAK_EVEN events (restore is always worthwhile) |

### Multi-Node Tests

| Test | Result | What It Proves |
|------|--------|----------------|
| `test_single_node_backward_compat` | PASS | N=1 mode produces valid metrics, no dispatch stats tracked |
| `test_more_nodes_reduce_queue_pressure` | PASS | 4 nodes → lower per-node queue pressure than 1 node |
| `test_push_affinity_dispatches` | PASS | Push dispatch achieves cache affinity (affinity_dispatches > 0) |
| `test_pull_no_starvation` | PASS | Pull mode distributes work to all 4 nodes |
| `test_pull_affinity_matches` | PASS | Pull mode processes all requests, all nodes active |
| `test_queue_wait_metric` | PASS | Queue wait metric is populated and non-negative |
| `test_local_l3a_mode` | PASS | Local L3A mode produces valid metrics |
| `test_global_vs_local_l3a_hit_rate` | PASS | Global L3A miss rate ≤ local L3A miss rate |
| `test_worker_topology_shared_l2` | PASS | 8 GPUs / 2 workers: verifies L2 sharing and worker_id |
| `test_intra_worker_no_l2_penalty` | PASS | Same-worker L2 access has no cross-node penalty |
| `test_local_l3a_worker_isolation` | PASS | Object on worker 0 NOT visible from worker 1 (local L3A) |
| `test_global_l3a_cross_worker_access` | PASS | Global L3A visible from any node |
| `test_session_migration_global_advantage` | PASS | Tiny L1/L2 + 2 workers → global > local hit rate |
| `test_multinode_perf_benchmark` | PASS | 10s sim with 4 nodes completes quickly |

## Glossary: Savings Classes

Each cache lookup is classified into one of five **savings classes** that describe the cost-benefit of the cache hit:

| Class | Meaning |
|-------|---------|
| **CACHE_HIT_L1** | KV state found in L1 (HBM). Zero transfer cost — prefill uses cached KV directly. Best outcome. |
| **CACHE_HIT_L2_WORTHWHILE** | KV state found in L2 (DRAM). Transfer cost exists but is always less than full recompute, so restoring from L2 is always classified as worthwhile. |
| **CACHE_HIT_L3A_WORTHWHILE** | KV state found in L3A (SSD). Transfer latency is significant, but still faster than recomputing the full prefill from scratch. Net savings positive. |
| **CACHE_HIT_L3A_BREAK_EVEN** | KV state found in L3A, but transfer latency is comparable to or exceeds recompute cost (e.g., small KV with high SSD latency). Restoring from cache provides no net benefit over recomputing. |
| **COLD_MISS** | No cached KV state found anywhere. Full prefill recompute required. |

The worthwhile vs. break-even distinction is determined by `is_cache_worthwhile()` in `sim/oracle.py`: it compares the transfer time (`latency_floor + size/bandwidth`) against the recompute time from the prefill oracle. L2 hits are always classified as worthwhile because DRAM bandwidth (64 GB/s) makes transfer cost negligible relative to prefill compute. L3A hits may be break-even when the KV object is small (low recompute cost) but SSD latency floor is high.

## Sanity-Check Plots (Stressed Scenario)

The sanity plots use a **stressed configuration**: 500 MB L1 (too small for most KV objects), 10 GB L2, 50 GB L3A, TTL_L2=20s, TTL_L3A=120s, eviction_hbm_threshold=0.6. This forces multi-tier activity.

### Stressed Scenario Report

```
Sharing factor:  2.5
Hit rates:       L1=0.0%  L2=80.03%  L3A=19.97%  miss=0.0%
Tier saturation: L1=0.0%  L2=98.23%  L3A=99.04%
L1->L2 pressure:   0.0/s
L1->L2 TTL:        0.0/s
Evictions L2->L3A:  0.5/s
Cold evictions:     10,891
```

**Key observations:**
- **L1 hit rate is 0%** because L1 is only 500 MB — a single 70B KV object at 2048 tokens is ~670 MB, so nothing fits in L1. All objects are placed directly into L2.
- **L2 hit rate is 80%** — the primary cache tier under this configuration. L2 saturation at 98% shows it is capacity-constrained.
- **L3A hit rate is 20%** with 99% saturation — hibernated objects are still useful but at higher latency cost.
- **Sharing factor of 2.5** — the 2048-token shared system prefix across sessions means each unique prefix token is served ~2.5 times on average.
- **L1->L2 pressure and TTL are both 0/s** — no objects enter L1 at all (too small), so nothing is evicted from it.
- **10,891 cold evictions** from L3A — capacity pressure forces permanent loss of some cached KV state.

### Generated Plots

All plots saved to `plots/`:

| Plot | File | What It Shows |
|------|------|---------------|
| Tier occupancy over time | `tier_occupancy.png` | L2 and L3A saturate quickly; L1 stays empty (stressed config) |
| TTFT distribution | `ttft_distribution.png` | Violin plots by hit source — L2 hits are ~1ms, L3A hits are ~10ms, cold misses are ~50-200ms |
| Hit rate pie | `hit_rate_pie.png` | Savings class breakdown: L2 WORTHWHILE dominates (~80%), L3A WORTHWHILE (~20%) |
| Queue depth | `queue_depth.png` | Prefill queue saturated at ~128 (queue max); decode queue empty |
| Recompute fraction | `recompute_fraction.png` | Distribution of how much of each request must be recomputed |
| L1 sensitivity | `l1_sensitivity.png` | Stacked bar of hit rates + L1->L2 movement breakdown vs L1 capacity |
| L2 sensitivity | `l2_sensitivity.png` | TTFT p50/p95 and L2 hit rate vs L2 capacity (2-50 GB) |
| Node scaling | `node_scaling.png` | 6-panel: hit rate, eviction, queue wait, utilization, TTFT, L3A latency — global vs local L3A across [1,2,4,8] nodes |
| TTL sensitivity | `ttl_sensitivity.png` | Hit rate and queue wait vs L2 TTL — global vs local L3A at 4 nodes |
| Sustaining QPS | `sustaining_qps.png` | Max QPS at p95 queue wait < 500ms SLA vs node count — global vs local L3A |

### Plot Analysis

#### Queue Depth (`queue_depth.png`)

The prefill queue is saturated at ~128 (the configured `prefill_queue_max`), indicating significant backpressure on the prefill stage. This is expected in the stressed config because the high arrival rate (batch profile at 200 req/s) exceeds the 32 prefill slots' processing capacity. The decode queue remains at 0 because decode slots (256) are abundant relative to the decode throughput required.

**Bug fixed:** Previously showed flat zero for both queues because the epoch report was reading `self.service.prefill_queue` (unused) instead of `self._pending_prefills` (where the engine actually queues prefills).

#### L1 Sensitivity (`l1_sensitivity.png`)

The right panel now separates two types of L1->L2 movement:

- **Pressure evictions** (red): Objects evicted from L1 because a new object needs space and L1 occupancy exceeds the `eviction_hbm_threshold` (0.6). This is the "forced out" case.
- **TTL migrations** (blue dashed): Objects moved from L1 to L2 when their TTL timer (`ttl_l2_s=20s`) expires. This is planned tier demotion, not pressure.

**Interpreting the pressure eviction curve:**
- At L1 < 1 GB: Objects bypass L1 entirely (a single 2048-token KV is ~670 MB, so nothing fits). Zero evictions because nothing enters L1.
- At L1 >= 1 GB: Objects start entering L1. The pressure eviction rate jumps to ~220/s and plateaus. This plateau is **correct behavior**: once L1 can accept objects, every new arrival triggers an eviction of an older object because the occupancy threshold (60%) is quickly reached. The eviction rate equals the arrival rate — it's a steady-state flow through L1, not a sign of waste.
- TTL migrations stay near zero because the occupancy-based eviction fires first (at threshold 0.6, well before the 20s TTL expires).

**Why evictions don't decrease with larger L1:** The eviction rate is bounded by the arrival rate, not the capacity. A larger L1 holds more objects simultaneously, but the throughput (objects entering and leaving per second) remains constant because the same number of requests arrive per second. To reduce pressure evictions, you would need to either raise the eviction threshold or reduce the arrival rate.

### L2 Sensitivity

**L2 capacity sweep** (5 points: 2 GB to 50 GB):
- L2 hit rate increases with capacity (more objects survive before TTL expiry)
- TTFT p95 for L2 hits is relatively stable (dominated by transfer bandwidth, not capacity)
- This matches intuition: more L2 capacity means more objects stay warm longer.

## How to Reproduce

```bash
# Run tests
pytest tests/ -v

# Generate plots
python scripts/sanity_plots.py

# Plots appear in plots/
```
