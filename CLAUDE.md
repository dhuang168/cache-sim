# CLAUDE.md — Project Intelligence for cache_sim

## What This Project Is

A discrete-event simulator for multi-tier (L1/L2/L3A) prompt KV cache systems targeting LLM inference workloads. The simulator models request lifecycles from arrival through prefill, decode, and KV cache placement, with TTL-based tier migration and occupancy-driven eviction.

## Quick Commands

```bash
# Run all tests
.venv/bin/python -m pytest tests/ -v

# Run just invariant tests
.venv/bin/python -m pytest tests/test_invariants.py -v

# Generate sanity plots
.venv/bin/python scripts/sanity_plots.py

# Run parameter sweep
.venv/bin/python scripts/sweep.py --config configs/default.json --output results/sweep.json
```

## Important: Do NOT append `2>&1` to pytest commands

This causes output issues. Run pytest without stderr redirection.

## Architecture Overview

- **sim/engine.py** — Main event loop. `SimEngine.run()` is the entry point. Uses a min-heap for event scheduling with microsecond sim clock.
- **sim/events.py** — Event types and request FSM. The FSM enforces legal state transitions (QUEUED -> CACHE_LOOKUP -> HIT/MISS -> PREFILLING -> DECODE_QUEUED -> DECODING -> KV_WRITE -> COMPLETE).
- **sim/cache.py** — `CacheObject`, `TierStore` (per-tier storage manager), `PrefixTrie` (token hash-based prefix matching), `kv_size_bytes()` formula.
- **sim/eviction.py** — `EvictionEngine` manages L1->L2->L3A movement. L1 eviction is watermark-based, L2->L3A is TTL-driven, L3A cleanup is LRU.
- **sim/oracle.py** — `PrefillOracle` (piecewise-linear interpolation from A100 benchmark table), `DecodeOracle` (sqrt batch degradation model), `transfer_time_us()`, `is_cache_worthwhile()`.
- **sim/workload.py** — `WorkloadSynthesizer` generates arrivals via NHPP (thinning algorithm) with sinusoidal diurnal modulation. Five profiles: chat, coding, batch, agent, agentic_coding.
- **sim/service.py** — `ServiceModel` tracks prefill/decode GPU slot pools with queue backpressure.
- **sim/metrics.py** — `MetricsCollector` accumulates TTFT, hit rates, evictions, occupancy, sharing factor. `.report()` produces the summary dict.
- **sim/config.py** — Dataclass-based config loaded from JSON. `SimConfig.from_json()` is the standard entry point.
- **sim/node.py** — `PrefillNode` class: per-node L1/L2 stores, prefill slots, pending queue, cache affinity check (`has_session_cached()`), queue pressure.
- **sim/dispatch.py** — `PushDispatcher` (affinity-aware routing) and `PullDispatcher` (global queue with affinity scoring). Push prefers nodes with session's KV cached; pull lets idle nodes pull from a global queue.
- **sim/analysis.py** — `find_sustaining_qps()` binary search for max arrival rate multiplier at a given SLA threshold (p95 queue wait).

## User/Session Model

The simulator models **multi-user concurrent workloads**:
- Each **session** = one user's conversation. Created by `_new_session()` with a unique `session_id`.
- Sessions are **independent**: each has its own `SessionState` (context growth, turn count), session prefix trie, and KV cache objects.
- **KV objects are per-session, not shared across users**. Two users with identical system prompts each store their own KV objects in the cache hierarchy. This models real LLM serving where KV cache tensors are per-sequence.
- The only **cross-session sharing** is the `shared_system_prefix_tokens` mechanism: a single shared prefix object (`sp-{profile_name}`) in the prefix trie avoids redundant recomputation of the system prompt during prefill. But each session still writes its own full KV object to L1/L2/L3A.
- With N concurrent coding users (each with 20k shared prefix + growing context), the cache must hold N separate multi-GB KV objects — creating real capacity pressure.

## Key Design Constraints

- **Sim clock is int64 microseconds** — never use floats for time in the engine.
- **Events must be scheduled in the future** — `assert event.time_us >= self.sim_clock_us`.
- **Request FSM is strict** — `validate_transition()` in events.py raises `SimError` on illegal transitions.
- **Metrics are only collected after warmup** — the `collecting` flag in `_dispatch()` controls this.
- **KV size formula**: `2 * n_layers * n_kv_heads * head_dim * bytes_per_element * token_count`. For 70B FP16: 327,680 bytes per token.

## Test Structure

### Invariant Tests (`tests/test_invariants.py`)
These are **mandatory** — must all pass before any exploratory run. They test boundary conditions:
1. `test_zero_ttl_collapses_l2` — TTL=0 means L2 occupancy stays ~0%
2. `test_infinite_l1_no_evictions` — L1 = 2^50 bytes means 0 evictions
3. `test_no_shared_prefix_sharing_factor_one` — No shared prefix means sharing_factor = 1.0
4. `test_zero_bandwidth_penalty_prefers_restore` — Infinite bandwidth means 0 BREAK_EVEN events

### Unit Tests
- `test_kv_size.py` — KV size math, block allocation, fragmentation
- `test_oracle.py` — Prefill/decode oracle correctness, transfer time calculation

### Multi-Node Tests (`tests/test_multinode.py`)
9 tests covering multi-node dispatch:
1. `test_single_node_backward_compat` — N=1 produces valid metrics, no dispatch tracking
2. `test_more_nodes_reduce_queue_pressure` — 4 nodes → lower per-node queue pressure than 1 node
3. `test_push_affinity_dispatches` — 4 nodes → affinity_dispatches > 0
4. `test_pull_no_starvation` — Pull mode, all 4 nodes get work
5. `test_pull_affinity_matches` — Pull mode processes requests, all nodes active
6. `test_queue_wait_metric` — Queue wait samples are populated and non-negative
7. `test_local_l3a_mode` — Local L3A mode runs and produces valid metrics
8. `test_global_vs_local_l3a_hit_rate` — Global L3A miss rate ≤ local L3A miss rate
9. `test_multinode_perf_benchmark` — 10s sim with 4 nodes completes in < 5s

## Debug History and Lessons Learned

### Bug: L2 occupancy not collapsing with TTL=0
**Symptom**: `test_zero_ttl_collapses_l2` failed — L2 showed non-zero occupancy.
**Root cause**: When `ttl_l2_s=0`, the TTL fire event was scheduled at exactly `sim_clock_us + 0`, but the object placement and TTL scheduling happened in the same event handler, so the object was briefly resident.
**Fix**: Added explicit check in `_place_kv_object()` — if `ttl_l2_s <= 0`, immediately hibernate the object from L2 to L3A after placement. Also added immediate hibernation path in `_on_ttl_fire()`.

### Bug: Infinite L1 still showed evictions
**Symptom**: `test_infinite_l1_no_evictions` failed with eviction count > 0.
**Root cause**: The eviction engine's `needs_l1_eviction()` was comparing against a threshold that could trigger even when the actual occupancy was well below 1%. The epoch report handler was calling `evict_l1_to_l2()` proactively.
**Fix**: Fixed the epoch report to only evict when `needs_l1_eviction()` returns true AND `used_bytes > target`.

### Bug: Sharing factor not returning 1.0 with zero shared prefix
**Symptom**: `test_no_shared_prefix_sharing_factor_one` failed.
**Root cause**: The sharing tracking in `_on_prefill_complete()` was counting tokens from prefix cache hits as "shared" even when `shared_system_prefix_tokens=0`. The contribution was calculated from `cached_tokens` regardless of whether it was a shared or session-private prefix.
**Fix**: Added guard `if profile.shared_system_prefix_tokens > 0 and cached > 0` before incrementing `tokens_served_from_shared_prefix`.

### Bug: BREAK_EVEN events with infinite bandwidth
**Symptom**: `test_zero_bandwidth_penalty_prefers_restore` failed with non-zero BREAK_EVEN count.
**Root cause**: `latency_floor_us` was not being zeroed out alongside bandwidth. Even with infinite bandwidth, the floor latency created a non-zero transfer cost that could make some restores break-even.
**Fix**: The test now sets both `bandwidth_bytes_per_s = sys.maxsize` AND `latency_floor_us = 0` for all tiers.

### Bug: Queue depth plot showed flat zero
**Symptom**: `queue_depth.png` showed both prefill and decode queues at 0 throughout the simulation.
**Root cause**: The engine queues pending prefills in `self._pending_prefills` (a deque), but `_on_epoch_report()` was reading `self.service.prefill_queue` which is never populated (the engine bypasses `ServiceModel.try_admit_prefill()`).
**Fix**: Changed epoch report to read `len(self._pending_prefills)` instead of `len(self.service.prefill_queue)`.

### Bug: L1 eviction metric conflated TTL migrations with pressure evictions
**Symptom**: L1 sensitivity plot showed eviction rate *increasing* with L1 capacity and then plateauing, which seemed counter-intuitive.
**Root cause**: `l1_to_l2_evictions` counted both TTL-driven tier demotions (time-based, constant rate) and occupancy-driven pressure evictions (capacity-dependent). The TTL-driven moves dominated, masking the capacity relationship.
**Fix**: Split into two metrics: `l1_to_l2_evictions` (pressure-driven only) and `l1_to_l2_ttl_migrations` (TTL-driven). The L1 sensitivity plot now shows both separately. The plateau behavior is correct — once L1 can accept objects, the throughput rate equals the arrival rate regardless of capacity.

### Bug: Short sims only generated batch traffic (diurnal rate = 0 at midnight)
**Symptom**: With v2 workload profiles, only `batch` produced KV objects. Chat, coding, agent, agentic_coding generated zero sessions.
**Root cause**: The NHPP diurnal rate model peaks at 9 AM (offset 32400s). At t=0 (midnight), all profiles with `diurnal_peak_trough_ratio=4.0` have rate=0. The first arrival for non-batch profiles was at ~22,400s (~6.2 hours) — well beyond the 60s sim duration. Batch survived because its ratio=1.5 (nearly flat).
**Fix**: Added `sim_start_time_s` config field (default: 0). Short sim configs now set `sim_start_time_s=36000` (10 AM) so all profiles have nonzero arrival rates. The offset is added to `time_s` in `WorkloadSynthesizer.diurnal_rate()`.

### Performance: Tests are slow (~10 min total)
Each invariant test runs a 60s simulation with 5s warmup. This is inherent to the DES approach — there's no way to skip the event processing. For faster iteration during development, reduce `sim_duration_s` and `warmup_s` further, but the invariant tests need enough events to be meaningful.

## Sanity Plots (`scripts/sanity_plots.py`)

Uses a "stressed" config: 500 MB L1, 10 GB L2, 50 GB L3A, short TTLs. This forces multi-tier activity and makes the plots informative. The default config has L1 = 80 GB which absorbs most objects without evictions, producing less interesting plots.

Generates 10 plots total:
1. `tier_occupancy.png` — tier fill levels over time
2. `ttft_distribution.png` — TTFT violin plots by hit source
3. `hit_rate_pie.png` — savings class breakdown
4. `queue_depth.png` — GPU queue utilization
5. `recompute_fraction.png` — per-request recompute distribution
6. `l1_sensitivity.png` — hit rate and eviction rate vs L1 capacity
7. `l2_sensitivity.png` — TTFT and hit rate vs L2 capacity
8. `node_scaling.png` — 6-panel: hit rate, eviction, queue wait, queue utilization, TTFT, L3A latency sensitivity (global vs local L3A)
9. `ttl_sensitivity.png` — hit rate and queue wait vs L2 TTL (global vs local L3A)
10. `sustaining_qps.png` — max QPS at SLA threshold vs node count (global vs local L3A)

## Metrics: Eviction Types

The simulator distinguishes two types of L1->L2 object movement:
- **Pressure evictions** (`l1_to_l2_evictions`): Forced out because L1 occupancy exceeded `eviction_hbm_threshold` or a new object needed space. These should decrease with larger L1.
- **TTL migrations** (`l1_to_l2_ttl_migrations`): Scheduled tier demotion when the `ttl_l2_s` timer expires. Rate is proportional to arrival rate, independent of L1 size.

## Savings Classes

- **WORTHWHILE**: Restoring cached KV from this tier is faster than recomputing from scratch. Transfer time < recompute time.
- **BREAK_EVEN**: Restoring from cache takes as long as (or longer than) recomputing. Only applies to L3A hits (SSD latency can exceed recompute cost for small prefills).
- L2 hits are always WORTHWHILE (DRAM bandwidth makes transfer negligible vs compute).

## Multi-Node Config Fields (`ServiceConfig`)

Added for multi-node prefill dispatch (all have backward-compatible defaults):

| Field | Default | Description |
|-------|---------|-------------|
| `n_prefill_nodes` | `1` | Number of prefill nodes (each with own L1/L2 and prefill slots) |
| `dispatch_algorithm` | `"push"` | `"push"` (affinity routing) or `"pull"` (global queue) |
| `inter_node_latency_us` | `5` | Base cross-node transfer latency |
| `inter_node_bandwidth_bytes_per_s` | `100_000_000_000` | 100 GB/s (NVLink) |
| `n_gpus_per_worker` | `8` | GPUs per worker sharing host DRAM (L2) and SSD (L3A) |
| `l3a_shared` | `True` | `True` = global shared L3A; `False` = per-worker local L3A |
| `l3a_remote_latency_us` | `50_000` | Additional latency for global L3A access (50ms) |

**Tier capacity semantics:**
- L1 capacity: **per-GPU** (each GPU has own HBM)
- L2 capacity: **per-worker** (shared by `n_gpus_per_worker` GPUs on the same host)
- L3A capacity: **global** when shared, **per-worker** when local
- `n_prefill_nodes` must be divisible by `n_gpus_per_worker` (or equal to 1)
- Existing `n_prefill_slots` and `prefill_queue_max` are **per-GPU**

**Transfer latency model:**
- Intra-worker L2 hit: no penalty (shared DRAM on same host)
- Intra-worker L1 hit from another GPU: small NVLink penalty (`inter_node_latency_us` only)
- Inter-worker hit: full `inter_node_latency_us` + bandwidth penalty

## Workload Profiles (v2)

The `coding` and `agentic_coding` profiles reflect real-world coding assistant workloads based on research into Claude Code, Cursor, and GitHub Copilot token usage:
- **coding**: 20k shared system prefix, 8k input/turn, 95→80% prefix stability. Models standard IDE coding assistants.
- **agentic_coding**: 30k shared system prefix, 15k input/turn, 95→85% prefix stability. Models heavy agent sessions (Claude Code agent, Cursor agent mode).

At 70B FP16, a 60k-token KV object is ~18.3 GB. L1 (80GB) fits only ~4 such objects, creating real eviction pressure.

Legacy v1 profiles (smaller token counts) are preserved in `configs/legacy_v1.json`.

## Config Tips

- `configs/default.json` is the v2 reference config. Don't modify it — load it and override fields programmatically.
- `configs/legacy_v1.json` has the original v1 profiles (smaller token counts) for backward compat.
- Tier order in `tiers[]` must be [L1, L2, L3A] — the engine indexes by position.
- `profile_mix` weights must sum to 1.0.
- `enable_suffix_cache` and `enable_l3b_object_store` are v2 features — setting them to `true` raises `NotImplementedError`.

## File Locations

- Benchmark tables: `benchmarks/latency_tables/`
- Configs: `configs/`
- Generated plots: `plots/`
- Sweep results: `results/` (gitignored)
