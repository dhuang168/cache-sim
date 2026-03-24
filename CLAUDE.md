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
- **sim/events.py** — Event types and request FSM. Colocated FSM: QUEUED -> CACHE_LOOKUP -> HIT/MISS -> PREFILLING -> DECODE_QUEUED -> DECODING -> KV_WRITE -> COMPLETE. Disaggregated FSM adds: PREFILLING -> KV_TRANSFERRING -> DECODE_QUEUED.
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

### Config Consistency Tests (`tests/test_config_consistency.py`)
8 regression tests for bugs found during development:
1. `test_oracle_covers_workload_range` — oracle table covers max expected context per profile
2. `test_oracle_monotonically_increasing` — oracle(n+1) ≥ oracle(n)
3. `test_l2_occupancy_never_exceeds_100` — L2 never exceeds capacity
4. `test_all_profiles_generate_sessions` — every profile with mix > 0 creates sessions
5. `test_session_proportions_match_profile_mix` — proportions within ±15%
6. `test_queue_depth_nonzero_under_load` — queue metric is wired correctly
7. `test_ttl_migrations_separate_from_pressure` — eviction counters are independent
8. `test_no_global_l3a_penalty_single_worker` — no remote penalty with 1 worker

### Block Sharing Tests (`tests/test_sharing.py`)
10 tests for cross-session prefix sharing:
- Sharing config defaults, tier stacking, ref count increment
- Memory saved metric, shared block groups, different profiles don't share
- Backward compatibility (sharing disabled), cross-worker duplication
- L3A bandwidth contention tracking, sharing reduces occupancy

### Block Allocation Tests (`tests/test_blocks.py`)
21 tests for token block math and cache hit granularity:
- Block count at 16/256/4096 token sizes, exact fit, legacy mode
- Block boundary rounding (cached tokens rounded to full blocks)
- Fragmentation (wasted bytes per block size)
- Integration sims at different block sizes
- Granularity tradeoff: larger blocks → higher recompute fraction

### Multi-Node Tests (`tests/test_multinode.py`)
14 tests covering multi-node dispatch, worker topology, and L3A isolation:
1. `test_single_node_backward_compat` — N=1 produces valid metrics, no dispatch tracking
2. `test_more_nodes_reduce_queue_pressure` — 4 nodes → lower per-node queue pressure than 1 node
3. `test_push_affinity_dispatches` — 4 nodes → affinity_dispatches > 0
4. `test_pull_no_starvation` — Pull mode, all 4 nodes get work
5. `test_pull_affinity_matches` — Pull mode processes requests, all nodes active
6. `test_queue_wait_metric` — Queue wait samples are populated and non-negative
7. `test_local_l3a_mode` — Local L3A mode runs and produces valid metrics
8. `test_global_vs_local_l3a_hit_rate` — Global L3A miss rate ≤ local L3A miss rate
9. `test_multinode_perf_benchmark` — 10s sim with 4 nodes completes in < 5s
10. `test_worker_topology_shared_l2` — 8 GPUs / 2 workers verify L2 sharing and worker_id
11. `test_intra_worker_no_l2_penalty` — same_worker() returns True/False correctly
12. `test_local_l3a_worker_isolation` — object on worker 0 NOT visible from worker 1
13. `test_global_l3a_cross_worker_access` — global L3A visible from any node
14. `test_session_migration_global_advantage` — tiny L1/L2 + 2 workers → global > local hit rate

### Chunk Store Unit Tests (`tests/test_chunk_store.py`)
10 tests for ChunkTierStore and ChunkIndex:
- novel_insert, dedup_insert, deref_removes, deref_keeps, capacity_check
- evict_lru_prefers_single_ref, occupancy_pct
- chunk_index_consecutive_lookup, chunk_index_gap_stops_consecutive, chunk_hash_for

### Chunk Dedup Integration Tests (`tests/test_chunk_dedup.py`)
11 tests for chunk-level deduplication and demand-pull promotion:
1. `test_per_session_dedup_matches_default` — per_session mode produces standard results
2. `test_chunk_requires_block_size` — chunk mode validates block_size_tokens > 0
3. `test_demand_pull_requires_lru` — demand_pull validates eviction_policy=lru
4. `test_chunk_dedup_ratio_positive` — shared prefixes produce >10% dedup ratio
5. `test_chunk_dedup_reduces_storage` — chunk mode deduplicates shared prefix chunks
6. `test_chunk_consecutive_lookup_hit` — chunk lookup finds cached chunks
7. `test_lmcache_config_loads_and_runs` — configs/lmcache.json works end-to-end
8. `test_demand_pull_promotes_on_hit` — demand-pull promotes chunks from lower tiers
9. `test_demand_pull_no_ttl_scheduling` — no TTL migrations in demand_pull mode
10. `test_demand_pull_object_mode` — demand-pull works with per-session objects too
11. `test_chunk_mode_perf_benchmark` — 10s chunk sim completes in <15s wall-clock

### Disaggregated P/D Tests (`tests/test_disaggregated.py`)
10 tests for disaggregated prefill-decode separation:
1. `test_backward_compat_disaggregated_false` — explicit disaggregated=false matches default
2. `test_kv_transfer_events_fire` — 3P:1D produces KV transfers with positive latencies
3. `test_prefill_not_blocked_by_decode` — prefill slots freed immediately, no decode backpressure
4. `test_decode_node_utilization` — decode nodes have active sequences under load
5. `test_kv_transfer_size_matches_model` — transfer bytes = kv_size_bytes(total_tokens)
6. `test_kv_write_targets_prefill_node` — KV written to prefill node L1, not decode node
7. `test_prefill_multiplier_effect` — multiplier=0.5 halves prefill duration
8. `test_transfer_bandwidth_sensitivity` — low BW → higher transfer times
9. `test_decode_backpressure` — 1 decode slot, heavy load → requests queue
10. `test_disaggregated_perf_benchmark` — 10s sim completes in <10s wall-clock

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

### Bug: Oracle table clamped at 32k tokens — inverted cache hit economics
**Symptom**: Global L3A cache hits appeared SLOWER than cold misses. Controlled load tests showed local L3A with BETTER TTFT despite 73% cold miss rate.
**Root cause**: Prefill oracle table only covered 512-32,768 tokens. `np.interp` clamped anything above 32k to 17s. With v2 coding workloads (50-100k tokens), cold misses cost 17s (clamped) while L3A cache hits cost transfer_time (2-4s) + partial recompute (5-10s) = 7-14s. But for agentic_coding at 65k+ tokens, transfer + partial exceeded the clamped 17s.
**Fix**: Extended oracle table to 262,144 tokens with O(n²) attention scaling extrapolation: 65k=55s, 131k=180s, 262k=600s. Now cold miss on 95k tokens = 111s, cache hit = 13s (88% savings).

### Bug: Local L3A searched all workers' SSDs (should only search own worker)
**Symptom**: Global and local L3A showed identical hit rates even at 20 min — but sessions were migrating (97% non-affinity dispatch).
**Root cause**: `_find_cache_object_with_node()` in local L3A mode searched ALL workers' L3A stores, effectively giving local mode the same cross-worker visibility as global mode.
**Fix**: In local mode, only search the requesting node's worker L3A via `from_node_id` parameter. Fallback to all-worker search only for global lookups (trie).
**Impact**: At 20 min heavy coding, local L3A drops from 99.8% to 26.6% hit rate — revealing the true value of global L3A for session migration.

### Bug: L2 saturation exceeded 100% (eviction bypassed capacity check)
**Symptom**: L2 tier occupancy reported as 154% — `used_bytes > capacity_bytes`.
**Root cause**: `EvictionEngine.evict_l1_to_l2()` inserted objects into L2 without checking `can_fit()`. When L1 pressure eviction fired and L2 was already full, objects were inserted unconditionally, exceeding capacity.
**Fix**: Added `can_fit()` check before L2 insert in `evict_l1_to_l2()`. When L2 is full, objects are hibernated directly to L3A via new `hibernate_l2_to_l3a_obj()` method.

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
- L3A capacity: **per-worker SSD** — Global mode pools all workers: `total = per_worker × n_workers`. Local mode: each worker uses its own SSD.
- `n_prefill_nodes` must be divisible by `n_gpus_per_worker` (or equal to 1)
- Existing `n_prefill_slots` and `prefill_queue_max` are **per-GPU**

**Transfer latency model:**
- Intra-worker L2 hit: no penalty (shared DRAM on same host)
- Intra-worker L1 hit from another GPU: small NVLink penalty (`inter_node_latency_us` only)
- Inter-worker hit: full `inter_node_latency_us` + bandwidth penalty

## LMCache-Compatible Chunk Dedup Mode

When `cache.deduplication="chunk"`, KV is stored at **fixed-size chunk granularity** (default 256 tokens) with **hash-based deduplication**. Identical chunks across sessions share storage via ref counting.

- **Shared prefix chunks** hash by profile: `chunk-{profile}-{index}` — deduped across all sessions of the same profile.
- **Session-unique chunks** hash by session: `chunk-{session}-{index}` — unique per session.
- **Demand-pull promotion** (`cache.tier_migration="demand_pull"`): on cache hit at L2/L3A, promote chunks to L1. No TTL scheduling. Pure LRU eviction.
- **Cascade eviction**: L1 evictions cascade to L2, L2 evictions cascade to L3A.

**New `CacheConfig` Fields:**

| Field | Default | Options |
|-------|---------|---------|
| `deduplication` | `"per_session"` | `"per_session"` (monolithic objects) or `"chunk"` (LMCache-style) |
| `tier_migration` | `"ttl_push"` | `"ttl_push"` (scheduled demotion) or `"demand_pull"` (promote on hit) |
| `chunk_eviction` | `"lru"` | `"lru"` (LMCache: position-unaware) or `"tail_first"` (vLLM: evict tail chunks first, preserve prefix) |

**Constraints:**
- `deduplication="chunk"` requires `block_size_tokens > 0`
- `tier_migration="demand_pull"` requires `eviction_policy="lru"`

**Key data structures** (`sim/chunk_store.py`):
- `ChunkObject` — per-chunk metadata with ref counting
- `ChunkTierStore` — dedup-aware store with `insert_or_ref`, `deref`, LRU eviction
- `ChunkIndex` — tracks cached chunks per session for consecutive lookup

**Reference config:** `configs/lmcache.json` — chunk dedup + demand-pull + LRU (full LMCache-style).

## Disaggregated Prefill-Decode Mode

When `service.disaggregated=true`, prefill and decode run on **separate GPU pools**:
- **Prefill nodes** (`n_prefill_nodes`): Handle prefill only. Prefill slots freed immediately on completion — no backpressure from decode.
- **Decode nodes** (`n_decode_nodes`): Handle decode only. Lightweight — no L1/L2/L3A cache stores. KV cache is written back to the originating prefill node after decode completes.
- **KV transfer**: After prefill completes, KV data is transferred from prefill GPU to decode GPU via RDMA. This adds a `KV_TRANSFERRING` FSM state between PREFILLING and DECODE_QUEUED.

**Disaggregated Config Fields (`ServiceConfig`)**

| Field | Default | Description |
|-------|---------|-------------|
| `disaggregated` | `False` | Enable prefill-decode separation |
| `n_decode_nodes` | `0` | Number of dedicated decode GPUs (0 = colocated mode) |
| `kv_transfer_bandwidth_bytes_per_s` | `50_000_000_000` | 50 GB/s RDMA for P→D KV transfer |
| `kv_transfer_latency_floor_us` | `2000` | 2ms fixed transfer overhead |
| `prefill_latency_multiplier` | `1.0` | <1.0 = speedup from no decode interference |
| `decode_batch_fill_factor` | `0.7` | Average decode batch utilization (higher in disaggregated) |

**Disaggregated request flow:**
```
ARRIVAL → CACHE_LOOKUP → [HIT/MISS] → PREFILLING (prefill node)
  → KV_TRANSFERRING (RDMA to decode node) → DECODE_QUEUED → DECODING (decode node)
  → KV_WRITE (back to prefill node L1) → COMPLETE
```

**Reference config:** `configs/disaggregated.json` — 3:1 prefill:decode ratio (3 prefill GPUs, 1 decode GPU).

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

# Development Methodology

## Workflow

Every non-trivial task follows this sequence:

1. **Research**: When empirical data is needed, research first. Save findings to `plan_and_progress/research_*.md` before proposing designs.
2. **Plan**: Answer questions and reason through the problem in text. Save plan to `plan_and_progress/<feature>_plan_<N>.md`. Get alignment before coding.
3. **Implement**: Code + tests + documentation in each phase. Tests must use extreme conditions that force the feature to be the sole differentiator.
4. **Verify**: Traffic/workload mix validation. Config consistency check (when changing parameter A, verify all dependent parameters B still valid). Crosscheck plan vs deliverables line-by-line.
5. **Report**: Create NEW report in `docs/` (never overwrite existing reports). Cross-check every claim against actual data.
6. **Ship**: Commit ALL artifacts (code, tests, plots, configs, docs, reports, plans). Push.

## Core Practices

### Observe → Question → Investigate → Improve
When results are unexpected or two configurations that should differ are identical:
1. Question the result (does it match physical intuition?)
2. Add instrumentation (metrics to expose the hidden mechanism)
3. Trace the data path (follow a request through the code)
4. Write an isolation test (expose the bug before fixing)
5. Fix and validate with data
6. Add regression test (extreme conditions that catch reintroduction)

### Performance Gate: Profile Before Scale
Before any sim longer than 2 min or wider than 8 GPUs with new code:
1. Run a 30-second sim with the feature enabled
2. Profile with `cProfile` — check top 10 by `tottime`
3. Verify no function is O(n²) in the hot path
4. If any function takes >10% of total time and scales with data, optimize first
5. Gate: 2-min sim at target scale must complete in <30s wall-clock

**Lesson**: Chunk store `evict_lru()` sorted the entire dict per insertion — O(n log n) × 390 chunks/request = 40+ min for a 10-min sim. A 30-second profile would have caught it instantly.

### Validate Hypotheses With Metrics
Never state a hypothesis as fact. Add metrics → measure → conclude. Example: "50ms latency causes feedback loop" → add prefill_duration, slot_utilization, cold_evictions_per_epoch → measure correlation → confirm or refute.

### Config Consistency
When changing any parameter, walk the dependency map:
- Changing workload sizes → check oracle/model covers the range
- Changing arrival rates → check system not in severe overload (>50% drops = meaningless metrics)
- Changing topology → check capacity semantics (per-node vs global)
- Changing sim duration → check steady state reached

### Test Design: Extreme Conditions
For every new feature:
- What metric proves it works?
- What extreme condition makes it the ONLY differentiator?
- What isolation test proves the boundary?
- What end-to-end test proves the value?

### Experiment Design
- Controlled load (below capacity to avoid drops)
- One variable at a time
- Multiple durations (short sims may be misleading)
- Seed sensitivity at capacity edges (run 3+ seeds)
- Always report: hit rate AND completed requests AND drop rate

### Reports
- Lead with key finding, not methodology
- Never overwrite existing reports — create new versioned ones
- Cross-check every claim against actual metric values
- Note caveats (sim duration, seed sensitivity, load regime)

### Documentation
Update with every implementation phase:
- Architecture doc (how it works end-to-end)
- User manual (config reference, recipes, caveats)
- Debug history (every bug and its fix)
- Comparison docs (alignment with production systems)

### Batch Operations
Combine multiple tests/commands into single invocations to minimize friction. Smoke test before long runs — profile a quick run to estimate total time.
