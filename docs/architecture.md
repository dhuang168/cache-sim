# Architecture: How the Simulation Works

This document explains the end-to-end flow of the simulator — how requests are generated, what assumptions are made, how cache hits are determined, and how results map to real-world metrics.

## 1. Time Model

The simulator is a **discrete-event simulator** (DES). Time advances by jumping to the next scheduled event, not by ticking a clock.

- **Sim clock**: `int64` microseconds. No floating-point drift.
- **Events**: stored in a min-heap, processed in time order.
- **Sim duration**: configurable (e.g., 60s for quick tests, 1200s for steady-state analysis).
- **Warmup period**: metrics are not collected until `warmup_s` has elapsed. This avoids transient startup effects.
- **Diurnal offset** (`sim_start_time_s`): the NHPP arrival rate is sinusoidal with a 24-hour period, peaking at 9 AM. Setting `sim_start_time_s=36000` (10 AM) places the sim in peak business hours. At midnight (default 0), most profiles have zero arrival rate.
- **Epoch reports**: metrics are sampled at regular intervals (`epoch_report_interval_s`). These produce the time-series data for occupancy, queue depth, and slot utilization plots.

**Steady-state caveat**: Short sims (60s) show early transient behavior. L2 saturates at ~5 min, session migration effects appear at ~5 min, and the full global vs local L3A difference requires 10-20 min. Always match sim duration to the phenomenon you're studying.

## 2. Request Generation

### Session Creation

Each workload profile has an independent **Non-Homogeneous Poisson Process** (NHPP) that generates new sessions. The rate is:

```
effective_rate(t) = arrival_rate_peak × profile_mix[name] × diurnal_factor(t)
```

Where `diurnal_factor` is sinusoidal with period 86400s, peaking at 9 AM. The thinning algorithm generates arrival times by proposing exponential candidates and accepting with probability `rate(t) / rate_max`.

Each new session gets:
- A unique `session_id`
- A `SessionState` tracking context growth and turn count
- A session-private `PrefixTrie` for cache matching
- Initial context = `shared_system_prefix_tokens` (e.g., 20k for coding)
- A random session duration from the configured distribution

### Within-Session Requests

After the first request, subsequent requests arrive at intervals drawn from `iat_mean_s` (exponential or lognormal). Each request:
1. Grows the session context by `context_growth_min_tokens` to `context_growth_max_tokens` (random)
2. Samples `input_len` tokens (new user input) and `output_len` tokens (model output)
3. Schedules a `PREFILL_START` event immediately

### Multi-User Model

Each session = one user's conversation. Sessions are **independent**:
- Each has its own KV cache objects keyed by session_id
- KV objects are **not shared** across users (even with identical prompts)
- The only cross-session sharing is the `shared_system_prefix_tokens` mechanism, which avoids redundant prefix **recomputation** during prefill (but each session still stores its own KV in the cache)

## 3. Request Lifecycle

```
ARRIVAL → PREFILL_START → PREFILL_COMPLETE → DECODE_START → DECODE_COMPLETE → KV_WRITE → COMPLETE
```

### Step 1: Arrival (`_on_arrival`)

- Creates a unique `request_id`
- Grows session context
- Samples input/output token counts
- Schedules `PREFILL_START` immediately (same sim clock)

### Step 2: Prefill Start (`_on_prefill_start`)

This is the most complex step. It performs:

**a) Dispatch (multi-node):**
- **Push mode**: Dispatcher picks a node based on cache affinity (session's KV in node's L1/L2). Falls back to least-loaded node.
- **Pull mode**: Request enters a global queue. Idle nodes pull with affinity scoring.

**b) Cache lookup:**
- Check shared prefix trie and session trie for a matching KV object
- Search order: all nodes' L1 → all nodes' L2 → L3A (global: shared pool; local: **own worker's SSD only**)
- Returns: hit tier (L1/L2/L3A) or MISS, and the node_id where the object was found

**c) Compute prefill time:**
- **Prefix stability** determines what fraction of context is cached vs must be recomputed:
  ```
  cached_tokens = total_context × stability(turn)
  uncached_tokens = total_context × (1 - stability) + input_tokens
  ```
  Even with a cache hit, `input_tokens` (new user message) is always uncached.

- **Prefill duration** depends on hit type:
  | Hit Type | prefill_us formula |
  |----------|-------------------|
  | L1 hit | `oracle(uncached_tokens)` |
  | L2 hit | `transfer_time(kv_bytes, L2) + oracle(uncached_tokens)` |
  | L3A hit (global) | `transfer_time(kv_bytes, L3A) + 50ms_remote + oracle(uncached_tokens)` |
  | Cold miss | `oracle(total_context + input_tokens)` — full recompute |

  Where `oracle(n)` is piecewise-linear interpolation from the benchmark table (512-262k tokens, O(n²) scaling).

  Where `transfer_time = latency_floor + size_bytes / bandwidth`.

- **Cross-node penalty**: If a cache hit is on a different GPU:
  - Same worker: NVLink base latency only (L2 hit on same worker = zero penalty, it's shared DRAM)
  - Different worker: full `inter_node_latency_us` + bandwidth-based transfer

**d) Slot admission:**
- If the assigned node has a free prefill slot → schedule `PREFILL_COMPLETE` after `prefill_us`
- If no slot but queue has space → enqueue (records `queue_wait` start time)
- If queue is full → **request is dropped** (backpressure). Dropped requests are counted in `savings_events` but have no queue_wait or TTFT recorded.

### Step 3: Prefill Complete (`_on_prefill_complete`)

- Records **TTFT** = `sim_clock - request_arrival_time`. This includes queue wait + prefill compute.
- Records **prefill_duration** = the `prefill_us` from the payload (compute only, no queue wait).
- Frees the prefill slot, drains pending queue.
- Hands off to decode (if decode slot available) or queues for decode.

### Step 4: Decode (`_on_decode_start`, `_on_decode_complete`)

- Decode latency = `output_tokens × base_latency × sqrt(active_sequences)` (batch degradation model)
- After decode: write KV to cache.

### Step 5: KV Write (`_on_decode_complete` → `_place_kv_object`)

Places the session's KV object into the assigned node's cache:
1. Try node's L1 (HBM). If full, evict to L2 (pressure eviction).
2. If L1 can't fit → try node's L2 (DRAM). If full, hibernate to L3A.
3. If L2 can't fit → try L3A (per-worker SSD or global pool).
4. Schedule TTL events for tier migration.

**Placement is per-session**: each user's KV goes into the cache independently. Two users with identical 20k system prompts create two separate 20k-token KV objects.

## 4. Cache Tier Movement

Objects flow downward through tiers over time:

```
L1 (HBM) ──TTL/pressure──► L2 (DRAM) ──TTL/pressure──► L3A (SSD)
   80 GB/GPU                  1 TB/worker                 8 TB/worker
```

### TTL-driven migration
- `ttl_l2_s`: After this many seconds without access, L1 objects move to L2 (node-local)
- `ttl_l3a_s`: After this many seconds, L2 objects hibernate to L3A

### Pressure-driven eviction
- When L1 occupancy exceeds `eviction_hbm_threshold`, evict oldest/least-referenced objects to L2
- L2 overflow goes to L3A. If L2 is too full, objects bypass L2 and go directly to L3A.
- L3A overflow → cold eviction (object permanently lost)

### Worker topology
- L1 → L2 eviction stays within the same worker (L2 is shared by 8 GPUs on the host)
- L2 → L3A hibernation goes to the worker's own SSD (local mode) or the global pool (global mode)
- **Affinity check only sees L1/L2**: once an object is in L3A, the dispatch affinity check can't find it. This is the root cause of session migration.

## 5. Session Migration

This is the most important dynamic in multi-worker deployments:

1. Session S is dispatched to Worker 0, GPU 3. KV placed in GPU 3's L1.
2. After `ttl_l2_s`, KV moves to Worker 0's shared L2.
3. After `ttl_l3a_s` or L2 pressure, KV moves to Worker 0's L3A (SSD).
4. Session S sends another request. Dispatch checks L1/L2 for affinity. **KV is in L3A — no affinity found.**
5. Request dispatched to Worker 1, GPU 5 (least loaded).
6. Cache lookup searches Worker 1's L3A (local mode) → **not found → cold miss**.
   Or searches global L3A pool → **found → cache hit**.
7. Cold miss: full recompute (37-120s for coding contexts). Cache hit: transfer + partial (5-15s).

**At steady state, 97-99% of dispatches are non-affinity.** This is why global L3A is essential for multi-worker coding deployments.

## 6. Metrics: What Measures What

### TTFT (Time-To-First-Token)
`TTFT = sim_clock_at_prefill_complete - sim_clock_at_request_arrival`

Includes: queue wait + prefill compute (transfer + oracle).
Does NOT include: decode time (that's a separate phase).
Recorded per cache-hit source (L1_hit, L2_hit, L3A_hit, cold_miss).

### Queue Wait
`queue_wait = sim_clock_at_slot_obtained - sim_clock_at_queue_entry`

Only recorded for requests that get a slot. **Dropped requests have no queue_wait** — if >50% are dropped, the queue_wait distribution has survivorship bias.

### Prefill Duration
`prefill_duration = prefill_us from _on_prefill_start payload`

This is the **compute-only** time: transfer + oracle(uncached_tokens). Does NOT include queue wait. To understand the full request cost, use TTFT (which includes both).

### Slot Utilization
`slot_utilization = busy_slots / total_slots` at each epoch.

Both global and local L3A modes show ~100% utilization under load. The difference is in **what each slot computes**: cache hit (fast, useful) vs cold miss (slow, wasteful).

### Hit Rate
`hit_rate = 1 - COLD_MISS / total_savings_events`

Only counted for requests that reach `_on_prefill_start`. Dropped requests (backpressure) are included in the denominator — they get classified as a savings event based on their cache lookup result, even if they never complete.

### Dropped Requests
Not directly in `MetricsCollector`. Infer from: `total_savings_events - len(queue_wait_us)` = approximate drops. If this exceeds 50%, the system is in severe overload and per-request metrics are unreliable.

## 7. Assumptions and Limitations

| Assumption | Impact |
|-----------|--------|
| **KV cache is per-session** | No cross-user KV sharing (each user has own objects). Real systems may deduplicate shared prefixes at the KV level. |
| **Prefill is single-sequence** | No batched prefill. Real systems batch multiple requests for GPU efficiency. |
| **Oracle is piecewise-linear** | Prefill latency interpolated from benchmark points. 64k+ tokens extrapolated with O(n²) scaling. |
| **Dispatch affinity only checks L1/L2** | Once objects reach L3A, affinity is lost → sessions migrate. A real system might maintain L3A-aware routing. |
| **Decode is simplified** | sqrt(batch_size) degradation model. Real decode is more complex (KV cache memory pressure, scheduling). |
| **No network topology** | Cross-worker transfers use a flat latency model. Real clusters have NUMA, switch hierarchy, etc. |
| **Session arrivals are independent** | No correlation between sessions. Real workloads may have bursty patterns. |
