# Multi-Node Prefill Dispatch — Progress #2: Global L3A Analysis Infrastructure

Date: 2026-03-21

## Phase 1 Complete

### 1.1 Queue wait time metric
- Added `queue_wait_us: list[int]` to `MetricsCollector` — tracks time from entering pending queue to slot obtained
- Requests that get immediate slots record queue_wait=0
- Report includes `queue_wait_ms` with p50/p95/p99/mean
- Implemented in both push and pull drain paths

### 1.2 Local vs Global L3A mode
- New `ServiceConfig` fields: `l3a_shared: bool = True`, `l3a_remote_latency_us: int = 50_000`
- When `l3a_shared=False`: each node gets its own L3A store with `capacity/n_nodes`
- When `l3a_shared=True`: shared L3A with configurable remote access latency penalty
- Engine handles both modes for: cache lookup, KV placement, eviction, TTL, epoch reporting

### 1.3 Sustaining QPS helper
- New `sim/analysis.py` with `find_sustaining_qps()` — binary search for max arrival rate multiplier at SLA threshold

### New files
- `sim/analysis.py` — sustaining QPS analysis helper

### Modified files
- `sim/config.py` — 2 new fields (`l3a_shared`, `l3a_remote_latency_us`)
- `sim/node.py` — optional `l3a_store` parameter
- `sim/engine.py` — local/global L3A paths, queue wait tracking, `_get_l3a_for_node()` helper
- `sim/metrics.py` — `queue_wait_us` field, queue wait stats in report
- `tests/test_multinode.py` — 3 new tests (queue_wait_metric, local_l3a_mode, global_vs_local_l3a_hit_rate)
- `scripts/sanity_plots.py` — 6-panel node_scaling plot with global vs local comparison + L3A latency sensitivity

### Test results
- **22/22 tests pass** (13 existing + 9 multinode)
- Queue wait metric working: p95 ~700ms for 4-node stressed, ~7-23ms for 8-node
- Local vs Global L3A both functional

### Phase 2 plot observations
- Phase transition at 8 nodes: queue wait drops from ~700ms to ~7-23ms (throughput > arrival rate)
- Current stressed config has 100% hit rate — L2 absorbs everything. Need a more cache-hostile config to see global vs local L3A miss rate differences.
- L3A latency sensitivity: TTFT rises ~70ms across 0→100ms remote latency range at 4 nodes
- TTL sensitivity (panel c) and sustaining QPS (panel e) not yet plotted — need arrival rate sweeps

### Next steps
- Create a more cache-hostile config (tiny L2, aggressive TTLs) to expose global vs local L3A hit rate gap
- Add TTL sensitivity panel
- Add sustaining QPS panel using `find_sustaining_qps()`
