# Multi-Node Prefill Dispatch — Implementation Progress #1

Date: 2026-03-21

## Implementation Complete

### New files
- **`sim/node.py`** — `PrefillNode` class with per-node L1/L2 stores, prefill slots, pending queue, affinity checking, and queue pressure metrics
- **`sim/dispatch.py`** — `PushDispatcher` (cache-affinity-aware) and `PullDispatcher` (global queue with affinity scoring)
- **`tests/test_multinode.py`** — 6 new tests (all passing)

### Modified files
- **`sim/config.py`** — 4 new `ServiceConfig` fields: `n_prefill_nodes=1`, `dispatch_algorithm="push"`, `inter_node_latency_us=5`, `inter_node_bandwidth_bytes_per_s=100GB/s`
- **`sim/events.py`** — Added `NODE_PULL_CHECK` event type
- **`sim/engine.py`** — Full multi-node refactor: per-node L1/L2 stores, shared L3A, push/pull dispatch, cross-node transfer penalty, per-node eviction engines, aggregate+per-node epoch reporting. `_tier_stores` kept as backward-compat alias
- **`sim/metrics.py`** — Added `per_node_queue_depth`, `per_node_l1/l2_occupancy_pct`, `per_node_prefill_count`, `affinity_dispatches`, `non_affinity_dispatches`, `cross_node_transfers`
- **`scripts/sanity_plots.py`** — Added `plot_node_scaling()` (queue depth, TTFT p95, affinity rate vs node count)

### Test results
- **19/19 tests pass** (13 existing + 6 new) in ~103s
- Backward compatibility confirmed: `n_prefill_nodes=1` produces identical behavior
- 4-node push: affinity dispatches work, per-node queue pressure is lower than single-node
- 4-node pull: no starvation, all nodes get work
- Performance: 4-node 10s stressed sim completes in ~3.5s wall time

### Node-scaling plot observations
- TTFT p95 drops from 1002ms (1 node) to 337ms (8 nodes)
- Queue depth increases up to 4 nodes (4x more queue capacity) then drops at 8 nodes (enough throughput to drain)
- Affinity is highest at 2 nodes (45%) and drops as session state scatters across more nodes (14% at 8 nodes)
