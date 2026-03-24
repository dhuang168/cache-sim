# Multi-Node Prefill Dispatch

## Context

The queue depth plot shows the prefill queue saturated at 128 (max) throughout the simulation — the single-node 32-slot prefill pool can't keep up with arrival rate. The user wants to scale by adding multiple prefill nodes, each with its own L1/L2 cache and prefill slots, with two dispatch algorithms: push (first) and pull (second).

## Architecture

```
                    ┌──────────────────────┐
                    │   Central Dispatcher  │
                    │  (push or pull mode)  │
                    └──────┬───────────────┘
                           │
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
        ┌──────────┐ ┌──────────┐    ┌──────────┐
        │  Node 0  │ │  Node 1  │    │  Node N  │
        │ L1 (HBM) │ │ L1 (HBM) │    │ L1 (HBM) │
        │ L2 (DRAM)│ │ L2 (DRAM)│    │ L2 (DRAM)│
        │ 32 slots │ │ 32 slots │    │ 32 slots │
        │ local Q  │ │ local Q  │    │ local Q  │
        └────┬─────┘ └────┬─────┘    └────┬─────┘
             └─────────────┼───────────────┘
                           ▼
              ┌─────────────────────────┐
              │  Shared L3A (SSD)       │
              │  Shared Decode Pool     │
              └─────────────────────────┘
```

- Each node owns a **local L1 TierStore** (HBM — physically per-GPU)
- Each node owns a **local L2 TierStore** (DRAM — node-local host memory)
- **Only L3A is shared** across nodes (SSD/networked storage, accessible by all)
- Each node has its own **prefill slot count** and **local pending queue**
- **Decode remains a single shared pool** (unchanged)
- A cache hit on a **remote node's L1 or L2** incurs configurable inter-node transfer latency
- L1→L2 eviction stays **node-local** (L1 objects move to that node's own L2)
- L2→L3A hibernation moves from a node's local L2 to the **shared L3A**

## Config Changes (`sim/config.py`)

Add 4 fields to `ServiceConfig` with backward-compatible defaults:

```python
n_prefill_nodes: int = 1                              # number of prefill nodes
dispatch_algorithm: str = "push"                      # "push" or "pull"
inter_node_latency_us: int = 5                        # base cross-node latency
inter_node_bandwidth_bytes_per_s: int = 100_000_000_000  # 100 GB/s (NVLink)
```

Existing fields reinterpreted as **per-node**: `n_prefill_slots`, `prefill_queue_max`.

Tier config in `tiers[]` reinterpreted: L1 and L2 capacities are **per-node**, L3A capacity is **global shared**.

## New File: `sim/node.py` — PrefillNode

Lightweight class owning per-node state:
- `node_id: int`
- `l1_store: TierStore` — node-local L1 (HBM)
- `l2_store: TierStore` — node-local L2 (DRAM)
- `eviction: EvictionEngine` — per-node (L1→L2 local, L2→L3A to shared)
- `prefill_slots_free: int` — local slot counter
- `pending_prefills: deque` — local queue
- `prefill_queue_max: int`
- `active_completions: list[int]` — sorted completion times for in-flight prefills

Key methods:
- `has_session_cached(session_id) -> bool` — checks if session KV is in this node's L1 or L2
- `projected_free_time_us(current_us) -> int` — earliest slot availability
- `queue_pressure() -> float` — `len(pending) / queue_max`

## New File: `sim/dispatch.py` — Push & Pull Dispatchers

### Push Dispatcher (implement first)

```
dispatch(session_id, request_id, payload, current_us) -> PrefillNode:
    1. Find nodes with cache affinity (session's KV in node's L1 or L2)
    2. If affinity nodes exist and not overloaded (queue_pressure < 0.9):
       → pick the one with earliest projected_free_time_us
    3. Otherwise: pick node with earliest projected_free_time_us
       → tiebreak by queue depth
```

### Pull Dispatcher (implement second)

Uses a **global queue** instead of per-node queues. Nodes pull when slots free up.

```
enqueue(session_id, request_id, payload, arrival_us):
    → append to global queue

pull(node, current_us) -> job or None:
    1. Scan global queue for affinity match (session in node's L1 or L2)
    2. Score = affinity_bonus (large) - age_us (older = higher priority)
    3. Return best-scoring job, or oldest if no affinity match
```

## Engine Changes (`sim/engine.py`)

### `__init__`:
- Create `self.nodes: list[PrefillNode]` (N nodes, each with own L1, L2, eviction engine)
- Create `self._shared_l3a: TierStore` for the shared L3A tier
- Create `self.dispatcher: PushDispatcher | PullDispatcher`
- Single-node (N=1): one node, trivial dispatch — identical to current behavior
- Keep `self._tier_stores` as backward-compat alias for single-node: `{L1: node[0].l1, L2: node[0].l2, L3A: shared_l3a}`

### `_on_prefill_start`:
- **Push mode**: call `dispatcher.dispatch()` to pick a node, store `node_id` in payload
- **Pull mode**: call `dispatcher.enqueue()`, then trigger pull on nodes with free slots
- Cache lookup: check assigned node's L1/L2 first (free), then other nodes' L1/L2 (with inter-node transfer penalty), then shared L3A

### `_on_prefill_complete`:
- Free slot on the specific node (`payload["node_id"]`)
- **Push**: drain that node's pending queue
- **Pull**: trigger pull from global queue

### `_place_kv_object`:
- Accept `node_id` param, target that node's L1 first, then that node's L2
- Fallthrough to shared L3A
- L1→L2 eviction stays within the same node
- L2→L3A hibernation moves to shared L3A

### `_find_cache_object`:
- Search all node L1s, all node L2s, then shared L3A
- New variant `_find_cache_object_with_node()` returns `(obj, node_id_or_none)` for dispatch decisions (node_id for L1/L2 hits, None for L3A)

### `_on_ttl_fire`:
- L1→L2 moves stay within the same node (use `payload["node_id"]` to find the right node)
- L2→L3A hibernation: move from node's L2 to shared L3A

### `_on_epoch_report`:
- Record per-node L1/L2 occupancy and queue depths
- Aggregate L1 and L2 across nodes for backward-compatible `tier_occupancy_pct["L1"]` / `["L2"]`

## Metrics Changes (`sim/metrics.py`)

New fields:
- `per_node_queue_depth: dict[int, list[int]]` — per-node prefill queue time series
- `per_node_l1_occupancy_pct: dict[int, list[float]]` — per-node L1 saturation
- `per_node_l2_occupancy_pct: dict[int, list[float]]` — per-node L2 saturation
- `per_node_prefill_count: dict[int, int]` — total prefills per node
- `affinity_dispatches: int` — dispatched to node with local cache hit
- `non_affinity_dispatches: int` — dispatched without affinity
- `cross_node_transfers: int` — remote L1/L2 hits requiring inter-node transfer

## Events Changes (`sim/events.py`)

Add `NODE_PULL_CHECK = auto()` for pull dispatcher polling.
No FSM changes — remote L1/L2 hits modeled as L1/L2 hits with added transfer latency.

## Implementation Sequence

1. `sim/config.py` — Add 4 new fields to `ServiceConfig`
2. `sim/node.py` — New file: `PrefillNode` class
3. `sim/dispatch.py` — New file: `PushDispatcher` + `PullDispatcher`
4. `sim/events.py` — Add `NODE_PULL_CHECK`
5. `sim/engine.py` — Refactor to multi-node (biggest change)
6. `sim/metrics.py` — Add per-node and dispatch metrics
7. `configs/default.json` — Add new service fields (optional, defaults work)
8. `tests/test_multinode.py` — New test file
9. `scripts/sanity_plots.py` — Add node-scaling plot

## Test Plan (`tests/test_multinode.py`)

| Test | What It Proves |
|------|---------------|
| `test_single_node_backward_compat` | N=1 produces same metrics as current code |
| `test_more_nodes_reduce_queue_depth` | 4 nodes → lower mean queue depth than 1 node |
| `test_push_affinity_dispatches` | Multi-turn sessions with 4 nodes → affinity_dispatches > 0 |
| `test_pull_no_starvation` | Pull mode with 4 nodes → all nodes get work |
| `test_pull_affinity_matches` | Pull mode → pull_affinity_matches > 0 |
| `test_multinode_perf_benchmark` | 60s sim with 4 nodes completes in < 15s |

Existing 13 tests pass unchanged (default config has no multi-node fields → defaults to N=1).

## Backward Compatibility

- `n_prefill_nodes=1` → single PrefillNode, trivial dispatch, identical behavior
- Old configs without multi-node fields → `ServiceConfig` defaults kick in
- `self._tier_stores` kept as compatibility alias: `{L1: node[0].l1, L2: node[0].l2, L3A: shared_l3a}`
- All existing tests, plots, and sweeps continue working

## Verification

1. Run `pytest tests/ -v` — all 13 existing tests pass
2. Run `pytest tests/test_multinode.py -v` — all new tests pass
3. Run `python scripts/sanity_plots.py` — plots generate correctly
4. Run `python scripts/profile_sim.py` — verify no significant perf regression for N=1
5. Manually verify: 4-node stressed sim shows reduced queue depth vs 1-node
