# Worker Topology — Progress #1

Date: 2026-03-21

## Delivered

### Config
- `n_gpus_per_worker: int = 8` in ServiceConfig
- L1 = per-GPU, L2 = per-worker (shared DRAM), L3A = per-worker (local) or global (shared)

### Engine changes
- GPUs grouped into workers: `worker_id = node_id // gpus_per_worker`
- GPUs on same worker share L2 TierStore and local L3A TierStore references
- `_same_worker()` helper for transfer penalty decisions
- `_cross_node_transfer_us()` distinguishes intra-worker (NVLink base latency only) vs inter-worker (full penalty)
- Intra-worker L2 hit: zero penalty (shared DRAM)
- Per-worker aggregation for occupancy, fragmentation, TTL checks (avoids double-counting)

### Tests
- `test_worker_topology_shared_l2`: 8 GPUs / 2 workers, verifies L2 sharing and worker_id
- `test_intra_worker_no_l2_penalty`: verifies `_same_worker()` logic
- 24/24 tests pass

### Crosscheck: all plan items delivered
No gaps.
