# Worker Topology — Plan #1: GPUs per Worker

Date: 2026-03-21

## Motivation

Currently each "prefill node" has independent L1/L2. In reality, 8 GPUs on one host share DRAM (L2) and local SSD (L3A). Only HBM (L1) is per-GPU.

```
Cluster
├── Worker 0 (1 host)
│   ├── GPU 0: L1 (own HBM, 80GB)
│   ├── GPU 1: L1 (own HBM)
│   ├── ...
│   ├── GPU 7: L1 (own HBM)
│   ├── Shared L2 (host DRAM, shared by 8 GPUs)
│   └── Local L3A (local SSD, shared by 8 GPUs)
├── Worker 1 (1 host)
│   └── (same structure)
└── Global L3A (networked storage, optional)
```

## Phase 1: Config changes

Add to `ServiceConfig`:
- `n_gpus_per_worker: int = 8` — GPUs sharing one host's DRAM/SSD

Reinterpretation of tier capacities:
- L1 (`tiers[0].capacity_bytes`): **per-GPU** (each GPU has own HBM)
- L2 (`tiers[1].capacity_bytes`): **per-worker** (shared by `n_gpus_per_worker` GPUs)
- L3A (`tiers[2].capacity_bytes`): **per-worker** when local, **global** when shared

Derived:
- `n_workers = n_prefill_nodes / n_gpus_per_worker` (must divide evenly)
- Each worker has `n_gpus_per_worker` PrefillNodes sharing one L2 and one local L3A

## Phase 2: Engine/node changes

### Worker grouping
- Introduce `Worker` concept (or group nodes by worker_id)
- GPUs on the same worker share:
  - One `TierStore` for L2 (DRAM)
  - One `TierStore` for local L3A (SSD) — when `l3a_shared=False`
- Each GPU still has its own:
  - `TierStore` for L1 (HBM)
  - Prefill slots and pending queue

### Transfer latency model
- Intra-worker L2/L3A access: **no inter-node penalty** (same host, PCIe/NVLink)
- Inter-worker L2/L3A access: `inter_node_latency_us` + bandwidth penalty
- Intra-worker L1 access from another GPU: small penalty (NVLink within host)

### Eviction paths
- L1→L2: GPU's L1 evicts to its worker's shared L2 (node-local in old model, now worker-local)
- L2→L3A: worker's L2 evicts to worker's local L3A (or global L3A if shared)
- Per-worker eviction engine manages the worker's L2 and L3A

### Cache lookup order
1. Assigned GPU's L1 (free)
2. Same-worker other GPUs' L1 (small NVLink penalty)
3. Same-worker shared L2 (no penalty — same host DRAM)
4. Other workers' L2 (inter-node penalty)
5. L3A — local (no penalty if same worker) or global (remote penalty)

### Backward compat
- `n_gpus_per_worker=1` (or `n_prefill_nodes=1`): identical to current behavior
- `n_gpus_per_worker=8, n_prefill_nodes=8`: 1 worker with 8 GPUs sharing L2/L3A

## Phase 3: Tests, plots, docs

- Verify 22 existing tests pass
- Add test: 2 workers × 4 GPUs = 8 nodes, verify shared L2 within worker
- Add test: intra-worker cache hits have no inter-node penalty
- Traffic mix verification before plot generation
- Update docs (CLAUDE.md, README, user_manual)
- Regenerate plots
- Crosscheck, commit, push
