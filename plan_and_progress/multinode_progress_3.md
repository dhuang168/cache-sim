# Multi-Node Prefill Dispatch — Progress #3: Gap Closure

Date: 2026-03-21

## Gaps Closed

All 3 gaps from the crosscheck have been addressed:

### 1. Sustaining QPS at SLA panel (was: missing)
- `plot_sustaining_qps()` implemented using `find_sustaining_qps()` binary search
- Sweeps node count [1,2,4,8] for both global and local L3A
- SLA: p95 queue wait < 500ms
- Results: sustaining QPS scales roughly linearly (10→126 QPS at 8 nodes)
- Global and local L3A nearly identical (both 100% hit rate)

### 2. TTL sensitivity panel (was: missing)
- `plot_ttl_sensitivity()` sweeps L2 TTL [1,5,10,20,60,120]s at 4 nodes
- Compares global vs local L3A hit rate and queue wait
- Results: 100% hit rate across all TTL values; queue wait slightly lower for local (no 50ms remote penalty)

### 3. Cache-hostile config investigation (was: 100% hit rate masks differences)
- Investigated: L2 32MB→2GB, L3A 2GB→50GB, TTL 1→120s
- Finding: gap is a **cliff effect** driven by per-node L3A capacity vs KV object size
  - Global L3A pools capacity → 0% miss even at 2GB total
  - Local L3A fragments → 100% miss when per-node capacity < largest KV object
  - Gradual degradation doesn't occur because KV objects have similar sizes
- This is a real finding about the architecture: global L3A's value is capacity pooling across the object-size cliff

## Crosscheck: All plan items delivered
| Plan Item | Status |
|-----------|--------|
| Queue wait metric (1.1) | Done |
| Local-L3 mode (1.2) | Done |
| Sustaining QPS helper (1.3) | Done |
| Cache hit rate panel (b) | Done |
| Per-node eviction rate panel (a) | Done |
| Queue wait p95 panel (d) | Done |
| Sustaining QPS panel (e) | Done |
| L3A latency sensitivity panel (f) | Done |
| TTL sensitivity panel (c) | Done |

## Test results
- 22/22 tests pass (13 existing + 9 multinode)
- New plots: ttl_sensitivity.png, sustaining_qps.png
- Updated: node_scaling.png (6 panels)
