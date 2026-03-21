# Multi-Node Prefill Dispatch — Plan #2: Global L3 Value Analysis

Date: 2026-03-21

## Motivation

The node-scaling plot revealed that TTFT improvement is dominated by throughput (more slots), not cache locality. The real question is: **what is the value of a shared central cache (global L3) as you scale nodes?**

Key dimensions:
- (a) Less eviction penalty for local nodes — objects remain reachable in global L3
- (b) Higher aggregate cache hit rate — cross-node sharing
- (c) TTL interaction — TTL controls flow into global L3
- (d) Reduced queue time at same injection rate — cache hits make slots more efficient
- (e) Sustaining QPS at SLA — the business metric
- (f) Global L3 latency sensitivity — at what latency does the benefit vanish?

## Phase 1: Infrastructure Changes

### 1.1 Add queue wait time metric
- Track `queue_wait_us` per request in `MetricsCollector` — time from entering pending queue to slot obtained
- This is the foundation for SLA-based analysis and TTFT decomposition

### 1.2 Add local-L3 mode
- New `ServiceConfig` fields: `l3a_shared: bool = True`, `l3a_remote_latency_us: int = 50_000` (50ms)
- When `l3a_shared=False`: each node gets its own L3A with `capacity / n_nodes`, no cross-node L3 access
- When `l3a_shared=True`: current behavior, but global L3 access adds `l3a_remote_latency_us` penalty

### 1.3 Add sustaining QPS metric
- Helper function that binary-searches for max arrival rate multiplier where p95 queue wait stays under a configurable SLA threshold
- Used in the analysis plots

## Phase 2: Analysis Plots (`plot_node_scaling` rewrite)

| Panel | X-axis | Series | What it shows |
|-------|--------|--------|---------------|
| Cache hit rate breakdown | Node count | Global L3 vs Local L3 | (b) Global L3 lifts aggregate hit rate |
| Per-node eviction rate | Node count | Global vs Local | (a) Same eviction rate but different miss penalty |
| Queue wait p95 | Node count | Global vs Local | (d) Global L3 reduces queue time |
| Sustaining QPS at SLA | Node count | Global vs Local | (e) Business value metric |
| TTFT p95 vs L3 latency | L3 remote latency (0-100ms) | Fixed node count (e.g., 4) | (f) Latency sensitivity |
| TTL sensitivity | TTL_L2 values | Fixed nodes, Global vs Local | (c) TTL interaction |

## Phase 3: Progress Report

Save analysis results and observations to `plan_and_progress/multinode_progress_2.md`.
