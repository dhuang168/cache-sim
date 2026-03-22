# Experiment B: Sharing × L3A Mode

**Config**: heavy_coding, 4 workers (32 GPUs), 5 min, peak=3, LRU eviction

## Results

| Sharing | L3A Mode | Hit Rate | Miss | Dropped | Queue Wait | TTFT Mean | Memory Saved | Contention |
|---------|---------|----------|------|---------|-----------|-----------|-------------|------------|
| Off | Global | 100% | 1 | 11% | 1,313ms | 12,237ms | — | 1,021 |
| Off | Local | 100% | 1 | 0% | 2ms | 9,232ms | — | 0 |
| On | Global | 100% | 1 | 10% | 1,533ms | 12,240ms | 5,547 GB | 952 |
| On | Local | 100% | 1 | 0% | 2ms | 9,187ms | 10,672 GB | 0 |

## Findings

1. **Local L3A outperforms global at low load (5 min).** Local has 0% drops, 2ms queue wait, 9.2s TTFT. Global has 10-11% drops, 1.3s queue wait, 12.2s TTFT. This is because at 5 min the system hasn't reached the session migration tipping point — L1/L2 still hold most objects, and local L3A's dedicated SSD bandwidth avoids contention.

2. **Global L3A bandwidth contention is significant.** 952-1,021 contention events even at low load. The 50ms remote latency + shared SSD bandwidth increases prefill time, which cascades into higher queue wait and 10% drops.

3. **Local L3A has much higher affinity.** Local: 95% affinity dispatch (1,946 of 2,118). Global: 20% affinity (421 of 2,118). With local L3A and LRU eviction, objects stay in L1/L2 on the same worker → affinity holds → sessions don't migrate.

4. **Sharing saves more memory with local L3A** (10,672 GB vs 5,547 GB). With local L3A, each worker has its own shared prefix copies. Sharing deduplicates within each worker's sessions, and there are more sessions per worker staying local.

5. **The 5-min sim is too short for session migration.** At 5 min with peak=3, L1/L2 are not saturated yet. The global L3A advantage (cross-worker access after migration) hasn't kicked in. This matches earlier findings: global L3A's advantage requires 10-20 min for L1/L2 to fill and sessions to migrate.

## 20-Minute Results (Steady State)

At 20 min, L1/L2 saturate and session migration reveals the true tradeoff:

| Sharing | L3A Mode | Hit Rate | Miss | Dropped | Completed | QW Mean | TTFT Mean | TTFT p95 | Mem Saved | Contention |
|---------|---------|----------|------|---------|-----------|---------|-----------|----------|-----------|------------|
| Off | Global | **99.8%** | 61 | **88%** | 2,927 | 72s | 149s | 737s | — | 21,498 |
| Off | Local | 77.1% | 5,708 | 40% | **14,828** | 50s | 79s | 299s | — | 0 |
| On | Global | **99.8%** | 61 | **89%** | 2,809 | 70s | 148s | 755s | 10,301 GB | 20,703 |
| On | Local | 78.0% | 5,471 | 40% | **14,892** | 49s | **78s** | **297s** | **69,766 GB** | 0 |

### Key Findings at 20 min

1. **Global L3A has 99.8% hit rate but 88-89% drop rate.** The bandwidth contention (20K+ events) makes global L3A transfers so slow that prefill slots are occupied much longer → queues fill → 88% of requests dropped. Only 2,809-2,927 requests complete out of ~25K total.

2. **Local L3A completes 5× more requests** (14,828 vs 2,927) despite lower hit rate (77% vs 99.8%). The dedicated SSD bandwidth per worker (no contention) means each L3A transfer is fast → slots free quickly → lower drop rate (40% vs 88%).

3. **Sharing saves 70 TB with local L3A.** The framework and workspace prefixes are ref-counted within each worker. With 4 workers × many sessions, the cumulative savings are massive.

4. **The hit rate vs throughput tradeoff is real.** Global L3A wins on hit rate (99.8%) but loses on throughput (2.8K completed). Local L3A loses on hit rate (78%) but wins on throughput (14.9K completed). The cold misses with local L3A cost 37-120s each, but the lack of contention means more requests get served overall.

5. **Contention is the deciding factor, not hit rate.** At 20 min, global L3A has 20K+ contention events. Each contention multiplies the SSD transfer time by the number of concurrent readers. With many sessions accessing the framework prefix via global L3A simultaneously, the effective bandwidth per reader drops to a fraction of the 7 GB/s SSD.

### Why the 5-min and 20-min results differ

| Metric | 5 min (Local) | 20 min (Local) | Change |
|--------|-------------|---------------|--------|
| Hit rate | 100% | 78% | L1/L2 saturated → misses |
| Dropped | 0% | 40% | Queue pressure from cold misses |
| Affinity | 95% | 75% | Session migration after L1/L2 fill |

| Metric | 5 min (Global) | 20 min (Global) | Change |
|--------|-------------|---------------|--------|
| Hit rate | 100% | 99.8% | Stable (global pool) |
| Dropped | 10% | **88%** | Contention cascade |
| Contention | 952 | **21,498** | 22× more as sessions accumulate |

## Implication

The **optimal L3A strategy depends on both time horizon AND contention**:

- **Short-term (< 5 min)**: Local L3A wins — dedicated bandwidth, no contention, high affinity
- **Long-term, low contention**: Global L3A wins — cross-worker access after session migration
- **Long-term, high contention**: Local L3A wins on throughput despite lower hit rate — bandwidth contention on global L3A is the bottleneck, not cache misses

**The real solution** is to combine global L3A's hit rate with local L3A's bandwidth:
- **Replicate hot blocks** to each worker's local SSD (like the LRU self-regulation in L1/L2)
- **L3A-aware affinity dispatch**: route sessions back to the worker that has their KV in local SSD
- **Tiered global L3A**: use global for cold blocks only, keep hot blocks local
