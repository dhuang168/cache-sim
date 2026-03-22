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

## 20-Minute Results (Steady State, corrected contention model)

**Contention model fix**: Global L3A = N workers' SSDs pooled. Concurrent reads to different objects on different SSDs don't contend. Per-SSD contention = `total_concurrent / n_workers`. Additionally, L3A reads populate local L1 — subsequent requests use L1 (no repeated SSD reads).

At 20 min, L1/L2 saturate and session migration reveals the true tradeoff:

| L3A Mode | Hit Rate | Miss | Dropped | Completed | QW Mean | TTFT Mean | Contention |
|---------|----------|------|---------|-----------|---------|-----------|------------|
| **Global** | **99.8%** | 61 | **4%** | **23,781** | **2.8s** | **25.6s** | 493 |
| Local | 78.0% | 5,471 | 40% | 14,892 | 48.6s | 78.2s | 0 |

*(With sharing enabled, LRU eviction, peak=3)*

### Key Findings at 20 min

1. **Global L3A wins decisively.** 99.8% hit rate, 4% drops, 60% more completed requests (23.8K vs 14.9K), 3× better TTFT (25.6s vs 78.2s).

2. **Corrected contention is modest.** Only 493 contention events (was 22K with wrong model). Objects are distributed across N workers' SSDs, so concurrent reads to different objects don't contend. First L3A read populates local L1 — subsequent requests use L1 (no repeated SSD reads).

3. **Local L3A's 40% drop rate** is caused by cold misses (78% hit rate). Each cold miss takes 37-120s, blocking slots and cascading into queue buildup. Global L3A avoids this with 99.8% hit rate.

4. **Sharing saves memory** but doesn't change the hit rate story at this scale. The real benefit of sharing appears in memory capacity planning for large deployments.

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

With the corrected contention model (per-SSD, not global), **global L3A is the clear winner** for multi-worker coding deployments. The 50ms remote latency + modest contention is a small price for 99.8% hit rate and 60% more throughput.
