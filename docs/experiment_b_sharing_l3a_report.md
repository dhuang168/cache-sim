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

## Implication

The **optimal L3A strategy depends on time horizon**:
- **Short-term (< 5 min)**: Local L3A wins — dedicated bandwidth, no contention, high affinity
- **Long-term (> 10 min)**: Global L3A wins — cross-worker access after session migration

This suggests a **hybrid approach**: start with local L3A, switch to global when session migration rate exceeds a threshold. Or: use local L3A with L3A-aware affinity dispatch to route migrated sessions back to their original worker.
