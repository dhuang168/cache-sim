# L3A Sensitivity Triage

Date: 2026-03-21

## Problem
SSD capacity sweep showed no difference between global and local L3A down to 200GB/worker. Why isn't L3A sensitive to capacity?

## Hypotheses
1. L2 TTL (300s) > sim duration (60s) — objects never expire from L2
2. Too few unique cache blocks — small working set

## Investigation

### Metrics captured
- Object placement by tier
- Total working set size vs tier capacity
- L2→L3A migration count with different TTLs

### Findings

**With stressed L1/L2 (500MB L1, 10GB L2), 4 workers:**

| Metric | Value |
|--------|-------|
| Total KV objects created | 5,560 |
| Total working set | 47,836 GB |
| Objects in L1 | 0 (too small) |
| Objects in L2 | 56 (only small chat/batch objects fit in 10GB) |
| Objects in L3A | 5,504 (avg 8.7 GB each) |
| L2→L3A migrations (TTL=300s) | 0 (TTL never fires in 60s sim) |
| L2→L3A migrations (TTL=10s) | 150 |
| Total L2+L3A capacity (local) | 240 GB |
| Working set oversubscription | 200× |

### Root cause
Both hypotheses were partially wrong. The actual cause is:

1. **L2 (10GB/worker) is too small for coding KV objects (8-16GB)**. They bypass L2 entirely and go straight to L3A. L2 TTL is irrelevant because objects never enter L2.

2. **With realistic L2 (1TB/worker), coding objects FIT in L2**. L3A is only used for overflow, which barely happens in a 60s sim. That's why SSD size doesn't matter at realistic hardware.

3. **The working set is massive (47TB) but objects are constantly churned**. 5,504 objects compete for 200GB L3A, with heavy eviction (5,538 cold evictions). The 99% hit rate comes from prefix trie matches finding objects that happen to still be resident.

### Conclusion (updated after longer sim analysis)

Initial conclusion was wrong. L2 does NOT absorb the entire working set — it saturates at 5 min. Both L2 and L3A reach 100% saturation. The real reason global vs local L3A are identical:

**Affinity dispatch keeps each session's objects on its assigned worker.** With push dispatch, once a session is assigned to a worker, all KV objects stay on that worker. Global L3A pooling provides no cross-worker benefit because no session's objects are accessed from a different worker.

At 20 min, each worker places ~740 TB through its 8 TB L3A (93× churn). Both global (32 TB) and local (8 TB/worker) show identical 99.77% hit rate because the access pattern is per-worker, not cross-worker.

Global L3A would help with:
- Session migration (requests land on different workers)
- Shared prefix deduplication (one copy serves all workers)
- Load imbalance (global pool absorbs overflow from busy workers)
