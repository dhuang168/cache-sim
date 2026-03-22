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

### Conclusion
SSD capacity sensitivity is invisible at realistic hardware because **L2 (1TB) absorbs the entire coding working set**. L3A only becomes the primary cache when L2 is constrained (stressed config). The SSD size cliff appears when L3A per-worker capacity < coding KV object size (~8-16 GB).

To see SSD sensitivity at realistic hardware, you would need:
- Much longer sim duration (hours, not 60s) to accumulate more objects than L2 can hold
- Higher arrival rate to increase concurrent sessions
- Or constrained L2 capacity
