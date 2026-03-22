# Global L3A Transfer Fix + Smart Push Dispatch — Plan

Date: 2026-03-22

## Fix 1: Global L3A transfer populates local L1

### Current (wrong)
Every L3A cache hit reads from global SSD. N concurrent readers share bandwidth.
Contention scales with total concurrent L3A reads.

### Correct behavior
1. Request hits global L3A → transfer KV to requesting node's L1 (one-time SSD read)
2. KV now lives in local L1 → subsequent requests for same session use L1 (no SSD)
3. Contention only on **first access per object per worker** (cache warming), not every request
4. After warming, the object has both an L3A copy (global) and L1 copy (local)

### Implementation
- After an L3A hit, the engine already places the KV in the assigned node's L1 (via `_place_kv_object` in `_on_decode_complete`)
- The fix: when computing contention, only count concurrent reads for **distinct objects**, not repeat reads
- Also: once an object is in a worker's L1/L2, future affinity checks find it → no more L3A reads
- The contention counter should track unique-object reads, not total reads

## Fix 2: Smart Push Dispatch

### Current push
1. Check L1/L2 affinity only
2. If affinity found and node not overloaded → route there
3. Else → least loaded node

### Enhanced push (predictive routing)
1. For each candidate node, compute expected total time:
   - `cache_tier` = check ALL tiers (L1, L2, L3A) for session's KV on that node's worker
   - `expected_prefill_us` = estimate based on cache_tier (L1 hit = fast, L3A = transfer + partial, miss = full recompute)
   - `expected_wait_us` = node's projected_free_time - current_time
   - `expected_total_us` = expected_wait_us + expected_prefill_us
2. Pick node with lowest `expected_total_us`
3. Tiebreak: prefer nodes with higher-tier cache hits (L1 > L2 > L3A > miss)

### Config
- `dispatch_algorithm: str = "push"` — current behavior
- `dispatch_algorithm: str = "push_smart"` — predictive routing with full tier visibility
- `dispatch_algorithm: str = "pull"` — unchanged

## Tests
1. `test_l3a_hit_populates_l1` — after L3A hit, verify object appears in local L1
2. `test_contention_only_first_access` — second access to same object has no contention
3. `test_smart_push_prefers_l1_hit` — smart push routes to node with L1 cache hit
4. `test_smart_push_avoids_overloaded` — smart push avoids node with long queue even if it has cache
5. `test_smart_push_uses_l3a_affinity` — smart push considers L3A hits (not just L1/L2)
6. `test_smart_push_vs_push_hit_rate` — smart push should have better hit rate than basic push

## Implementation sequence
1. Fix L3A contention model (first-access only)
2. Add L3A-aware affinity to dispatch
3. Implement smart push with expected_total_us scoring
4. Tests
5. Re-run Experiment B and C
6. Reports, commit
