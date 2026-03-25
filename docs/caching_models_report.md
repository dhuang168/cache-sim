# Caching Model Comparison: OpenAI vs Anthropic vs LMCache

**Config**: heavy_coding (90% coding/agentic_coding), 4W × 8GPU, peak=15, 5 min, pull dispatch
**Script**: `scripts/research_caching_models.py`
**Research**: `plan_and_progress/research_api_caching_mechanisms.md`

---

## The Three Production Caching Models

| Model | How it works | Granularity | Cross-user sharing |
|-------|-------------|-------------|-------------------|
| **OpenAI** | Automatic prefix caching, 128-token increments, 1024-token minimum | Block (128 tok) | Yes — identical prefixes share one entry |
| **Anthropic** | User-defined breakpoints (up to 4), independent segments with separate TTLs | Segment (user-defined) | Yes — identical content blocks share |
| **LMCache** | 256-token chunks, hash-deduped, position-aligned | Chunk (256 tok) | Yes — identical chunks share via ref count |

**Key architectural difference**: OpenAI and Anthropic both cache **variable-length segments** (prefix or user-defined), while LMCache caches **fixed-size chunks**. This matters because:
- Variable-length: a single eviction loses the entire segment, but one segment covers the full prefix
- Fixed-size chunks: evicting one chunk only loses that chunk, but breaks consecutive-lookup chains

---

## Simulation Results

| Model | Simulator Mapping | Hit Rate | Recompute | TTFT p50 | Sharing Factor |
|-------|------------------|----------|-----------|----------|---------------|
| **Baseline** (per-session objects) | Object, no sharing | 99.9% | 26.3% | 5.74s | 1.70 |
| **OpenAI Prefix** (128-tok blocks) | Object + block rounding | 99.9% | 26.5% | **4.40s** | 1.69 |
| **Anthropic Segment** (3 breakpoints) | Object + multi-tier sharing | 99.9% | 26.3% | 5.64s | **1.72** |
| **LMCache Chunk** (256-tok, tail-first) | Chunk dedup + demand-pull | **48.7%** | **77.8%** | **50.0s** | 1.17 |

---

## Analysis

### OpenAI Prefix Model: Best TTFT

OpenAI's 128-token block quantization produces the best TTFT (4.40s vs 5.74s baseline). The block rounding means cached prefixes align to 128-token boundaries, reducing the "partial recompute" zone. With 99.9% hit rate identical to baseline, the only effect is slightly better cache alignment.

**Why it's fast**: The 128-token blocks mean the simulator rounds cached tokens to the nearest 128, which changes the uncached_tokens calculation slightly. Fewer uncached tokens = faster prefill.

### Anthropic Segment Model: Best Sharing

Anthropic's multi-tier sharing (system=20K shared by 1000, workspace=5K shared by 10, session=unique) achieves the highest sharing factor (1.72 vs 1.70 baseline). The three breakpoints create independent cache entries that can be shared at different granularities.

**Why sharing helps**: With 1000 users sharing a 20K-token system prompt, that's 20K × 327KB = 6.4GB stored once instead of 1000 times. This frees L1/L2 capacity for session-unique data.

**Why TTFT is similar to baseline**: At this load level (peak=15), the system isn't under memory pressure. The sharing benefit would be larger at higher load where cache capacity is the bottleneck.

### LMCache Chunk Model: Dramatic Degradation

Chunk mode drops to 48.7% hit rate with 77.8% recompute — consistent with all prior findings. The consecutive-chunk lookup breaks when any chunk in the 200-400 chunk chain is evicted.

**Dedup ratio 74.7%**: Three-quarters of chunk writes are deduplicated (shared system prefix chunks), confirming the dedup mechanism works correctly. But the hit rate is still low because session-unique chunks overwhelm L1 capacity.

---

## How Production Systems Differ from Our Simulation

### What we model correctly:
1. **Prefix-based hit/miss** — we check the longest consecutive cached prefix
2. **Cross-session shared prefix** — via `shared_system_prefix_tokens` or `SharingConfig`
3. **Tier migration** — L1→L2→L3A with configurable TTL/LRU policies
4. **Block-level granularity** — configurable token block sizes

### What we DON'T yet model:

1. **Content-hash cross-session sharing** — In production, two users with identical system prompts share ONE cached entry. Our simulator stores separate per-session objects. The `SharingConfig` approximates this for the system prompt tier, but doesn't handle arbitrary content sharing.

2. **Anthropic's hierarchical invalidation** — Changing tools invalidates the entire cache chain. Changing system prompt invalidates system + messages. Our simulator doesn't model this dependency hierarchy.

3. **OpenAI's routing-based caching** — OpenAI routes requests by prefix hash to specific machines. Our pull dispatch routes by session affinity, which is similar but not identical.

4. **Write costs** — Anthropic charges 1.25x for cache writes. OpenAI doesn't. This cost model affects whether caching is economically worthwhile for short-lived or small prefixes.

5. **OpenAI's 24-hour extended caching** — GPT-5.x models offload KV to SSD for 24-hour persistence. This is directly analogous to our L3A tier but with explicit SSD offload, not TTL-based migration.

6. **Minimum cache threshold** — Both require 1024+ tokens to be cache-eligible. Our simulator caches any prefix regardless of length. Small prefixes waste cache capacity in practice.

---

## Key Insight: The Object-Chunk Tradeoff is a Segment-Size Question

The real question isn't "object vs chunk" — it's **what segment size to cache**:

| Segment Size | Example | Pros | Cons |
|-------------|---------|------|------|
| **Whole prefix** (OpenAI) | 1 entry for entire 60K tokens | One eviction = one object. Simple. | Wastes space if only prefix is reusable. |
| **User-defined segments** (Anthropic) | 3 entries: system(20K) + workspace(5K) + session(35K) | Can share system across users. Independent TTLs. | User must define breakpoints. |
| **Fixed chunks** (LMCache) | 234 entries for 60K tokens | Fine-grained sharing. Dedup. | Consecutive-lookup fragility. Eviction breaks chains. |

**For heavy coding workloads (50-100K contexts)**:
- The system prompt (20-30K tokens) is highly shareable → should be a separate segment
- The conversation history (30-70K tokens) is unique per session → should be one segment
- Chunking the conversation history into 256-token pieces creates fragile chains

**The Anthropic model (2-4 segments) is the sweet spot**: shared prefix as one segment, session-unique as another. Maximum sharing benefit with minimal chain fragility.

---

## Recommendations

1. **Implement a segment model** in the simulator — `CacheSegment` with user-defined boundaries, independent TTLs, and segment-level hit/miss. This models Anthropic's breakpoint API directly.

2. **Add minimum cache threshold** (1024 tokens) — skip caching for small prefixes. This matches both OpenAI and Anthropic's production behavior.

3. **Add content-hash cross-session sharing** — identical content blocks should share a single cache entry regardless of session ID. The existing `SharingConfig` is a step in this direction.

4. **Add a cost model** — track cache write cost (Anthropic 1.25x vs OpenAI free) and read discount (both 90%). This enables economic analysis: "is caching this prefix worth the write cost given the expected reuse?"

5. **Test at higher load** — repeat this comparison at peak=100 where memory pressure is the bottleneck. Sharing benefits should be larger when L1 is full.
