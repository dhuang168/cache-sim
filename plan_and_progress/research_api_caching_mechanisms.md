# Research: API-Level Prompt Caching Mechanisms (OpenAI & Anthropic)

**Date**: 2026-03-24
**Purpose**: Inform simulator features for modeling real-world prompt caching as exposed at the API layer.

---

## 1. OpenAI Automatic Prompt Caching

### How it works

- **Fully automatic** — no user code changes required. All API requests benefit automatically.
- **Prefix-based** — the system caches the longest matching prefix of the prompt. Static content must go at the beginning; variable content at the end.
- **Exact match required** — even a single differing token in the prefix invalidates the cache. No fuzzy/partial matching.

### Granularity

- **Minimum prefix**: **1024 tokens** to be cache-eligible.
- **Increment size**: Cache hits occur in **128-token increments** above the 1024-token floor. A 1500-token prefix caches 1024 + 3*128 = **1408 tokens** (the largest multiple of 128 that fits within the prefix, starting from 1024). The remaining 92 tokens are recomputed.
- **Routing**: Requests are routed to a specific machine based on a hash of the initial ~256 tokens of the prompt. The optional `prompt_cache_key` parameter can be combined with the prefix hash to influence routing and improve hit rates.

### TTL / Eviction

- **Default (in-memory)**: Cached prefixes remain active for **5-10 minutes of inactivity**, sometimes up to **1 hour** during off-peak periods.
- **Extended (24-hour)**: Available for GPT-5.x and GPT-4.1 family. Set `"prompt_cache_retention": "24h"`. This **offloads KV tensors to GPU-local SSD storage** when VRAM is full, loading them back on cache hit. Note: extended caching opts out of Zero Data Retention (ZDR).

### Pricing

- **No additional fees** for caching or cache writes.
- **Cache hit discount**: Originally 50% (Oct 2024), now **up to 90% cheaper** than uncached input tokens for current models.
- **No charge for cache writes or storage**.

### Multi-turn conversations

- The growing prefix (system prompt + conversation history) is naturally cached turn-over-turn because each new request starts with the same prefix.
- Request N+1 reuses the cached prefix from Request N. Only the new assistant response + user message at the end are uncached.

### Hit rate data

- One OpenAI coding customer improved hit rate from **60% to 87%** by using `prompt_cache_key`.
- Requests exceeding ~**15 requests/minute** for the same prefix may overflow to additional machines, reducing hit rate.

### Latency improvements

- Short prompts (1024 tokens): **~7% faster** TTFT.
- Long prompts (150k+ tokens): **~67% faster** TTFT.
- Claimed up to **80% latency reduction**.

### Supported models

All models GPT-4o and newer, including GPT-5.x, GPT-4.1, fine-tuned variants.

---

## 2. Anthropic Message API Cache Breakpoints

### How cache_control breakpoints work

- **Explicit user control** — the developer places `cache_control` markers on specific content blocks in the request.
- **Two modes**:
  1. **Automatic caching** (recommended): Add `cache_control` at the request top level. The system automatically applies the breakpoint to the last cacheable block and moves it forward as conversations grow.
  2. **Explicit breakpoints**: Place `cache_control` directly on individual content blocks for fine-grained control over exactly what gets cached.

### Breakpoint placement

Breakpoints can be placed on:
- Tool definitions
- System message content blocks
- Text messages (user and assistant turns)
- Images and documents (user turns only)
- Tool use blocks and tool results

**Cannot** have direct `cache_control`:
- Thinking blocks (but they ARE cached alongside other content as input tokens)
- Sub-content blocks (citations, etc.) — cache the parent block instead

### Maximum breakpoints

**Up to 4 explicit `cache_control` markers** per request.

### Lookback window

Each breakpoint has a **20-block lookback window**. The system walks backward from the breakpoint checking for previously cached entries. Only finds entries from prior requests.

### Minimum cacheable tokens

| Model | Minimum Tokens |
|-------|---------------|
| Claude Opus 4.6, 4.5 | 4,096 |
| Claude Sonnet 4.6 | 2,048 |
| Claude Sonnet 4.5, 4, Opus 4.1, 4 | 1,024 |
| Claude Haiku 4.5 | 4,096 |
| Claude Haiku 3.5, 3 | 2,048 |

Prompts shorter than the minimum are processed without caching (no error, just no cache).

### TTL

- **Default "ephemeral"**: **5-minute** lifetime. Refreshes on each use within the window.
- **Extended**: **1-hour** lifetime. Set via `"ttl": "1h"` in cache_control. Costs more to write.
- **Constraint**: When mixing TTLs in one request, longer TTL entries must appear before shorter ones.

### Pricing

| Operation | Cost Multiplier |
|-----------|----------------|
| Cache write (5m TTL) | **1.25x** base input token price |
| Cache write (1h TTL) | **2x** base input token price |
| Cache read/refresh | **0.1x** base input token price (90% discount) |
| Regular input tokens | 1x (base price) |

**Example for Claude Opus 4.6** ($5/MTok base):
- 5m cache write: $6.25/MTok
- 1h cache write: $10/MTok
- Cache read: $0.50/MTok

### Growing conversations (multi-turn)

With automatic caching:
- **Request 1**: Everything written to cache.
- **Request 2**: Previous content read from cache; new content written.
- **Request 3**: System through User(2) read from cache; Asst(2) + User(3) written.

The breakpoint automatically moves to the last cacheable block, so no manual breakpoint updates are needed.

### The "ephemeral" cache type

"Ephemeral" is currently the **only** supported cache type. The name refers to the short-lived nature (5 min default). The `ttl` field within `ephemeral` controls the duration (5m or 1h).

### Cache invalidation rules

Changes cascade through a hierarchy: `tools -> system -> messages`. Changing tools invalidates all caches. Changing system invalidates system + messages. Specifically:

| Change | Invalidates Tools | Invalidates System | Invalidates Messages |
|--------|:-:|:-:|:-:|
| Tool definitions | yes | yes | yes |
| Web search toggle | no | yes | yes |
| Tool choice | no | no | yes |
| Extended thinking settings | no | no | yes |

### Response metrics

```json
{
  "usage": {
    "cache_creation_input_tokens": 5000,
    "cache_read_input_tokens": 10000,
    "input_tokens": 50,
    "output_tokens": 200
  }
}
```

### Cache isolation

Organization-level isolation (workspace-level starting Feb 2026). Caches are NOT shared across organizations/workspaces. Exact matching required for cache hits.

---

## 3. Key Architectural Differences

| Dimension | OpenAI | Anthropic |
|-----------|--------|-----------|
| **User control** | Fully automatic (opt-out not possible) | Explicit breakpoints (up to 4) or automatic mode |
| **Granularity** | 128-token increments above 1024 floor | Content-block level (user defines boundaries) |
| **Minimum** | 1024 tokens | 1024-4096 tokens (model-dependent) |
| **TTL (default)** | 5-10 min (up to 1 hr off-peak) | 5 min (refreshes on use) |
| **TTL (extended)** | 24 hours (GPU-local SSD, GPT-5.x only) | 1 hour |
| **Write cost** | Free | 1.25x (5m) or 2x (1h) base price |
| **Read discount** | Up to 90% | 90% (0.1x base) |
| **Max entries per request** | 1 (longest prefix) | Up to 4 breakpoints |
| **Cache key** | Prefix hash + optional `prompt_cache_key` | Exact content match per segment |
| **Cache storage** | VRAM (default) or GPU-local SSD (extended) | Not publicly documented |

### How they differ structurally

**OpenAI** = **Single contiguous prefix model**. The cache stores ONE entry per unique prefix. The entry covers tokens [0, N] where N is the longest cached prefix in 128-token increments. There is no way to cache a "middle" segment independently.

**Anthropic** = **Segment model with up to 4 user-defined boundaries**. A request with 3 breakpoints creates 3 separate cacheable segments:
1. Tokens [0, breakpoint_1] — e.g., tool definitions
2. Tokens [breakpoint_1+1, breakpoint_2] — e.g., system prompt
3. Tokens [breakpoint_2+1, breakpoint_3] — e.g., conversation history

Each segment has its own cache hit/miss status and can have different TTLs.

---

## 4. Mapping to Simulator Models

### Current simulator models

The simulator currently supports two cache object models:

1. **Per-session object model** (`CacheObject` in `sim/cache.py`): One KV cache entry per session covering a token range. The entry grows as the session progresses. This maps well to **OpenAI's prefix model** where the entire prefix is one cache entry.

2. **Block/chunk model** (token blocks at 16/256/4096 token granularity): KV cache is divided into fixed-size token blocks. Cache hits are rounded to block boundaries. This maps to **the internal KV cache management** in inference engines (vLLM PagedAttention, etc.) but does NOT map to either API's external caching model.

### What's missing: The Segment Model

Neither the per-session object model nor the block model captures Anthropic's **segment model**. The key differences:

| Property | Per-session Object | Block/Chunk | Segment (Anthropic) | Prefix (OpenAI) |
|----------|:-:|:-:|:-:|:-:|
| Entries per session | 1 | N (fixed-size) | Up to 4 | 1 |
| Boundaries | Session-defined | Fixed grid | User-defined | Automatic (128-tok) |
| Independent TTL per segment | No | No | Yes | No |
| Cross-session sharing | Via shared prefix | Via ref-counted blocks | Via exact content match | Via prefix hash match |
| Partial hit | All or nothing | Block-level | Segment-level | 128-token increments |

### Proposed segment model for the simulator

To model Anthropic's breakpoint system, we would need:

1. **CacheSegment** — a new object type representing a user-defined cache segment with:
   - `segment_id` (hash of content)
   - `token_range: (start, end)`
   - `ttl_s` (5 min or 1 hour)
   - `created_at_us`, `last_accessed_at_us`
   - `size_bytes`

2. **Per-request segment decomposition** — each request declares up to 4 breakpoint positions. The simulator splits the prompt into segments at these boundaries.

3. **Segment-level cache lookup** — for each segment, check if an identical segment (by content hash) exists in the cache. Segments can hit/miss independently.

4. **Cross-session segment sharing** — unlike the current model where KV objects are per-session, segments with identical content (e.g., the same system prompt used by different sessions) can share cache entries. This is a major difference from the current model.

5. **Independent TTL per segment** — system prompt segments might use 1h TTL; conversation history uses 5m TTL.

### Proposed prefix model for OpenAI

The current per-session object model already approximates OpenAI's prefix model, but needs:

1. **128-token quantization** — cache hits should be rounded to 128-token boundaries (above 1024 floor).
2. **Automatic prefix matching** — no user control over what gets cached.
3. **Cross-session prefix sharing** — requests with identical prefixes (from different sessions/users) share the same cache entry. This is how OpenAI works, and it's the main difference from our per-session model.
4. **TTL options** — 5-10 min default, or 24h extended (with SSD offload modeling).

### Key insight: Cross-session sharing semantics differ

- **Current simulator**: KV objects are per-session. Two users with identical system prompts each store their own KV objects. Only `shared_system_prefix_tokens` provides cross-session sharing, and only for prefill avoidance — NOT for shared KV storage.
- **OpenAI**: Two requests with identical 1024+ token prefixes literally share the same cached KV tensors. This is true cross-session, cross-user sharing.
- **Anthropic**: Two requests with identical content at a breakpoint share the same cached segment. The segment identity is the content hash.

This is a fundamental architectural difference that would significantly impact cache hit rates and capacity requirements in the simulator.

---

## 5. Implications for Simulator Development

### Short-term: Improve existing model fidelity

1. Add **128-token quantization** to cache hit calculations (models OpenAI's increment granularity).
2. Add **minimum prefix threshold** (1024 tokens) below which no caching occurs.
3. Add **content-hash-based cross-session sharing** for prefix segments (models both OpenAI and Anthropic's actual sharing behavior).

### Medium-term: Segment model

1. Introduce **CacheSegment** as a new object type alongside CacheObject.
2. Model **up to 4 breakpoints** per request with independent TTLs.
3. Model **segment-level cache hit/miss** with separate metrics per segment type (tools, system, history, recent).
4. Model **write cost overhead** (Anthropic's 1.25x/2x write penalty).

### Long-term: Dual-mode simulation

1. **OpenAI mode**: Automatic prefix caching, 128-token blocks, single prefix entry, free writes, up to 90% read discount.
2. **Anthropic mode**: Explicit breakpoints, segment-level caching, 4 segments, paid writes, 90% read discount.
3. **Cost model**: Track total API cost including cache writes, cache reads, and uncached input tokens.

---

## Sources

- [OpenAI Prompt Caching Guide](https://developers.openai.com/api/docs/guides/prompt-caching)
- [OpenAI Prompt Caching Announcement (Oct 2024)](https://openai.com/index/api-prompt-caching/)
- [OpenAI Prompt Caching 201 Cookbook](https://developers.openai.com/cookbook/examples/prompt_caching_201)
- [OpenAI GPT-5.1 Extended Caching Announcement](https://openai.com/index/gpt-5-1-for-developers/)
- [Anthropic Prompt Caching Docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Anthropic Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [Anthropic Prompt Caching Announcement](https://www.anthropic.com/news/prompt-caching)
- [OpenAI Community: 1024 token minimum discussion](https://community.openai.com/t/why-does-prompt-caching-requires-at-least-1024-tokens/1363167)
- [OpenAI Community: prompt_cache_key hit rate improvement](https://community.openai.com/t/prompt-token-cache-gaming-to-save-money/984600)
