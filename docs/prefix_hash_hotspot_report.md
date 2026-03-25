# Prefix-Hash Routing Hotspot Analysis

**Config**: heavy_coding, 4W × 8GPU, global L3A, 5 min
**Script**: `scripts/research_prefix_hash_hotspot.py`

---

## Results

| Config | Algorithm | Hit Rate | TTFT p50 | Load Imbalance | Overflow |
|--------|-----------|----------|----------|---------------|----------|
| peak=15 | push | 99.9% | **4.56s** | 5.6× | 0% |
| peak=15 | pull | 99.9% | 5.74s | 4.8× | 0% |
| peak=15 | **prefix_hash** | 99.9% | 6.36s | **24.5×** | **70%** |
| peak=50 | push | 99.9% | **5.67s** | 3.7× | 0% |
| peak=50 | pull | 99.9% | 6.85s | 3.0× | 0% |
| peak=50 | **prefix_hash** | 99.9% | **12.27s** | **27.5×** | **89%** |

---

## Key Findings

### 1. Prefix-Hash Creates Extreme Load Imbalance

Load imbalance (max node / min node requests):
- Push/Pull: 3-6× (moderate, natural variation)
- **Prefix-Hash: 24-28×** (extreme — some nodes get 24× more requests than others)

The heavy_coding config has only 3 profiles with nonzero mix (coding 45%, agentic_coding 45%, chat 5%). Since prefix-hash routes by profile name, all coding requests go to one node and all agentic_coding to another — only 2-3 of 32 nodes do most of the work.

### 2. High Overflow Rate

At peak=15: **70% overflow** — most requests can't fit on their target node (only 2-3 nodes receiving 90% of traffic). At peak=50: **89% overflow**. Overflow means the request goes to a different node where the KV isn't cached.

**However**, hit rate stays at 99.9% even with 89% overflow. This is because:
- Overflow nodes still have L3A (shared globally) with cached KV from previous requests
- The shared system prefix (20K tokens) is cached on the target node and pulled from L3A on overflow nodes
- At these load levels, L3A hasn't saturated

### 3. TTFT Degradation

Despite same hit rate, prefix_hash has **2× worse TTFT** at peak=50 (12.3s vs 5.7-6.9s). The overflow creates additional overhead:
- Overflow requests may not benefit from L1 cache affinity
- Queue wait on target nodes is higher (all traffic concentrated)
- The 2-3 hot nodes are fully saturated while others are idle

### 4. Push Actually Wins at Controlled Load

Surprisingly, push dispatch has the best TTFT (4.56s at peak=15, 5.67s at peak=50). This is because push assigns requests to the least-loaded node at arrival, achieving natural load balancing without the concentration problem.

---

## Implications for Production Deployments

1. **Prefix-hash routing works well when prefix diversity is high** (many different system prompts → requests spread across many nodes). It works poorly when prefix diversity is low (2-3 dominant system prompts → extreme concentration).

2. **For coding workloads** (dominated by 1-2 system prompts), prefix-hash creates severe hotspots. OpenAI's ~15 req/min overflow threshold is hit almost immediately, causing 70-89% of requests to overflow with degraded latency.

3. **Pull dispatch avoids the hotspot entirely** while maintaining higher cache affinity (53% vs 50% L1 hit) because nodes self-select work. Push gives best TTFT by distributing load evenly.

4. **The overflow mechanism is essential** — without it, the 2-3 target nodes would be completely overwhelmed. The high overflow rate (89%) means prefix-hash routing essentially degrades to load-balanced dispatch for most requests, just with extra overhead.

---

## Configuration

```json
{
  "service": {
    "dispatch_algorithm": "prefix_hash",
    "prefix_hash_tokens": 256,
    "prefix_hash_overflow_threshold": 0.9
  }
}
```

`prefix_hash_tokens` controls how many prefix tokens are hashed (default 256, matching OpenAI). Lower values increase collision (more sharing) but also more concentration. Higher values reduce collision but may split shared prefixes.
