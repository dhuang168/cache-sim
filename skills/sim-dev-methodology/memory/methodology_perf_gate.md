---
name: Performance gate — profile before scale
description: Always profile new features at small scale before running large sims; catch O(n²) and worse before they waste hours
type: feedback
---

**Rule: Profile every new feature at small scale before running long or large-scale simulations.**

Before any sim longer than 2 minutes or wider than 8 GPUs with new code:
1. Run a 30-second sim with the new feature enabled
2. Profile with `cProfile` — check the top 10 functions by `tottime`
3. Verify no function is O(n²) or worse in the hot path (per-request or per-chunk)
4. If any function takes >10% of total time and scales with data size, optimize first
5. Only then scale up to full experiment duration/cluster size

**Why:** This mistake happened twice:
- Chunk store `evict_lru()` sorted the entire dict on every insertion — O(n log n) per chunk × 390 chunks per request = 40+ min for a 10-min sim. A 30-second profile would have shown `sorted()` consuming 50%+ of runtime.
- The same pattern: a function that's O(1) for small N becomes dominant at scale. Profile catches this in seconds, not hours.

**How to apply:**
```python
import cProfile, pstats
profiler = cProfile.Profile()
profiler.enable()
engine = SimEngine(cfg)  # short sim config
m = engine.run()
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('tottime')
stats.print_stats(15)
```

Check for:
- `sorted()` or `list.sort()` in hot paths → use heap or ordered container
- Dict iteration (`items()`, `values()`) in per-request code → precompute or index
- `allocated_blocks()` / `math.ceil()` called millions of times → precompute constants
- Any function called N_chunks × N_requests times → must be O(1)

**Gate criteria:** A new feature must run a 2-min sim at target scale in <30s wall-clock before being approved for long experiments.
