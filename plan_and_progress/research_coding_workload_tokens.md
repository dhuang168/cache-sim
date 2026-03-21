# Research: Modern Coding Workload Token Characteristics

Date: 2026-03-21

## Question

Modern coding workloads (Claude Code, Cursor, GitHub Copilot) — what are typical prompt lengths and cacheability?

## Findings

### Total prompt length: ~40-80k tokens (median for agentic coding turns)

- **Claude Code**: ~40k tokens early-session (system prompt 2.7k + tool defs 16.8k + agents/skills 2.3k + memory/CLAUDE.md 7.4k + conversation 9.6k), growing to 100k+ in longer sessions. 200k context window.
- **Cursor**: 60k-135k input tokens per request commonly reported. A modest example is ~13k (2k system + 6k code context + 4k conversation + 1k user message). Agent mode routinely hits 60k+.
- **GitHub Copilot**: 64k-128k context windows, ~30% reserved for output, leaving 40-90k for input.

**60k tokens is a reasonable central estimate** for agentic coding. Simple completions are smaller (13k), long sessions go to 200k+.

### Cacheability: 80-90% is conservative, real-world is 92-99%+

- **Claude Code achieves 92% cache hit rate** in real 30-min sessions, reaching 99.7% by message 10. One power user: 4.2B tokens read from cache vs 197M written vs 1.3M uncached (>95% cache hits).
- **Static prefix (system + tools) is ~20-30k tokens** — identical across every turn.
- **Academic paper** ("Don't Break the Cache", arxiv 2601.06007): prompt caching reduces costs 41-80%, cacheable percentages 54-89% at 50k prompt sizes.
- **Anthropic docs**: for a 50-turn coding agent with 10k system prompt, 500k tokens of identical instructions without caching.

### Token distribution breakdown (Claude Code, representative)

| Component | Tokens | % of Total | Cacheable? |
|---|---|---|---|
| System prompt | ~2,700 | ~7% | Yes (static) |
| Tool definitions | ~16,800 | ~42% | Yes (static) |
| Custom agents/skills | ~2,300 | ~6% | Yes (static per session) |
| Memory/CLAUDE.md files | ~7,400 | ~19% | Yes (static per session) |
| Prior conversation history | ~9,600+ | ~24% | Yes (grows but stable prefix) |
| Current user message | ~1,000 | ~2% | No (new each turn) |
| **Total** | **~40,000** | | **~98% cacheable** |

As sessions progress, conversation grows to 100k+ but remains part of stable prefix (prior turns don't change).

## Sources

- [Anthropic Prompt Caching Docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Claude Code Context Buffer: The 33K-45K Token Problem](https://claudefa.st/blog/guide/mechanics/context-buffer-management)
- [Why 99% of What You Send to Claude is Already Cached](https://www.frr.dev/posts/prompt-caching-99-percent-cache-hit/)
- [How Prompt Caching Actually Works in Claude Code](https://www.claudecodecamp.com/p/how-prompt-caching-actually-works-in-claude-code)
- [Claude's system prompt is over 24k tokens with tools (HN)](https://news.ycombinator.com/item?id=43909409)
- [The Anatomy of a Cursor Prompt](https://dev.to/tawe/the-anatomy-of-a-cursor-prompt-2hb8)
- [Cursor: 3 sentences into 60k+ tokens (Forum)](https://forum.cursor.com/t/cursor-always-turns-3-sentences-into-60k-tokens/149061)
- [GitHub Copilot Context Window Discussion](https://github.com/orgs/community/discussions/188691)
- [Don't Break the Cache (arxiv 2601.06007)](https://arxiv.org/abs/2601.06007)
- [Anthropic Prompt Caching Announcement](https://www.anthropic.com/news/prompt-caching)

## Implication for Simulator

The current "coding" profile has `input_len_mean_tokens=800` and `shared_system_prefix_tokens=2048` — **both underestimate real workloads by 10-30x**. This means the simulator's KV objects are much smaller than reality, which is why L3A capacity pressure and global vs local L3A differences don't manifest in the stressed config.
