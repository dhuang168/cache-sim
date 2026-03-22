# Research: Cross-Session KV Cache Sharing Patterns

Date: 2026-03-22

## 1. System Prompt Diversity

### How many unique system prompts exist?

| Product | Total Prompt | Shared Across Users | Personalized | Unique Templates:Users |
|---------|-------------|--------------------:|------------:|-----------------------|
| Claude Code | ~80K tokens | ~70% (56K) | ~30% (CLAUDE.md, env) | 1:millions |
| Cursor | ~5K tokens | ~90%+ | <10% (.cursor/rules) | 1:millions |
| Copilot | undisclosed | Layer 1 (majority) | Layers 2+3 | ~3:20M |
| Claude.ai web | ~24K tokens | ~99% | <1% (datetime) | 1:millions |

**Key finding**: 1-3 base system prompt templates serve millions of users. The shared prefix (70-99% of tokens) is the primary caching opportunity. Personalization happens via appended context (CLAUDE.md, rules files, env), not by modifying the base prompt — preserving the prefix.

### Enterprise customization
Enterprises add custom instructions via CLAUDE.md, Cursor rules, Copilot instructions — but these are **appended after** the shared system prompt, preserving prefix caching.

## 2. Multi-Agent Context Sharing

### Subagent pattern (Claude Code)
- Each subagent gets a **fresh context window** — does NOT inherit parent conversation
- **System prompt is duplicated** for every subagent invocation (~56K shared tokens re-sent each time)
- Only the final message returns to parent. No shared KV state.
- Subagents cannot spawn other subagents (no nesting)

### Multi-step tool use
- Sequential tool calls within a conversation share the **entire conversation as growing prefix**
- Input-to-output ratio exceeds **100:1** in agentic workflows (Manus, llm-d benchmarks)
- Each tool call adds ~1-15K new tokens to a 40-80K existing context

### KVFlow (NeurIPS 2025)
- Workflow-aware prefix caching for multi-agent: **1.83x speedup** single workflow, **2.19x** concurrent
- Agents share fixed prompt prefixes; system context dominates

## 3. User Diversity

### Scale
- Copilot: 20M all-time users, 4.7M paid, 77K enterprise customers
- Large enterprise: Accenture 50K devs, Siemens 30K devs
- Typical deployment: hundreds to thousands of concurrent sessions

### Cross-user sharing rates (production data)
- CacheSolidarity (NDSS 2025): **30% cross-user prefix reuse** in real workloads
- Same-user reuse: 9-60% depending on workload type
- llm-d benchmark: 150 enterprises, 5 concurrent users each → prefix-aware scheduling 57x faster TTFT

## 4. Framework-Level Patterns

| Framework | Overhead | Shared Prefix |
|-----------|---------|---------------|
| CrewAI | ~3-4K tokens per workflow (3x LangChain) | Per-crew template, customizable |
| LangChain | <900 tokens | Minimal — PromptTemplate per chain |
| LlamaIndex | Variable | Per-index prompts |

**No significant cross-framework prompt overlap** — each framework has its own prompt architecture.

## 5. Production Data Points

- Anthropic treats **cache miss rate drops as production incidents**
- Cached reads = 10% of standard pricing ($0.30 vs $3.00/M tokens for Sonnet)
- BatchLLM: 58-92% token savings with prefix sharing
- MARCONI: 34.4x token hit rate improvement on SWEBench (agentic)
- llm-d: P90 TTFT 0.542s with prefix-aware scheduling (57x faster than random)

## Assumptions for Phase 2 Implementation

Based on the research, the simulator should model:

1. **System prompt sharing groups**: Users share system prompts within a "framework group" (e.g., all Claude Code users share the same 56K base prefix). Different frameworks have different base prompts.

2. **Sharing structure**:
   - **Tier 1 — Framework base** (70-90% of prompt): Identical across all users of that framework. 1-3 unique templates.
   - **Tier 2 — Workspace/project** (10-25%): CLAUDE.md, rules files. Shared within a team/project, unique across teams.
   - **Tier 3 — Session-specific** (5-10%): Conversation history, tool outputs. Unique per session.

3. **Subagent sharing**: Each subagent invocation duplicates the system prompt (Tier 1 + Tier 2). With block-level sharing, these can share physical KV blocks for the common prefix.

4. **Concurrent users per framework**: Model N users per framework group. All share Tier 1 blocks. Groups of M users share Tier 2 blocks.

5. **Sharing benefit**: With ref-counted blocks, N users × 56K shared prefix = 1 physical copy instead of N copies. Memory savings = (N-1) × 56K × bytes_per_token.

## Sources

- Claude Code system prompts repo (Piebald-AI)
- Cursor leaked system prompt (GitHub)
- Copilot agent mode prompt structure (dev.to)
- CacheSolidarity (NDSS 2025), BatchLLM, MARCONI (MLSys 2025)
- KVFlow (NeurIPS 2025), llm-d benchmarks
- Anthropic prompt caching docs, cache incident practice
