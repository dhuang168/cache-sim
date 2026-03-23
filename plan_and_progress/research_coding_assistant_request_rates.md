# Research: Coding AI Assistant Request Rates and Interaction Frequency

Date: 2026-03-23

## Summary

This research gathers production usage data on how frequently developers interact with AI coding assistants, to inform the simulator's arrival rate model. The key distinction is between **human-paced** interactions (manual prompts with think time) and **agentic** interactions (automated tool loops with minimal human involvement).

---

## 1. Developer Active Coding Time (Baseline)

Before computing requests/hour, we need to know how much time developers actually spend coding:

- **52 minutes/day** median active coding time (editing code in IDE), from 250K+ developer telemetry (Software.com Code Time Report)
- **~4h 21min/week** total code editing time (Mon-Fri)
- Only **10% of developers** code more than 2 hours/day; **40%** code more than 1 hour/day
- Developers spend **32%** of their time writing/improving code, **35%** on maintenance/testing, **23%** on meetings

**Implication**: A "typical developer" has ~1 hour of active coding per day. Power users may have 2-4 hours. This bounds the denominator for "requests per hour of active coding."

---

## 2. GitHub Copilot

### Autocomplete (Tab Completions)
- **312 daily completions per user (DCPU)** average (ACM Communications study, GitHub telemetry)
- Acceptance rate: **27-30%** (so ~90 accepted completions/day out of 312 shown)
- 88% of accepted code retained in final submission
- Copilot now writes ~46% of average user's code

### Derived Rates
- If active coding = 1 hour/day: **312 completions/hour** (but most are auto-triggered, not user-initiated)
- If active coding = 2 hours/day: **156 completions/hour**
- These are **passive suggestions** (triggered by typing), not explicit user requests
- Each completion is a small request (few hundred tokens context, few lines response)
- Latency: **~200ms** for small completions, **2-5s** for complex multi-line

### Chat Requests (Copilot Chat)
- No published data on chat requests per hour
- Copilot Free: 2,000 completions/month + limited chat
- Copilot Pro: unlimited completions + 300 "premium requests"/month (~10/day)

### Inter-Request Time for Autocomplete
- Autocomplete fires on every keystroke pause (~300-500ms debounce)
- Effective rate: **one suggestion every 5-15 seconds** during active typing
- This is ~240-720 requests/hour during active coding bursts

---

## 3. Cursor

### Autocomplete (Tab)
- Cursor serves **billions of AI code completions daily** across all users
- Backend handles **>1 million QPS** primarily from autocomplete requests
- Latency: **95-140ms** for autocomplete (faster than Copilot)
- Free tier: 2,000 completions/month (~67/day)
- Pro tier: unlimited completions (no hard cap since June 2025)

### Chat/Agent Requests (Premium Requests)
- Pre-June 2025: **500 fast requests/month** on Pro plan (~17/day)
- Developers reported hitting 500 limit "very quickly" under intensive coding
- Post-June 2025: unlimited with rate limiting
- Each active developer uses Cursor for **~10 coding tasks/day** average

### Derived Rates
- **~10 tasks/day** with multiple requests per task
- If each task involves 3-5 chat/agent requests: **30-50 chat requests/day**
- Over 2 hours of active use: **15-25 chat requests/hour**
- Autocomplete runs continuously: similar to Copilot, **200-700/hour** during active typing

---

## 4. Claude Code

### Session Characteristics
- Average session: **~13-14 minutes** duration
- Average session token consumption: **500,000 - 1,000,000 tokens**
- Context starts at ~5,000 tokens, balloons to ~50,000 after 30 minutes
- Average cost: **$6/developer/day**, below $12 for 90% of users

### User Prompts Per Session
- Pro users: **10-40 prompts** per 5-hour window
- Max 5x users: ~50-200 prompts per window
- Max 20x users: ~200-800 prompts per window

### API Calls Per User Command (Agentic Amplification)
- A single user command can generate **8-12 API calls** within 60 seconds (e.g., "lint, fix, test, fix" cycle)
- Per-turn tool call limit: ~20 tool calls per response (recently reduced from 60-80+)
- A "seemingly simple" edit command consumes **50,000-150,000 tokens** per API call

### Derived Rates (Human-Paced)
- Pro user, 13-min session: ~10-40 prompts / 0.22 hours = **45-180 user prompts/hour**
- But each prompt triggers 2-12 API calls, so: **90-2,160 API calls/hour**
- Realistic median estimate: ~20 prompts/hour with ~4 API calls each = **~80 API calls/hour**

### Inter-Turn Arrival Time
- Human think time between prompts: reviewing output, deciding next step
- Estimated: **30-120 seconds** between user-initiated prompts
- But API calls within a single agentic loop: **5-15 seconds** apart

---

## 5. Agentic Coding (Autonomous Agent Mode)

### SWE-bench Agent Benchmarks
- Efficient agents (Vibekanban): **average <19 steps** per SWE-bench task
- Typical agents: **>40 steps** per task
- Step limit in evaluations: **80-250 turns** maximum
- SWE-bench Pro uses **250-turn limit**

### Claude Code Agent Loop
- Complex tasks: **dozens of actions** chained together with auto-correction
- Sessions can stretch across **45+ minutes** with 60-80+ tool calls (before recent limits)
- Current per-turn limit: ~20 tool calls per response
- Sub-agents can be spawned for parallel work

### Agentic Request Patterns
- **No human think time** between tool calls -- agent loops as fast as API allows
- API response time: **5-30 seconds** per call (depending on complexity)
- Tool call rate: **2-12 calls/minute** during active agent execution
- A single complex task might generate **50-200 API calls** over 10-30 minutes

### Tasks Per Hour (Agentic)
- Simple bug fixes: 5-15 minutes each = **4-12 tasks/hour**
- Complex feature implementations: 30-60 minutes each = **1-2 tasks/hour**
- Each task: 19-80+ agent steps

---

## 6. Summary Table: Request Rates by Mode

| Mode | User Actions/hr | API Calls/hr | Inter-Request (s) | Notes |
|------|----------------|-------------|-------------------|-------|
| **Copilot autocomplete** | N/A (passive) | 200-700 | 5-15 | Auto-triggered by typing |
| **Cursor autocomplete** | N/A (passive) | 200-700 | 5-15 | Auto-triggered by typing |
| **Copilot chat** | 5-15 | 5-15 | 240-720 | Manual, think-heavy |
| **Cursor chat/agent** | 15-25 | 30-75 | 50-240 | Semi-agentic |
| **Claude Code (human)** | 20-60 | 80-360 | 30-120 | Each prompt -> multi-call |
| **Claude Code (agent)** | 1-4 tasks | 200-800 | 5-15 | Autonomous loop |
| **SWE-bench agent** | N/A | 120-480 | 5-30 | Benchmark, no human |

---

## 7. Key Derived Parameters for Simulator

### For "coding" profile (human-paced IDE assistant)
- **Autocomplete requests**: ~300-500/hour during active coding (but these are tiny, <1KB KV)
- **Chat/edit requests**: ~15-30/hour (these create meaningful KV cache entries)
- **Effective KV-generating request rate**: ~20/hour per active developer
- **Inter-request time**: ~120-180 seconds (human think time dominates)

### For "agentic_coding" profile (agent mode)
- **API calls per task**: 20-80 (median ~40)
- **Tasks per hour**: 2-6
- **Effective API call rate**: 80-300/hour
- **Inter-request time**: 5-15 seconds (agent loop speed, no human think time)
- **KV cache reuse**: Very high -- agent revisits same context repeatedly within a task

### Ratio: Agentic vs Human
- Agentic generates **5-15x more API calls/hour** than human-paced coding
- But agentic has **much higher prefix cache hit potential** (same context, repeated queries)
- Session duration: agentic sessions run longer (30-60 min vs 13 min human)

---

## Sources

- [GitHub Copilot Usage Metrics - GitHub Docs](https://docs.github.com/en/copilot/concepts/copilot-usage-metrics/copilot-metrics)
- [Measuring GitHub Copilot's Impact on Productivity - ACM](https://cacm.acm.org/research/measuring-github-copilots-impact-on-productivity/)
- [How Cursor Serves Billions of AI Code Completions Every Day - ByteByteGo](https://blog.bytebytego.com/p/how-cursor-serves-billions-of-ai)
- [Claude Code Rate Limits Explained - SitePoint](https://www.sitepoint.com/claude-code-rate-limits-explained/)
- [Claude Code Rate Limits - Northflank](https://northflank.com/blog/claude-rate-limits-claude-code-pricing-cost)
- [Manage Costs Effectively - Claude Code Docs](https://code.claude.com/docs/en/costs)
- [How AI is Transforming Work at Anthropic](https://www.anthropic.com/research/how-ai-is-transforming-work-at-anthropic)
- [Scoring 71% on SWE-bench in Half the Steps - Vibekanban](https://www.vibekanban.com/blog/scoring-71-on-swe-bench-verified-in-half-the-steps)
- [Code Time Report - Software.com](https://www.software.com/reports/code-time-report)
- [SWE-bench Pro - Scale AI](https://static.scale.com/uploads/654197dc94d34f66c0f5184e/SWEAP_Eval_Scale%20(9).pdf)
- [Cursor Plans - Cursor Docs](https://docs.cursor.com/account/plans-and-usage)
- [GitHub Copilot Statistics 2026 - QuantumRun](https://www.quantumrun.com/consulting/github-copilot-statistics/)
- [Claude Code Token Limits - Faros AI](https://www.faros.ai/blog/claude-code-token-limits)
