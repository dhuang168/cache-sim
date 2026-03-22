# sim-dev-methodology

A portable Claude Code methodology skill for simulation and systems development. Distilled from building a multi-node KV cache simulator.

## Installation

### Option 1: Global (all projects)

Copy the CLAUDE.md snippet to your global config:

```bash
cat skills/sim-dev-methodology/CLAUDE_SNIPPET.md >> ~/.claude/CLAUDE.md
```

### Option 2: Per-project

Copy to your project root:

```bash
cp skills/sim-dev-methodology/CLAUDE_SNIPPET.md ./CLAUDE_METHODOLOGY.md
# Then add to your CLAUDE.md:
echo "See CLAUDE_METHODOLOGY.md for development methodology." >> CLAUDE.md
```

### Option 3: Memory files (strongest — auto-loaded every conversation)

```bash
cp skills/sim-dev-methodology/memory/*.md ~/.claude/projects/<your-project>/memory/
# Update MEMORY.md index
cat skills/sim-dev-methodology/memory/MEMORY_ENTRIES.md >> ~/.claude/projects/<your-project>/memory/MEMORY.md
```

## What's Included

Three layers, from lightest to most comprehensive:

| Layer | File(s) | When Loaded | Purpose |
|-------|---------|------------|---------|
| **Rules** | `CLAUDE_SNIPPET.md` | Every prompt (via CLAUDE.md) | Compact rules: workflow, core practices, checklists (~60 lines) |
| **Memory** | `memory/*.md` (6 files) | Every conversation (auto-loaded) | Individual skill areas with "why" and "how to apply" |
| **Reference** | `methodology_full.md` | On demand (read when needed) | Complete 8-section guide with concrete examples from a real project |

Use all three for maximum effect, or just the CLAUDE_SNIPPET for lightweight adoption.
