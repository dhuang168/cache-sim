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

| File | Purpose |
|------|---------|
| `CLAUDE_SNIPPET.md` | Drop-in CLAUDE.md rules (compact, always loaded) |
| `methodology_full.md` | Complete guide with examples (reference doc) |
| `memory/*.md` | Individual memory files for auto-loading |
| `memory/MEMORY_ENTRIES.md` | Index entries to add to MEMORY.md |
