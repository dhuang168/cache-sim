# AgentSim Developer Onboarding
# Migration branch already created — start here

---

## What Is in `files/`

The `files/` folder is **flat** — all documents and code artifacts sit
directly at `files/`, not in subdirectories.

```
cache-sim/files/
│
│  ── PLANNING DOCUMENTS ──────────────────────────────────────────
├── MASTER_INDEX.md              ← index of every document (read this first)
├── agentsim_final_plan.md       ← PRIMARY PLAN (read this second)
├── cache_sim_synthesis.md       ← why the plan looks the way it does
├── gap_analysis.md              ← what changed from the prototype and why
├── agentsim_design.md           ← SUPERSEDED (historical only)
├── simulator_architecture.md   ← SUPERSEDED (historical only)
├── simulator_architecture_v2.md ← SUPERSEDED (historical only)
│
│  ── CODE ARTIFACTS ─────────────────── destination in repo ──────
├── contracts.py      →  agentsim/core/contracts.py           (Phase 0 — copy first)
├── oracle.py         →  agentsim/core/des/oracle.py          (Phase 1 starter)
├── profiles.py       →  agentsim/integration/chips/profiles.py  (Phase 3 starter)
├── request_sweep.py  →  agentsim/sweep/request_sweep.py      (sweep tool)
├── events.py         →  agentsim/core/events.py              (pending Phase 2 move)
├── hardware_model.py →  agentsim/core/hardware_model.py      (DEPRECATED)
├── session_model.py  →  agentsim/core/session_model.py       (DEMOTED)
└── request_sim.py    →  agentsim/sim/request_sim.py          (DEPRECATED)
```

---

## Before You Touch Anything — Read These Two Files

Both are in `files/`. Read them in this order:

1. `files/MASTER_INDEX.md` — complete map of every document and its status
2. `files/agentsim_final_plan.md` — the authoritative execution plan

Pay specific attention to:
- The 10 Guiding Principles (non-negotiable)
- The three-layer architecture diagram
- All phase gate checklists
- The old code lifecycle: Deprecate → Disconnect → Delete

---

## Current Repo State

The migration branch `feature/des-core-swap` is already created and
checked out. `v0.1-prototype` is already tagged. Skip directly to Step 5.

---

## Step 5 — Copy Files from `files/` into the Repo

Run all commands from `cache-sim/` root.

### 5a. Create the agentsim package skeleton

```bash
mkdir -p agentsim/core/des
mkdir -p agentsim/core/observation
mkdir -p agentsim/integration/chips
mkdir -p agentsim/integration/adapters
mkdir -p agentsim/sweep
mkdir -p agentsim/sim

touch agentsim/__init__.py
touch agentsim/core/__init__.py
touch agentsim/core/des/__init__.py
touch agentsim/core/observation/__init__.py
touch agentsim/integration/__init__.py
touch agentsim/integration/chips/__init__.py
touch agentsim/integration/adapters/__init__.py
touch agentsim/sweep/__init__.py
touch agentsim/sim/__init__.py
```

### 5b. Copy contracts.py — do this first, everything else depends on it

```bash
cp files/contracts.py agentsim/core/contracts.py
```

This is the single most important file. It defines every interface
boundary across all three layers. Nothing else should be built until
this file is reviewed and approved by the team.

### 5c. Copy the Phase 1 oracle starter

```bash
cp files/oracle.py agentsim/core/des/oracle.py
```

Replaces the roofline model in `hardware_model.py` with piecewise-linear
interpolation from benchmark tables, plus roofline as `ANALYTICAL_ONLY`
fallback. Review against `contracts.py` before extending.

### 5d. Copy the integration layer starter

```bash
cp files/profiles.py agentsim/integration/chips/profiles.py
```

Replaces `hardware_model.py`'s `ChipProfile` with a byte-based version.
`page_size_tokens` is removed — use `TierSpec.block_size_bytes` instead.

### 5e. Copy the sweep tool

```bash
cp files/request_sweep.py agentsim/sweep/request_sweep.py
```

SimPy sweep tool. Non-authoritative. All outputs labeled `"sweep-estimate"`.
SimPy cannot own cache state — constraints enforced in `contracts.py`.

### 5f. Copy deprecated files — keep them, they still run

These are deprecated but remain functional during the migration so
nothing breaks while the new engine is being built.

```bash
cp files/hardware_model.py agentsim/core/hardware_model.py
cp files/session_model.py  agentsim/core/session_model.py
cp files/request_sim.py    agentsim/sim/request_sim.py
```

### 5g. Copy events.py — moves to observation/ in Phase 2

```bash
cp files/events.py agentsim/core/events.py
```

Content is correct and preserved from the original prototype. Location
changes in Phase 2 when it moves to `agentsim/core/observation/events.py`
and the mappers are refactored to implement `ObserverBase`.

### 5h. Create des/README.md

```bash
cat > agentsim/core/des/README.md << 'EOF'
# Core DES Layer

This is the only authoritative simulation engine in AgentSim.
All other components treat it as the source of truth.

For all interface definitions, see ../contracts.py.
EOF
```

---

## Step 6 — Update pyproject.toml

Open `pyproject.toml` and ensure it picks up both the old `sim` package
and the new `agentsim` package during the migration:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["sim*", "agentsim*"]
```

Also add `import-linter` to dev dependencies if not already present:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7",
    "import-linter",
    # ... your existing dev deps
]
```

---

## Step 7 — Set Up Import Enforcement

Create `.importlinter` at repo root:

```ini
[importlinter]
root_package = agentsim

[importlinter:contract:no-old-modules-in-new-code]
name = New code must not import from deprecated modules
type = forbidden
source_modules =
    agentsim.core.des
    agentsim.core.observation
    agentsim.integration
    agentsim.sweep
forbidden_modules =
    agentsim.core.hardware_model
    agentsim.core.session_model
    agentsim.sim.request_sim

[importlinter:contract:observation-layer-is-downstream-only]
name = Observation layer must not import from integration or sweep
type = forbidden
source_modules =
    agentsim.core.observation
forbidden_modules =
    agentsim.integration
    agentsim.sweep

[importlinter:contract:des-core-has-no-upward-dependencies]
name = Core DES must not import from observation, integration, or sweep
type = forbidden
source_modules =
    agentsim.core.des
forbidden_modules =
    agentsim.core.observation
    agentsim.integration
    agentsim.sweep
```

---

## Step 8 — Verify Everything Works

```bash
pip install -e ".[dev]"

# New package is importable
python -c "from agentsim.core.contracts import CacheKey; print('agentsim OK')"

# Old package still works — nothing broken
python -c "from sim.engine import SimEngine; print('sim OK')"

# Old tests still pass
pytest tests/ -v

# Import boundaries enforced
lint-imports
```

All four must pass before proceeding to Phase 0.

---

## Step 9 — Commit the Setup

```bash
git add agentsim/ .importlinter pyproject.toml
git commit -m "chore: add agentsim package skeleton and import boundary enforcement"
git push origin feature/des-core-swap
```

---

## Step 10 — Pre-Phase 0 Checklist

```
[ ] pytest tests/ -v passes (all existing tests green)
[ ] python -c "from agentsim.core.contracts import CacheKey" prints OK
[ ] python -c "from sim.engine import SimEngine" prints OK
[ ] lint-imports passes
```

When all four boxes are checked, proceed to Phase 0.

---

## Phase 0 — What To Do Next

**Read:** `files/agentsim_final_plan.md`, section "Phase 0 — Architecture Contracts"

**Goal:** Review and freeze `agentsim/core/contracts.py` with the team.
No porting of cache-sim code begins until this file is approved.

**Tasks in order:**

1. Open `agentsim/core/contracts.py` and read every class and docstring
2. Verify all of the following are present and correctly defined:
   - `CacheKey` — exact prefix identity, not token-count proximity
   - `SavingsEvent` — with `classify()` classmethod
   - `TierSpec` — byte-based, no token fields anywhere
   - `CacheObject` — `size_bytes` authoritative, `token_count` metadata only
   - `RequestResult` — `prefill_latency_us` and `decode_latency_us` separated
   - `ObserverBase` — time-coupling constraints documented and enforced
   - `SweepEstimator` — explicit list of what SimPy is prohibited from doing
   - `ReportMetadata` — `confidence` appears in the `headline` property
3. Write `tests/test_phase0_contracts.py` — see the Phase 0 gate in
   `files/agentsim_final_plan.md` for exactly what the tests must cover
4. Run:
   ```bash
   pytest tests/test_phase0_contracts.py -v
   ```
5. Get team sign-off before moving to Phase 0.5

**Phase 0 gate (all must pass before Phase 0.5):**
```
[ ] contracts.py reviewed and approved by team
[ ] No ambiguity about what Layer 1 vs Layer 2 owns
[ ] SimPy constraints understood by all contributors
[ ] core/des/README.md exists and points to contracts.py
[ ] lint-imports active in CI
[ ] test_phase0_contracts.py written and passing
```

---

## Quick Reference: File Status at Start

| Destination in repo | Source in files/ | Status | Active in phase |
|---|---|---|---|
| `agentsim/core/contracts.py` | `contracts.py` | ✅ Review now | 0 |
| `agentsim/core/des/oracle.py` | `oracle.py` | 🔜 Starter | 1 |
| `agentsim/integration/chips/profiles.py` | `profiles.py` | 🔜 Starter | 3 |
| `agentsim/sweep/request_sweep.py` | `request_sweep.py` | 🔜 Sweep tool | 1 |
| `agentsim/core/events.py` | `events.py` | ⚠️ Move to observation/ | 2 |
| `agentsim/core/hardware_model.py` | `hardware_model.py` | ⚠️ Deprecated | Delete phase 2/3 |
| `agentsim/core/session_model.py` | `session_model.py` | ⚠️ Demoted | Delete phase 2/3 |
| `agentsim/sim/request_sim.py` | `request_sim.py` | ⚠️ Deprecated | Delete phase 2/3 |
| `sim/` (entire folder) | unchanged | ✅ Still runs | Delete after phase 1 merge |

---

## For Coding Agents: Strict Execution Order

Do not skip steps. Do not proceed past a gate without all checkboxes passing.
Gates are hard stops — bugs found later cost 10× more to fix.

```
1.  Read files/agentsim_final_plan.md completely
2.  Read files/MASTER_INDEX.md
3.  Execute Step 5  — copy files into correct repo locations
4.  Execute Step 6  — update pyproject.toml
5.  Execute Step 7  — create .importlinter at repo root
6.  Execute Step 8  — verify all four checks pass
7.  Execute Step 9  — commit setup to feature/des-core-swap
8.  Complete Step 10 checklist (4 boxes)
9.  Execute Phase 0  — review contracts.py, write tests, get approval
10. Pass Phase 0 gate (6 boxes in agentsim_final_plan.md)
11. Execute Phase 0.5 — write golden benchmark files and tolerance tests
12. Pass Phase 0.5 gate (4 boxes)
13. Execute Phase 1  — port cache-sim DES core into agentsim/core/des/
14. Pass Phase 1 gate (11 boxes) — only then merge to main
15. Execute Phase 2, 3, 4 in order, passing each gate before the next
```

The complete gate checklists for every phase are in
`files/agentsim_final_plan.md`. That file is the ground truth.
