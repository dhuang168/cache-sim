# Realistic Config — Plan #1: Hardware-Accurate Capacities

Date: 2026-03-21

## Config semantics change

L3A capacity in config now means **per-worker SSD capacity** (physical hardware per server):
- **Global L3A**: total capacity = `l3a_capacity × n_workers` (pool all SSDs)
- **Local L3A**: per-worker capacity = `l3a_capacity` (own SSD only)

Previously: config value = total, local = total / n_workers. This was wrong — each server has its own SSD regardless of mode.

## Realistic hardware values

| Tier | Config Field | Value | Meaning |
|------|-------------|-------|---------|
| L1 (HBM) | `tiers[0].capacity_bytes` | 80 GB | Per-GPU (H100) |
| L2 (DRAM) | `tiers[1].capacity_bytes` | 1 TB | Per-worker host DRAM |
| L3A (SSD) | `tiers[2].capacity_bytes` | 8 TB | Per-worker NVMe SSD |

## Effective capacities at scale

| Workers | L1 Total | L2 Total | Global L3A | Local L3A/worker |
|---------|---------|---------|-----------|-----------------|
| 1 | 640 GB | 1 TB | 8 TB | 8 TB |
| 2 | 1.28 TB | 2 TB | 16 TB | 8 TB |
| 4 | 2.56 TB | 4 TB | 32 TB | 8 TB |
| 8 | 5.12 TB | 8 TB | 64 TB | 8 TB |

## Code changes

1. `sim/engine.py`: Global L3A capacity = `l3a_cfg.capacity_bytes × n_workers`
2. `sim/engine.py`: Local L3A capacity = `l3a_cfg.capacity_bytes` (no division)
3. `configs/default.json`: Update L2 to 1TB, L3A to 8TB per worker
4. Fix single-worker global L3A penalty (no remote penalty when n_workers=1)
5. Update stressed_config() in sanity_plots for visible effects in short sims

## Phase sequence
1. Code + config changes
2. Tests + traffic mix verification
3. Regenerate plots
4. Docs, crosscheck, commit
