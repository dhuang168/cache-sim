# Sanity Check Experiments — Plan

Date: 2026-03-22

## Experiments

### A: Block size comparison (5 min, 1 worker, peak=5)
Sweep block_size_tokens = [0, 16, 256, 4096] with sharing + LRU

### B: Sharing + global vs local L3A (5 min, 4 workers, peak=3)
2×2: sharing on/off × global/local L3A, all with LRU

### C: Push vs Pull (5 min, 4 workers, peak=3)
Push vs pull dispatch, global L3A + LRU + sharing

## Interim reports after each experiment group
- docs/experiment_a_block_size_report.md
- docs/experiment_b_sharing_l3a_report.md
- docs/experiment_c_push_pull_report.md
