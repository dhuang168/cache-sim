#!/usr/bin/env python3
"""
Capture baseline metrics for a tagged version.

Usage:
    python scripts/capture_baseline_metrics.py [--config CONFIG] [--tag TAG] [--output DIR]

Defaults:
    --config configs/heavy_coding.json
    --tag    (reads from git)
    --output results/v0.1-prototype-baseline/
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import SimConfig
from sim.engine import SimEngine


def get_git_info():
    """Get current git tag and hash."""
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except subprocess.CalledProcessError:
        tag = None
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    return tag, commit


def run_and_capture(config_path: str) -> dict:
    """Run simulation and return metrics summary."""
    cfg = SimConfig.from_json(config_path)
    cfg.sim_start_time_s = 36000.0  # ensure traffic for all profiles

    m = SimEngine(cfg).run()
    r = m.report()

    completed = sum(m.savings_events.values())
    hit_rate = r.get("cache_hit_rate", {})
    combined_hit = 1.0 - hit_rate.get("miss", 1.0)

    all_ttft = []
    for v in m.ttft_us.values():
        all_ttft.extend(v)
    ttft_p50 = np.percentile(all_ttft, 50) / 1e6 if all_ttft else 0
    ttft_p95 = np.percentile(all_ttft, 95) / 1e6 if all_ttft else 0
    ttft_mean = np.mean(all_ttft) / 1e6 if all_ttft else 0

    recomp = np.mean(m.recompute_fraction) if m.recompute_fraction else 0

    prefill_mean = np.mean(m.prefill_duration_us) / 1e6 if m.prefill_duration_us else 0
    prefill_p95 = np.percentile(m.prefill_duration_us, 95) / 1e6 if m.prefill_duration_us else 0

    qw_mean = np.mean(m.queue_wait_us) / 1e6 if m.queue_wait_us else 0
    qw_p95 = np.percentile(m.queue_wait_us, 95) / 1e6 if m.queue_wait_us else 0

    slot_util = r.get("mean_slot_utilization_pct", 0)

    return {
        "completed": completed,
        "combined_hit_rate": round(combined_hit, 4),
        "hit_rate": {k: round(v, 4) for k, v in hit_rate.items()},
        "recompute_fraction_mean": round(recomp, 4),
        "ttft_mean_s": round(ttft_mean, 3),
        "ttft_p50_s": round(ttft_p50, 3),
        "ttft_p95_s": round(ttft_p95, 3),
        "prefill_mean_s": round(prefill_mean, 3),
        "prefill_p95_s": round(prefill_p95, 3),
        "queue_wait_mean_s": round(qw_mean, 3),
        "queue_wait_p95_s": round(qw_p95, 3),
        "slot_utilization_pct": round(slot_util, 1),
        "tier_saturation_pct": r.get("tier_saturation_pct", {}),
        "eviction_rate": r.get("eviction_rate_per_s", {}),
        "sharing_factor": r.get("sharing_factor", 1.0),
        "sim_duration_s": cfg.sim_duration_s,
        "warmup_s": cfg.warmup_s,
    }


def main():
    parser = argparse.ArgumentParser(description="Capture baseline metrics")
    parser.add_argument("--config", default="configs/heavy_coding.json")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--output", default="results/v0.1-prototype-baseline")
    args = parser.parse_args()

    git_tag, git_commit = get_git_info()
    tag = args.tag or git_tag or "unknown"

    print(f"Capturing baseline: tag={tag}, commit={git_commit[:8]}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}/")

    metrics = run_and_capture(args.config)

    summary = {
        "captured_at": datetime.datetime.now().isoformat(),
        "git_tag": tag,
        "git_commit": git_commit,
        "config": args.config,
        **metrics,
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "key_metrics.json"

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMetrics saved to {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
