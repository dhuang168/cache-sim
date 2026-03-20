"""
Sensitivity sweep runner.
Vary one parameter at a time over ±50% of nominal value.
Runs are independent -> use multiprocessing.Pool.

Usage:
    python scripts/sweep.py --config configs/default.json --output results/sweep.json
"""
from __future__ import annotations
import argparse
import json
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import SimConfig
from sim.engine import SimEngine

SWEEP_PARAMS = {
    "ttl_l2_s": [150, 225, 300, 375, 450],
    "tiers[1].capacity_bytes": [2e12, 3e12, 4e12, 5e12, 6e12],
    "tiers[1].block_size_bytes": [8e6, 16e6, 32e6, 64e6],
    "service.n_prefill_slots": [16, 24, 32, 48, 64],
}

TARGET_METRICS = ["ttft_p95", "ttft_p99", "gpu_utilization", "sharing_factor"]


def set_nested(config: SimConfig, param_path: str, value) -> None:
    """Apply a dot/bracket-notation parameter override to a config."""
    import re
    parts = re.split(r'\.|\[|\]', param_path)
    parts = [p for p in parts if p]

    obj = config
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        elif hasattr(obj, part):
            obj = getattr(obj, part)
        elif isinstance(obj, (list, tuple)):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)

    final = parts[-1]
    if isinstance(obj, list) and final.isdigit():
        obj[int(final)] = type(obj[int(final)])(value)
    else:
        current = getattr(obj, final)
        setattr(obj, final, type(current)(value))


def run_single(args: tuple) -> dict:
    config_path, param_path, param_value = args
    config = SimConfig.from_json(config_path)
    set_nested(config, param_path, param_value)
    config.run_id = f"sweep-{param_path}-{param_value}"
    metrics = SimEngine(config).run()
    return {
        "param": param_path,
        "value": param_value,
        "metrics": metrics.report(),
    }


def main():
    parser = argparse.ArgumentParser(description="Sensitivity sweep runner")
    parser.add_argument("--config", required=True, help="Path to base config JSON")
    parser.add_argument("--output", default="results/sweep.json", help="Output path")
    parser.add_argument("--workers", type=int, default=None, help="Pool size")
    args = parser.parse_args()

    jobs = [
        (args.config, param, value)
        for param, values in SWEEP_PARAMS.items()
        for value in values
    ]

    print(f"Running {len(jobs)} sweep configurations...")

    with multiprocessing.Pool(args.workers) as pool:
        results = pool.map(run_single, jobs)

    # Compute elasticity for each param-metric pair
    elasticity = {}
    for param in SWEEP_PARAMS:
        param_results = [r for r in results if r["param"] == param]
        if len(param_results) < 2:
            continue
        param_results.sort(key=lambda r: r["value"])
        elasticity[param] = {}
        for metric_key in TARGET_METRICS:
            vals = []
            for pr in param_results:
                report = pr["metrics"]
                if metric_key == "ttft_p95":
                    v = report.get("ttft_ms", {}).get("cold_miss", {}).get("p95", 0)
                elif metric_key == "ttft_p99":
                    v = report.get("ttft_ms", {}).get("cold_miss", {}).get("p99", 0)
                elif metric_key == "sharing_factor":
                    v = report.get("sharing_factor", 1.0)
                else:
                    v = 0
                vals.append(v)
            if len(vals) >= 2 and vals[0] != 0:
                pct_change_metric = (vals[-1] - vals[0]) / abs(vals[0]) if vals[0] else 0
                pct_change_param = (param_results[-1]["value"] - param_results[0]["value"]) / abs(param_results[0]["value"]) if param_results[0]["value"] else 0
                elasticity[param][metric_key] = (
                    pct_change_metric / pct_change_param if pct_change_param else 0
                )

    output = {
        "results": results,
        "elasticity": elasticity,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results written to {args.output}")
    print("\nElasticity summary:")
    for param, metrics in elasticity.items():
        print(f"  {param}:")
        for metric, e in metrics.items():
            print(f"    {metric}: {e:.4f}")


if __name__ == "__main__":
    main()
