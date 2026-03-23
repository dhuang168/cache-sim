#!/usr/bin/env python3
"""
Disaggregated P/D + LMCache study: extending heavy_coding_report.

Compares legacy colocated vs disaggregated (N-1):1 worker P:D vs disagg+LMCache,
across 4/8/12 worker clusters (each worker = 8 GPUs).

P:D ratios (by worker): 3:1 (4W), 7:1 (8W), 11:1 (12W)
Arrival rate scales with cluster size: peak=3/6/9

Git hash: 49f075af44aadd9cff76e8007b805bf93e8d0c16 (Phase 2 complete)
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import SimConfig
from sim.engine import SimEngine

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "heavy_coding.json")

SIM_DURATION = 600.0   # 10 min (reduced for chunk-mode feasibility at scale)
WARMUP = 60.0          # 1 min
EPOCH_INTERVAL = 30.0
GPUS_PER_WORKER = 8

# Cluster configs: (n_workers, n_decode_workers, arrival_peak)
CLUSTERS = [
    (4,  1, 3),   # 4 workers: 3 prefill + 1 decode, peak=3
    (8,  1, 6),   # 8 workers: 7 prefill + 1 decode, peak=6
    (12, 1, 9),   # 12 workers: 11 prefill + 1 decode, peak=9
]


def make_config(
    n_workers: int,
    arrival_peak: float,
    l3a_shared: bool = True,
    disaggregated: bool = False,
    n_decode_workers: int = 0,
    chunk_dedup: bool = False,
    label: str = "",
) -> SimConfig:
    cfg = SimConfig.from_json(CONFIG_PATH)
    cfg.sim_duration_s = SIM_DURATION
    cfg.warmup_s = WARMUP
    cfg.epoch_report_interval_s = EPOCH_INTERVAL
    cfg.sim_start_time_s = 36000.0  # 10 AM

    # Scale arrival rate with cluster size
    for p in cfg.profiles:
        p.arrival_rate_peak = arrival_peak

    # Topology: each worker = 8 GPUs
    cfg.service.n_gpus_per_worker = GPUS_PER_WORKER

    if disaggregated:
        n_prefill_workers = n_workers - n_decode_workers
        n_prefill_gpus = n_prefill_workers * GPUS_PER_WORKER
        n_decode_gpus = n_decode_workers * GPUS_PER_WORKER
        cfg.service.n_prefill_nodes = n_prefill_gpus
        cfg.service.disaggregated = True
        cfg.service.n_decode_nodes = n_decode_gpus
        cfg.service.kv_transfer_bandwidth_bytes_per_s = 50_000_000_000  # 50 GB/s RDMA
        cfg.service.kv_transfer_latency_floor_us = 2000                 # 2ms
        cfg.service.prefill_latency_multiplier = 0.85
        cfg.service.decode_batch_fill_factor = 0.85
    else:
        n_total_gpus = n_workers * GPUS_PER_WORKER
        cfg.service.n_prefill_nodes = n_total_gpus
        cfg.service.disaggregated = False

    cfg.service.l3a_shared = l3a_shared
    cfg.service.dispatch_algorithm = "pull"

    # LMCache chunk dedup
    if chunk_dedup:
        cfg.cache.block_size_tokens = 256
        cfg.cache.eviction_policy = "lru"
        cfg.cache.deduplication = "chunk"
        cfg.cache.tier_migration = "demand_pull"

    cfg.run_id = label
    return cfg


def run_and_report(cfg, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}", flush=True)

    engine = SimEngine(cfg)
    m = engine.run()
    r = m.report()

    completed = sum(m.savings_events.values())
    hit_rate = r.get("cache_hit_rate", {})
    tier_sat = r.get("tier_saturation_pct", {})

    # Combined hit rate (L1+L2+L3A)
    combined_hit = 1.0 - hit_rate.get("miss", 1.0)

    all_ttft = []
    for v in m.ttft_us.values():
        all_ttft.extend(v)

    ttft_mean = np.mean(all_ttft) / 1000 if all_ttft else 0
    ttft_p50 = np.percentile(all_ttft, 50) / 1000 if all_ttft else 0
    ttft_p95 = np.percentile(all_ttft, 95) / 1000 if all_ttft else 0

    prefill_mean = np.mean(m.prefill_duration_us) / 1000 if m.prefill_duration_us else 0
    prefill_p95 = np.percentile(m.prefill_duration_us, 95) / 1000 if m.prefill_duration_us else 0

    qw_mean = np.mean(m.queue_wait_us) / 1000 if m.queue_wait_us else 0
    qw_p95 = np.percentile(m.queue_wait_us, 95) / 1000 if m.queue_wait_us else 0

    slot_util = r.get("mean_slot_utilization_pct", 0)

    # Recompute fraction
    recomp_mean = np.mean(m.recompute_fraction) if m.recompute_fraction else 0
    recomp_p95 = np.percentile(m.recompute_fraction, 95) if m.recompute_fraction else 0

    # Throughput
    eff_s = max(1, m.effective_sim_us) / 1_000_000
    throughput = completed / eff_s

    xfer_mean = np.mean(m.kv_transfer_us) / 1000 if m.kv_transfer_us else 0
    xfer_p95 = np.percentile(m.kv_transfer_us, 95) / 1000 if m.kv_transfer_us else 0

    dedup = r.get("chunk_dedup", {})
    promotions = m.tier_promotions

    print(f"  Completed:     {completed:,}")
    print(f"  Throughput:    {throughput:.1f} req/s")
    print(f"  Hit rate:      combined={combined_hit:.3f}  L1={hit_rate.get('L1',0):.3f} L2={hit_rate.get('L2',0):.3f} L3A={hit_rate.get('L3A',0):.3f} miss={hit_rate.get('miss',0):.4f}")
    print(f"  Recompute:     mean={recomp_mean:.3f}  p95={recomp_p95:.3f}")
    print(f"  TTFT:          mean={ttft_mean:.0f}ms  p50={ttft_p50:.0f}ms  p95={ttft_p95:.0f}ms")
    print(f"  Prefill:       mean={prefill_mean:.0f}ms  p95={prefill_p95:.0f}ms")
    print(f"  Queue wait:    mean={qw_mean:.0f}ms  p95={qw_p95:.0f}ms")
    print(f"  Slot util:     {slot_util:.1f}%")
    print(f"  Tier sat:      L1={tier_sat.get('L1',0):.1f}%  L2={tier_sat.get('L2',0):.1f}%  L3A={tier_sat.get('L3A',0):.1f}%")
    if m.kv_transfer_us:
        print(f"  KV transfer:   mean={xfer_mean:.0f}ms  p95={xfer_p95:.0f}ms  (n={len(m.kv_transfer_us)})")
    if dedup:
        print(f"  Chunk dedup:   ratio={dedup.get('dedup_ratio',0):.3f}  novel={dedup.get('novel_chunks',0)}  hits={dedup.get('dedup_hits',0)}")
    if promotions > 0:
        print(f"  Promotions:    {promotions}")
    sys.stdout.flush()

    return {
        "label": label,
        "completed": completed,
        "throughput": round(throughput, 1),
        "hit_rate": hit_rate,
        "combined_hit": round(combined_hit, 4),
        "recompute_mean": round(recomp_mean, 4),
        "recompute_p95": round(recomp_p95, 4),
        "ttft_mean_ms": round(ttft_mean),
        "ttft_p50_ms": round(ttft_p50),
        "ttft_p95_ms": round(ttft_p95),
        "prefill_mean_ms": round(prefill_mean),
        "prefill_p95_ms": round(prefill_p95),
        "qw_mean_ms": round(qw_mean),
        "qw_p95_ms": round(qw_p95),
        "slot_util_pct": round(slot_util, 1),
        "tier_sat": tier_sat,
        "kv_xfer_mean_ms": round(xfer_mean) if m.kv_transfer_us else None,
        "kv_xfer_p95_ms": round(xfer_p95) if m.kv_transfer_us else None,
        "dedup_ratio": dedup.get("dedup_ratio") if dedup else None,
        "promotions": promotions,
    }


def main():
    results = []

    for n_workers, n_decode_workers, peak in CLUSTERS:
        n_prefill_workers = n_workers - n_decode_workers
        total_gpus = n_workers * GPUS_PER_WORKER
        prefill_gpus = n_prefill_workers * GPUS_PER_WORKER
        decode_gpus = n_decode_workers * GPUS_PER_WORKER

        print(f"\n\n{'#'*70}")
        print(f"  CLUSTER: {n_workers} workers ({total_gpus} GPUs), peak={peak}")
        print(f"  Disagg: {n_prefill_workers}W prefill ({prefill_gpus} GPUs) + {n_decode_workers}W decode ({decode_gpus} GPUs)")
        print(f"{'#'*70}")

        for l3a_mode in ["global", "local"]:
            l3a_shared = (l3a_mode == "global")

            # 1. Legacy colocated
            cfg = make_config(
                n_workers=n_workers, arrival_peak=peak,
                l3a_shared=l3a_shared, disaggregated=False,
                label=f"Legacy {n_workers}W {l3a_mode}",
            )
            results.append(run_and_report(cfg, f"Legacy {n_workers}W {l3a_mode}"))

            # 2. Disaggregated (per-session objects)
            cfg = make_config(
                n_workers=n_workers, arrival_peak=peak,
                l3a_shared=l3a_shared, disaggregated=True,
                n_decode_workers=n_decode_workers,
                label=f"Disagg {n_prefill_workers}P:{n_decode_workers}D {l3a_mode}",
            )
            results.append(run_and_report(cfg, f"Disagg {n_prefill_workers}P:{n_decode_workers}D {l3a_mode}"))

            # 3. Disaggregated + LMCache
            cfg = make_config(
                n_workers=n_workers, arrival_peak=peak,
                l3a_shared=l3a_shared, disaggregated=True,
                n_decode_workers=n_decode_workers, chunk_dedup=True,
                label=f"Disagg+LMC {n_prefill_workers}P:{n_decode_workers}D {l3a_mode}",
            )
            results.append(run_and_report(cfg, f"Disagg+LMC {n_prefill_workers}P:{n_decode_workers}D {l3a_mode}"))

    # Summary table
    print("\n\n" + "=" * 160)
    print("SUMMARY TABLE")
    print("=" * 160)
    hdr = f"{'Config':<35} {'Compl':>7} {'Tput':>5} {'Hit%':>5} {'Recomp':>6} {'TTFT p50':>9} {'TTFT p95':>9} {'QW p95':>8} {'Prefill':>8} {'Miss%':>5} {'Dedup':>5}"
    print(hdr)
    print("-" * 160)
    for r in results:
        dedup = f"{r['dedup_ratio']:.2f}" if r['dedup_ratio'] is not None else "  —"
        miss = r['hit_rate'].get('miss', 0) * 100
        hit = r['combined_hit'] * 100
        recomp = r['recompute_mean'] * 100
        print(f"{r['label']:<35} {r['completed']:>7,} {r['throughput']:>5.1f} {hit:>4.1f}% {recomp:>5.1f}% {r['ttft_p50_ms']:>8,}ms {r['ttft_p95_ms']:>8,}ms {r['qw_p95_ms']:>7,}ms {r['prefill_mean_ms']:>7,}ms {miss:>4.1f}% {dedup:>5}")

    # Save JSON
    out_path = Path(__file__).resolve().parent.parent / "results" / "disagg_lmcache_study.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
