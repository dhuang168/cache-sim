"""Profile a single simulation run to identify bottlenecks."""
import cProfile
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import SimConfig
from sim.engine import SimEngine

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "default.json")


def short_run():
    cfg = SimConfig.from_json(CONFIG_PATH)
    cfg.sim_duration_s = 60.0
    cfg.warmup_s = 5.0
    cfg.epoch_report_interval_s = 5.0
    # Use stressed config to match sanity tests
    cfg.tiers[0].capacity_bytes = 500 * 1024**2
    cfg.tiers[1].capacity_bytes = 10 * 1024**3
    cfg.tiers[2].capacity_bytes = 50 * 1024**3
    cfg.ttl_l2_s = 20.0
    cfg.ttl_l3a_s = 120.0
    cfg.eviction_hbm_threshold = 0.6
    cfg.eviction_ram_threshold = 0.8
    engine = SimEngine(cfg)
    return engine.run()


# Wall-clock timing
t0 = time.perf_counter()
short_run()
t1 = time.perf_counter()
print(f"\nWall-clock: {t1 - t0:.2f}s\n")

# cProfile
profiler = cProfile.Profile()
profiler.enable()
short_run()
profiler.disable()

stats = pstats.Stats(profiler)
stats.strip_dirs()
print("=" * 80)
print("Top 30 by cumulative time:")
print("=" * 80)
stats.sort_stats("cumulative")
stats.print_stats(30)

print("=" * 80)
print("Top 30 by total time (self):")
print("=" * 80)
stats.sort_stats("tottime")
stats.print_stats(30)
