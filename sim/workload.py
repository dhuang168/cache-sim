from __future__ import annotations
import math
import numpy as np

from sim.config import SimConfig, WorkloadProfile

# Pre-computed constants
_TWO_PI_OVER_86400 = 2.0 * math.pi / 86400.0


class WorkloadSynthesizer:
    """
    Generates request arrival events from configured session profiles.
    Uses non-homogeneous Poisson process with sinusoidal diurnal rate.
    """

    def __init__(self, config: SimConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self._start_time_s = config.sim_start_time_s
        # Pre-compute per-profile rate constants
        self._rate_cache: dict[str, tuple[float, float]] = {}
        for p in config.profiles:
            mean_rate = p.arrival_rate_peak / p.diurnal_peak_trough_ratio
            amplitude = mean_rate * (p.diurnal_peak_trough_ratio - 1) / 2
            self._rate_cache[p.name] = (mean_rate, amplitude)

    def diurnal_rate(self, time_s: float, profile: WorkloadProfile) -> float:
        """
        Rate function for NHPP.
        Sinusoidal with period 86400s, peak at 9 AM (offset 32400s).
        """
        mean_rate, amplitude = self._rate_cache[profile.name]
        phase = _TWO_PI_OVER_86400 * (time_s + self._start_time_s - 32400.0)
        return max(0.0, mean_rate + amplitude * math.sin(phase))

    def sample_iat(self, profile: WorkloadProfile) -> float:
        """Inter-arrival time within a session, in seconds."""
        if profile.iat_dist == "exponential":
            return float(self.rng.exponential(profile.iat_mean_s))
        elif profile.iat_dist == "lognormal":
            sigma = 0.8
            mu = np.log(profile.iat_mean_s) - 0.5 * sigma ** 2
            return float(self.rng.lognormal(mu, sigma))
        raise ValueError(f"Unknown iat_dist: {profile.iat_dist}")

    def sample_input_length(self, profile: WorkloadProfile) -> int:
        """Tokens, log-normal."""
        mu = np.log(profile.input_len_mean_tokens)
        sigma = profile.input_len_sigma_tokens / profile.input_len_mean_tokens
        return max(1, int(self.rng.lognormal(mu, sigma)))

    def sample_output_length(self, profile: WorkloadProfile) -> int:
        """Tokens, Pareto-tailed. Direct numpy instead of scipy.stats.pareto."""
        # Pareto: x = xmin / U^(1/alpha), where U ~ Uniform(0,1)
        u = self.rng.random()
        return max(1, int(profile.output_len_pareto_xmin / u ** (1.0 / profile.output_len_pareto_alpha)))

    def sample_session_duration(self, profile: WorkloadProfile) -> float:
        """Session lifetime in seconds."""
        if profile.session_duration_dist == "lognormal":
            return float(self.rng.lognormal(
                np.log(profile.session_duration_mean_s) - 0.125,
                0.5,
            ))
        # Weibull: x = scale * (-ln(U))^(1/k), k=0.8
        u = self.rng.random()
        return float(profile.session_duration_mean_s * (-math.log(max(u, 1e-15))) ** (1.0 / 0.8))

    def prefix_stability(self, profile: WorkloadProfile, turn: int, total_turns: int) -> float:
        """Fraction of current context that is a stable cached prefix."""
        if total_turns <= 1:
            return profile.prefix_stability_initial
        t = turn / (total_turns - 1)
        return profile.prefix_stability_initial * (1 - t) + profile.prefix_stability_final * t

    def choose_profile(self) -> WorkloadProfile:
        """Select a workload profile according to the mix distribution."""
        names = list(self.config.profile_mix.keys())
        weights = [self.config.profile_mix[n] for n in names]
        chosen = self.rng.choice(len(names), p=weights)
        name = names[chosen]
        for p in self.config.profiles:
            if p.name == name:
                return p
        raise ValueError(f"Profile '{name}' not found in config")

    def sample_next_arrival_time(self, current_time_s: float, profile: WorkloadProfile) -> float:
        """
        Thinning algorithm for NHPP.
        Returns next arrival time in seconds.

        Batched: generate multiple exponential samples at once to reduce
        Python loop overhead.
        """
        rate_max = profile.arrival_rate_peak
        inv_rate_max = 1.0 / max(rate_max, 1e-9)
        t = current_time_s
        rng = self.rng

        # Generate candidates in batches of 32
        while True:
            dts = rng.exponential(inv_rate_max, size=32)
            uniforms = rng.random(size=32)
            for i in range(32):
                t += dts[i]
                rate = self.diurnal_rate(t, profile)
                if uniforms[i] < rate * inv_rate_max:
                    return t
