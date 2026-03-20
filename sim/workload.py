from __future__ import annotations
import numpy as np
from scipy.stats import pareto, lognorm, weibull_min

from sim.config import SimConfig, WorkloadProfile


class WorkloadSynthesizer:
    """
    Generates request arrival events from configured session profiles.
    Uses non-homogeneous Poisson process with sinusoidal diurnal rate.
    """

    def __init__(self, config: SimConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    def diurnal_rate(self, time_s: float, profile: WorkloadProfile) -> float:
        """
        Rate function for NHPP.
        Sinusoidal with period 86400s, peak at 9 AM (offset 32400s).
        """
        mean_rate = profile.arrival_rate_peak / profile.diurnal_peak_trough_ratio
        amplitude = mean_rate * (profile.diurnal_peak_trough_ratio - 1) / 2
        phase = (2 * np.pi * (time_s - 32400)) / 86400
        return max(0.0, mean_rate + amplitude * np.sin(phase))

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
        """Tokens, Pareto-tailed."""
        return max(1, int(pareto.rvs(
            profile.output_len_pareto_alpha,
            scale=profile.output_len_pareto_xmin,
            random_state=self.rng,
        )))

    def sample_session_duration(self, profile: WorkloadProfile) -> float:
        """Session lifetime in seconds."""
        if profile.session_duration_dist == "lognormal":
            return float(self.rng.lognormal(
                np.log(profile.session_duration_mean_s) - 0.125,
                0.5,
            ))
        return float(weibull_min.rvs(
            0.8, scale=profile.session_duration_mean_s, random_state=self.rng,
        ))

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
        """
        rate_max = profile.arrival_rate_peak
        t = current_time_s
        while True:
            # Exponential with max rate
            dt = self.rng.exponential(1.0 / max(rate_max, 1e-9))
            t += dt
            rate = self.diurnal_rate(t, profile)
            if self.rng.random() < rate / max(rate_max, 1e-9):
                return t
