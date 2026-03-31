from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

import config
from operators import G_LS, G_gamma


A = 1.0


# ============================================================
# Basic utilities
# ============================================================

def positive_clip(y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.maximum(y, eps)


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def cumtrapz_zero(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    dx = np.diff(x)
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)
    return out


# ============================================================
# Smooth biological shapes
# ============================================================

def gaussian_kernel(
    a: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    baseline: float = 0.0,
) -> np.ndarray:
    return baseline + amplitude * np.exp(-0.5 * ((a - center) / sigma) ** 2)


def fertility_shape(
    a: np.ndarray,
    base: float,
    amp: float,
    center: float,
    sigma: float,
) -> np.ndarray:
    return positive_clip(gaussian_kernel(a, amp, center, sigma, baseline=base))


def mortality_shape(
    a: np.ndarray,
    mu_min: float,
    mu_juv: float,
    r_juv: float,
    mu_sen: float,
    p_sen: float,
) -> np.ndarray:
    mu = mu_min + mu_juv * np.exp(-r_juv * a) + mu_sen * np.power(a, p_sen)
    return positive_clip(mu)


def interaction_shape(
    a: np.ndarray,
    offset: float,
    amp: float,
    center: float,
    sigma: float,
) -> np.ndarray:
    return positive_clip(gaussian_kernel(a, amp, center, sigma, baseline=offset))


def random_smooth_unit_field(
    a: np.ndarray,
    rng: np.random.Generator,
    n_bumps_range: Tuple[int, int] = (2, 5),
    sigma_range: Tuple[float, float] = (0.10, 0.28),
) -> np.ndarray:
    raw = np.zeros_like(a)

    n_bumps = rng.integers(n_bumps_range[0], n_bumps_range[1] + 1)
    for _ in range(n_bumps):
        amp = rng.uniform(-1.0, 1.0)
        center = rng.uniform(0.0, 1.0)
        sigma = rng.uniform(*sigma_range)
        raw += amp * np.exp(-0.5 * ((a - center) / sigma) ** 2)

    raw += rng.uniform(-0.75, 0.75)
    raw += rng.uniform(-0.5, 0.5) * (a - 0.5)

    return 1.0 / (1.0 + np.exp(-raw))


# ============================================================
# Lotka-Sharpe-related utilities
# ============================================================

def lotka_sharpe_integral(mu: np.ndarray, k: np.ndarray, a: np.ndarray, zeta: float) -> float:
    return float(
        np.trapezoid(
            k * np.exp(-cumtrapz_zero(mu, a) - zeta * a),
            a,
        )
    )


def reproductive_number(mu: np.ndarray, k: np.ndarray, a: np.ndarray) -> float:
    return lotka_sharpe_integral(mu, k, a, 0.0)


def stable_age_profile(mu: np.ndarray, zeta: float, a: np.ndarray) -> np.ndarray:
    return np.exp(-cumtrapz_zero(mu, a) - zeta * a)


# ============================================================
# Data containers
# ============================================================

@dataclass
class UniformRange:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass
class FertilityRanges:
    base: UniformRange
    amp: UniformRange
    center: UniformRange
    sigma: UniformRange


@dataclass
class MortalityRanges:
    mu_min: UniformRange
    mu_juv: UniformRange
    r_juv: UniformRange
    mu_sen: UniformRange
    p_sen: UniformRange


@dataclass
class InteractionRanges:
    offset: UniformRange
    amp: UniformRange
    center: UniformRange
    sigma: UniformRange


@dataclass
class SharedEnvelopes:
    a: np.ndarray
    k_lower: np.ndarray
    k_upper: np.ndarray
    mu_lower: np.ndarray
    mu_upper: np.ndarray
    g1_lower: np.ndarray
    g1_upper: np.ndarray
    g2_lower: np.ndarray
    g2_upper: np.ndarray
    k_ranges: FertilityRanges
    mu_ranges: MortalityRanges
    g1_ranges: InteractionRanges
    g2_ranges: InteractionRanges


@dataclass
class Plant:
    a: np.ndarray
    k1: np.ndarray
    mu1: np.ndarray
    g1: np.ndarray
    k2: np.ndarray
    mu2: np.ndarray
    g2: np.ndarray
    R0_1: float
    R0_2: float
    zeta1: float
    zeta2: float
    n1: np.ndarray
    n2: np.ndarray
    gamma1: float
    gamma2: float


# ============================================================
# Envelope builder
# ============================================================

class EnvelopeGenerator:
    def __init__(
        self,
        seed: int = 1234,
        n_grid: int = 201,
        target_R0_min: float = 1.2,
    ):
        self.rng = np.random.default_rng(seed)
        self.a = np.linspace(0.0, A, n_grid)
        self.target_R0_min = target_R0_min

    @staticmethod
    def _fertility_envelopes(a: np.ndarray, ranges: FertilityRanges) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.full_like(a, ranges.base.low, dtype=float)
        upper = np.full_like(a, ranges.base.high + ranges.amp.high, dtype=float)
        return positive_clip(lower), positive_clip(upper)

    @staticmethod
    def _mortality_envelopes(a: np.ndarray, ranges: MortalityRanges) -> Tuple[np.ndarray, np.ndarray]:
        lower = (
            ranges.mu_min.low
            + ranges.mu_juv.low * np.exp(-ranges.r_juv.high * a)
            + ranges.mu_sen.low * np.power(a, ranges.p_sen.high)
        )
        upper = (
            ranges.mu_min.high
            + ranges.mu_juv.high * np.exp(-ranges.r_juv.low * a)
            + ranges.mu_sen.high * np.power(a, ranges.p_sen.low)
        )
        return positive_clip(lower), positive_clip(upper)

    @staticmethod
    def _interaction_envelopes(a: np.ndarray, ranges: InteractionRanges) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.full_like(a, ranges.offset.low, dtype=float)
        upper = np.full_like(a, ranges.offset.high + ranges.amp.high, dtype=float)
        return positive_clip(lower), positive_clip(upper)

    def build(self) -> SharedEnvelopes:
        a = self.a

        k_ranges = FertilityRanges(
            base=UniformRange(config.K_BASE_MIN, config.K_BASE_MAX),
            amp=UniformRange(config.K_AMP_MIN, config.K_AMP_MAX),
            center=UniformRange(config.K_CENTER_MIN, config.K_CENTER_MAX),
            sigma=UniformRange(config.K_SIGMA_MIN, config.K_SIGMA_MAX),
        )
        mu_ranges = MortalityRanges(
            mu_min=UniformRange(config.MU_MIN_MIN, config.MU_MIN_MAX),
            mu_juv=UniformRange(config.MU_JUV_MIN, config.MU_JUV_MAX),
            r_juv=UniformRange(config.MU_R_JUV_MIN, config.MU_R_JUV_MAX),
            mu_sen=UniformRange(config.MU_SEN_MIN, config.MU_SEN_MAX),
            p_sen=UniformRange(config.MU_P_SEN_MIN, config.MU_P_SEN_MAX),
        )
        g1_ranges = InteractionRanges(
            offset=UniformRange(config.G_OFFSET_MIN, config.G_OFFSET_MAX),
            amp=UniformRange(config.G_AMP_MIN, config.G_AMP_MAX),
            center=UniformRange(config.G_CENTER_MIN, config.G_CENTER_MAX),
            sigma=UniformRange(config.G_SIGMA_MIN, config.G_SIGMA_MAX),
        )
        g2_ranges = InteractionRanges(
            offset=UniformRange(config.G_OFFSET_MIN, config.G_OFFSET_MAX),
            amp=UniformRange(config.G_AMP_MIN, config.G_AMP_MAX),
            center=UniformRange(config.G_CENTER_MIN, config.G_CENTER_MAX),
            sigma=UniformRange(config.G_SIGMA_MIN, config.G_SIGMA_MAX),
        )

        k_lower, k_upper = self._fertility_envelopes(a, k_ranges)
        mu_lower, mu_upper = self._mortality_envelopes(a, mu_ranges)
        g1_lower, g1_upper = self._interaction_envelopes(a, g1_ranges)
        g2_lower, g2_upper = self._interaction_envelopes(a, g2_ranges)

        env = SharedEnvelopes(
            a=a,
            k_lower=positive_clip(k_lower),
            k_upper=positive_clip(k_upper),
            mu_lower=positive_clip(mu_lower),
            mu_upper=positive_clip(mu_upper),
            g1_lower=positive_clip(g1_lower),
            g1_upper=positive_clip(g1_upper),
            g2_lower=positive_clip(g2_lower),
            g2_upper=positive_clip(g2_upper),
            k_ranges=k_ranges,
            mu_ranges=mu_ranges,
            g1_ranges=g1_ranges,
            g2_ranges=g2_ranges,
        )

        self._check(env)
        return env

    def _check(self, env: SharedEnvelopes) -> None:
        if not np.all(env.k_lower <= env.k_upper):
            raise ValueError("k envelope ordering failed.")
        if not np.all(env.mu_lower <= env.mu_upper):
            raise ValueError("mu envelope ordering failed.")
        if not np.all(env.g1_lower <= env.g1_upper):
            raise ValueError("g1 envelope ordering failed.")
        if not np.all(env.g2_lower <= env.g2_upper):
            raise ValueError("g2 envelope ordering failed.")


# ============================================================
# Plant sampler
# ============================================================

class PlantSampler:
    def __init__(
        self,
        envelopes: SharedEnvelopes,
        seed: int = 4321,
        target_R0_min: float = 1.2,
    ):
        self.env = envelopes
        self.a = envelopes.a
        self.rng = np.random.default_rng(seed)
        self.target_R0_min = target_R0_min

    def sample_one_species(
        self,
        g_ranges: InteractionRanges,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
        for _ in range(500):
            k = fertility_shape(
                self.a,
                self.env.k_ranges.base.sample(self.rng),
                self.env.k_ranges.amp.sample(self.rng),
                self.env.k_ranges.center.sample(self.rng),
                self.env.k_ranges.sigma.sample(self.rng),
            )
            mu = mortality_shape(
                self.a,
                self.env.mu_ranges.mu_min.sample(self.rng),
                self.env.mu_ranges.mu_juv.sample(self.rng),
                self.env.mu_ranges.r_juv.sample(self.rng),
                self.env.mu_ranges.mu_sen.sample(self.rng),
                self.env.mu_ranges.p_sen.sample(self.rng),
            )
            g = interaction_shape(
                self.a,
                g_ranges.offset.sample(self.rng),
                g_ranges.amp.sample(self.rng),
                g_ranges.center.sample(self.rng),
                g_ranges.sigma.sample(self.rng),
            )

            r0 = reproductive_number(mu, k, self.a)
            if r0 <= self.target_R0_min:
                continue

            zeta = G_LS(k, mu, self.a)
            if not np.isfinite(zeta) or zeta <= 0.0:
                continue

            n = stable_age_profile(mu, zeta, self.a)
            return k, mu, g, r0, zeta, n

        raise RuntimeError("Could not sample a valid species with R0 > target.")

    def sample_plant(self) -> Plant:
        k1, mu1, g1, r0_1, zeta1, n1 = self.sample_one_species(self.env.g1_ranges)
        k2, mu2, g2, r0_2, zeta2, n2 = self.sample_one_species(self.env.g2_ranges)

        gamma1 = G_gamma(g1, zeta2, mu2, self.a)
        gamma2 = G_gamma(g2, zeta1, mu1, self.a)

        return Plant(
            a=self.a.copy(),
            k1=k1,
            mu1=mu1,
            g1=g1,
            k2=k2,
            mu2=mu2,
            g2=g2,
            R0_1=r0_1,
            R0_2=r0_2,
            zeta1=zeta1,
            zeta2=zeta2,
            n1=n1,
            n2=n2,
            gamma1=gamma1,
            gamma2=gamma2,
        )


# ============================================================
# Reasonable initial conditions
# ============================================================

def sample_initial_conditions(
    plant: Plant,
    x1_star_age0: float,
    x2_star_age0: float,
    seed: int = 999,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Choose biologically reasonable initial population profiles.

    Generation recipe:
    1. Start from the equilibrium profile shapes `plant.n1` and `plant.n2`.
    2. Draw scalar amplitudes `c1`, `c2` by multiplying the equilibrium
       age-zero values by independent Uniform[0.70, 1.35] factors.
    3. Draw smooth multiplicative fields `mod1`, `mod2` of the form
       `0.85 + 0.30 * random_smooth_unit_field(...)`, so each profile gets
       a gentle age-dependent distortion.
    4. Draw one Gaussian bump per species with random center in [0.2, 0.8]
       and random width in [0.10, 0.22], then multiply by
       `1 + 0.20 * bump` to create a localized excess cohort.
    5. Multiply everything together and positive-clip the result.

    So the initial conditions keep the equilibrium decay shape as a baseline,
    while randomizing overall size and adding smooth age-structured
    perturbations.
    """
    rng = np.random.default_rng(seed)
    a = plant.a

    c1 = x1_star_age0 * rng.uniform(0.70, 1.35)
    c2 = x2_star_age0 * rng.uniform(0.70, 1.35)

    mod1 = 0.85 + 0.30 * random_smooth_unit_field(a, rng)
    mod2 = 0.85 + 0.30 * random_smooth_unit_field(a, rng)

    bump1 = 1.0 + 0.20 * np.exp(-0.5 * ((a - rng.uniform(0.2, 0.8)) / rng.uniform(0.10, 0.22)) ** 2)
    bump2 = 1.0 + 0.20 * np.exp(-0.5 * ((a - rng.uniform(0.2, 0.8)) / rng.uniform(0.10, 0.22)) ** 2)

    x1_0 = c1 * plant.n1 * mod1 * bump1
    x2_0 = c2 * plant.n2 * mod2 * bump2

    return positive_clip(x1_0), positive_clip(x2_0)
