from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from operators import G_LS, cumtrapz_zero, positive_clip


A = 1.0


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Local trapz utility for core module."""
    return float(np.trapz(y, x))


def _survivorship(mu: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Local survivorship utility for core module."""
    M = cumtrapz_zero(mu, a)
    return np.exp(-M)


def reproductive_number(mu: np.ndarray, k: np.ndarray, a: np.ndarray) -> float:
    S = _survivorship(mu, a)
    return _trapz(k * S, a)



def fertility_shape(
    a: np.ndarray,
    base: float,
    amp: float,
    center: float,
    sigma: float,
) -> np.ndarray:
    k = base + amp * np.exp(-0.5 * ((a - center) / sigma) ** 2)
    return positive_clip(k)


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
    g = offset + amp * np.exp(-0.5 * ((a - center) / sigma) ** 2)
    return positive_clip(g)


@dataclass
class UniformRange:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass
class FamilyRanges:
    K_base: UniformRange
    K_amp: UniformRange
    K_center: UniformRange
    K_sigma: UniformRange

    MU_min: UniformRange
    MU_juv: UniformRange
    MU_r_juv: UniformRange
    MU_sen: UniformRange
    MU_p_sen: UniformRange

    G_offset: UniformRange
    G_amp: UniformRange
    G_center: UniformRange
    G_sigma: UniformRange


@dataclass
class SampledParams:
    K_base: float
    K_amp: float
    K_center: float
    K_sigma: float

    MU_min: float
    MU_juv: float
    MU_r_juv: float
    MU_sen: float
    MU_p_sen: float

    G_offset: float
    G_amp: float
    G_center: float
    G_sigma: float


def build_default_ranges(config_module) -> FamilyRanges:
    c = config_module
    return FamilyRanges(
        K_base=UniformRange(c.K_BASE_MIN, c.K_BASE_MAX),
        K_amp=UniformRange(c.K_AMP_MIN, c.K_AMP_MAX),
        K_center=UniformRange(c.K_CENTER_MIN, c.K_CENTER_MAX),
        K_sigma=UniformRange(c.K_SIGMA_MIN, c.K_SIGMA_MAX),

        MU_min=UniformRange(c.MU_MIN_MIN, c.MU_MIN_MAX),
        MU_juv=UniformRange(c.MU_JUV_MIN, c.MU_JUV_MAX),
        MU_r_juv=UniformRange(c.MU_R_JUV_MIN, c.MU_R_JUV_MAX),
        MU_sen=UniformRange(c.MU_SEN_MIN, c.MU_SEN_MAX),
        MU_p_sen=UniformRange(c.MU_P_SEN_MIN, c.MU_P_SEN_MAX),

        G_offset=UniformRange(c.G_OFFSET_MIN, c.G_OFFSET_MAX),
        G_amp=UniformRange(c.G_AMP_MIN, c.G_AMP_MAX),
        G_center=UniformRange(c.G_CENTER_MIN, c.G_CENTER_MAX),
        G_sigma=UniformRange(c.G_SIGMA_MIN, c.G_SIGMA_MAX),
    )


class ParametricFamilySampler:
    def __init__(
        self,
        ranges: FamilyRanges,
        seed: int,
        n_grid: int,
        target_R0_min: float,
        max_tries: int = 10000,
    ):
        self.rng = np.random.default_rng(seed)
        self.ranges = ranges
        self.a = np.linspace(0.0, A, n_grid)
        self.target_R0_min = target_R0_min
        self.max_tries = max_tries

    def sample_parameters(self) -> SampledParams:
        r = self.ranges
        return SampledParams(
            K_base=r.K_base.sample(self.rng),
            K_amp=r.K_amp.sample(self.rng),
            K_center=r.K_center.sample(self.rng),
            K_sigma=r.K_sigma.sample(self.rng),
            MU_min=r.MU_min.sample(self.rng),
            MU_juv=r.MU_juv.sample(self.rng),
            MU_r_juv=r.MU_r_juv.sample(self.rng),
            MU_sen=r.MU_sen.sample(self.rng),
            MU_p_sen=r.MU_p_sen.sample(self.rng),
            G_offset=r.G_offset.sample(self.rng),
            G_amp=r.G_amp.sample(self.rng),
            G_center=r.G_center.sample(self.rng),
            G_sigma=r.G_sigma.sample(self.rng),
        )

    def build_family(self, p: SampledParams) -> Dict[str, np.ndarray | float | SampledParams]:
        k = fertility_shape(self.a, p.K_base, p.K_amp, p.K_center, p.K_sigma)
        mu = mortality_shape(self.a, p.MU_min, p.MU_juv, p.MU_r_juv, p.MU_sen, p.MU_p_sen)
        g = interaction_shape(self.a, p.G_offset, p.G_amp, p.G_center, p.G_sigma)

        R0 = reproductive_number(mu, k, self.a)
        zeta = G_LS(k, mu, self.a)

        return {
            "k": k,
            "mu": mu,
            "g": g,
            "R0": R0,
            "zeta": zeta,
            "params": p,
        }

    def sample_one(self) -> Dict[str, np.ndarray | float | SampledParams]:
        for _ in range(self.max_tries):
            p = self.sample_parameters()
            fam = self.build_family(p)

            R0 = float(fam["R0"])
            zeta = float(fam["zeta"])

            if R0 > self.target_R0_min and np.isfinite(zeta):
                return fam

        raise RuntimeError(
            f"Could not sample a valid family with R0 > {self.target_R0_min} "
            f"and finite LS zeta after {self.max_tries} attempts."
        )


def generate_dataset_from_parametric_families(
    n_families: int,
    seed: int,
    n_grid: int,
    target_R0_min: float,
    config_module,
) -> Tuple[dict, pd.DataFrame]:
    ranges = build_default_ranges(config_module)
    sampler = ParametricFamilySampler(
        ranges=ranges,
        seed=seed,
        n_grid=n_grid,
        target_R0_min=target_R0_min,
    )

    samples: List[Dict[str, np.ndarray | float | SampledParams]] = []
    rows: List[Dict[str, float]] = []

    for j in range(n_families):
        fam = sampler.sample_one()
        p = fam["params"]
        samples.append(fam)

        rows.append(
            {
                "family_id": j,
                "R0": float(fam["R0"]),
                "zeta": float(fam["zeta"]),
                "K_base": p.K_base,
                "K_amp": p.K_amp,
                "K_center": p.K_center,
                "K_sigma": p.K_sigma,
                "MU_min": p.MU_min,
                "MU_juv": p.MU_juv,
                "MU_r_juv": p.MU_r_juv,
                "MU_sen": p.MU_sen,
                "MU_p_sen": p.MU_p_sen,
                "G_offset": p.G_offset,
                "G_amp": p.G_amp,
                "G_center": p.G_center,
                "G_sigma": p.G_sigma,
            }
        )

    df = pd.DataFrame(rows)

    arrays = {
        "a": sampler.a,
        "k": np.stack([s["k"] for s in samples], axis=0),
        "mu": np.stack([s["mu"] for s in samples], axis=0),
        "g": np.stack([s["g"] for s in samples], axis=0),
        "R0": np.array([s["R0"] for s in samples], dtype=float),
        "zeta": np.array([s["zeta"] for s in samples], dtype=float),
    }

    return arrays, df