from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ============================================================
# Basic numerical utilities
# ============================================================

def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def cumtrapz_zero(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Cumulative trapezoidal integral with initial value zero.
    Returns I such that
        I[j] = integral_0^{x[j]} y(s) ds.
    """
    out = np.zeros_like(y, dtype=float)
    dx = np.diff(x)
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)
    return out


def positive_clip(y: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    return np.maximum(y, eps)


# ============================================================
# Shared exponential weights
# ============================================================

def exp_weight(mu: np.ndarray, zeta: float, a: np.ndarray) -> np.ndarray:
    """
    Returns
        w(a) = exp(- integral_0^a (zeta + mu(s)) ds)
             = exp(-zeta a - integral_0^a mu(s) ds).
    """
    return np.exp(-zeta * a - cumtrapz_zero(mu, a))


# ============================================================
# G_LS : (k, mu) -> zeta
# ============================================================

def LS_residual(k: np.ndarray, mu: np.ndarray, a: np.ndarray, zeta: float) -> float:
    """
    Residual of the Lotka-Sharpe equation:
        F(zeta) - 1
    where
        F(zeta) = integral_0^A k(a) exp(- integral_0^a (zeta + mu(s)) ds ) da.
    """
    w = exp_weight(mu, zeta, a)
    return trapezoid(k * w, a) - 1.0


def G_LS(
    k: np.ndarray,
    mu: np.ndarray,
    a: np.ndarray,
    zeta_upper_start: float = 5.0,
    tol: float = 1e-10,
    max_iter: int = 150,
) -> float:
    """
    Lotka-Sharpe operator:
        (k, mu) -> zeta

    Solves for zeta in
        integral_0^A k(a) exp(- integral_0^a (zeta + mu(s)) ds ) da = 1.

    Uses monotonicity + bisection.

    Returns:
        zeta > 0 if a positive root exists,
        np.nan otherwise.
    """
    f0 = LS_residual(k, mu, a, 0.0)
    if f0 <= 0.0:
        return np.nan

    lo = 0.0
    hi = zeta_upper_start
    f_hi = LS_residual(k, mu, a, hi)

    while f_hi > 0.0 and hi < 1e6:
        hi *= 2.0
        f_hi = LS_residual(k, mu, a, hi)

    if f_hi > 0.0:
        return np.nan

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = LS_residual(k, mu, a, mid)

        if abs(f_mid) < tol:
            return mid

        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


# ============================================================
# G_kappa : (k, mu, zeta) -> kappa
# ============================================================

def G_kappa(
    k: np.ndarray,
    mu: np.ndarray,
    zeta: float,
    a: np.ndarray,
) -> float:
    """
    Computes
        kappa = integral_0^A a k(a) exp(- integral_0^a (zeta + mu(s)) ds ) da.
    """
    w = exp_weight(mu, zeta, a)
    return trapezoid(a * k * w, a)


# ============================================================
# G_gamma : (g, zeta, mu) -> gamma
# ============================================================

def G_gamma(
    g: np.ndarray,
    zeta: float,
    mu: np.ndarray,
    a: np.ndarray,
) -> float:
    """
    Computes
        gamma = integral_0^A g(a) exp(- integral_0^a (zeta + mu(s)) ds ) da.
    """
    w = exp_weight(mu, zeta, a)
    return trapezoid(g * w, a)


def G_pi(
    k: np.ndarray,
    mu: np.ndarray,
    zeta: float,
    a: np.ndarray,
) -> np.ndarray:
    """
    Computes
        pi_0(a) = integral_a^A k(s) exp( - integral_a^s (zeta + mu(l)) dl ) ds
    """
    I = zeta * a + cumtrapz_zero(mu, a)
    pi0 = np.zeros_like(a, dtype=float)

    for j in range(len(a)):
        exponent = -(I[j:] - I[j]) # we only use a≤s case and so exponent will be negative. 
        integrand = k[j:] * np.exp(exponent)
        pi0[j] = float(np.trapezoid(integrand, a[j:]))

    return pi0


# ============================================================
# Batch operator computation
# ============================================================


def compute_all_species_operators(
    k1: np.ndarray,
    mu1: np.ndarray,
    g1: np.ndarray,
    k2: np.ndarray,
    mu2: np.ndarray,
    g2: np.ndarray,
    a: np.ndarray,
):
    """
    Convenience function matching your notation.

    Computes:
      zeta1 = G_LS(k1, mu1)
      zeta2 = G_LS(k2, mu2)

      kappa1 = G_kappa(k1, mu1, zeta1)
      kappa2 = G_kappa(k2, mu2, zeta2)

      gamma1 = G_gamma(g1, zeta2, mu2)
      gamma2 = G_gamma(g2, zeta1, mu1)

      pi0_1 = G_pi(k1, mu1, zeta1)
      pi0_2 = G_pi(k2, mu2, zeta2)
    """
    zeta1 = G_LS(k1, mu1, a)
    zeta2 = G_LS(k2, mu2, a)

    if not np.isfinite(zeta1) or not np.isfinite(zeta2):
        raise ValueError("Failed to compute zeta1 or zeta2.")

    kappa1 = G_kappa(k1, mu1, zeta1, a)
    kappa2 = G_kappa(k2, mu2, zeta2, a)

    gamma1 = G_gamma(g1, zeta2, mu2, a)
    gamma2 = G_gamma(g2, zeta1, mu1, a)

    pi0_1 = G_pi(k1, mu1, zeta1, a)
    pi0_2 = G_pi(k2, mu2, zeta2, a)

    return {
        "zeta1": zeta1,
        "zeta2": zeta2,
        "kappa1": kappa1,
        "kappa2": kappa2,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "pi0_1": pi0_1,
        "pi0_2": pi0_2,
    }