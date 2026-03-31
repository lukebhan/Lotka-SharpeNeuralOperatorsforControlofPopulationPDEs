from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def positive_clip(y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.maximum(y, eps)


def boundary_prediction_error(
    k_hat: np.ndarray,
    x: np.ndarray,
    a: np.ndarray,
) -> float:
    """
    e(t) = x(0,t) - integral_0^A k_hat(a,t) x(a,t) da
    """
    return float(x[0] - np.trapezoid(k_hat * x, a))


def normalization_denominator(
    x: np.ndarray,
    a: np.ndarray,
) -> float:
    """
    Global normalization term for the k-estimator update:
        1 + integral_0^A x(a,t)^2 da
    """
    return float(1.0 + np.trapezoid(x * x, a))


def estimator_rhs(
    k_hat: np.ndarray,
    x: np.ndarray,
    a: np.ndarray,
    Gamma: float,
) -> tuple[np.ndarray, float, float]:
    """
    Normalized update law:
        d/dt k_hat(a,t)
            = Gamma * x(a,t) * e(t) / (1 + integral_0^A x(alpha,t)^2 d alpha)
    """
    err = boundary_prediction_error(k_hat, x, a)
    denom = normalization_denominator(x, a)
    rhs = Gamma * x * err / denom
    return rhs, err, denom


@dataclass
class KEstimatorConfig:
    adapt: bool = True
    Gamma_1: float = 0.5
    Gamma_2: float = 0.5
    project_to_nonnegative: bool = True


class KEstimator:
    """
    Evolves k-hat for both species.
    """

    def __init__(
        self,
        a: np.ndarray,
        k1_init: np.ndarray,
        k2_init: np.ndarray,
        config: KEstimatorConfig,
    ):
        self.a = a
        self.k1_hat = np.asarray(k1_init, dtype=float).copy()
        self.k2_hat = np.asarray(k2_init, dtype=float).copy()
        self.config = config

    def step(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        dt: float,
    ) -> dict:
        if not self.config.adapt:
            err1 = boundary_prediction_error(self.k1_hat, x1, self.a)
            err2 = boundary_prediction_error(self.k2_hat, x2, self.a)
            denom1 = normalization_denominator(x1, self.a)
            denom2 = normalization_denominator(x2, self.a)
            return {
                "k1_hat": self.k1_hat.copy(),
                "k2_hat": self.k2_hat.copy(),
                "err1": err1,
                "err2": err2,
                "denom1": denom1,
                "denom2": denom2,
            }

        rhs1, err1, denom1 = estimator_rhs(self.k1_hat, x1, self.a, self.config.Gamma_1)
        rhs2, err2, denom2 = estimator_rhs(self.k2_hat, x2, self.a, self.config.Gamma_2)

        self.k1_hat = self.k1_hat + dt * rhs1
        self.k2_hat = self.k2_hat + dt * rhs2

        if self.config.project_to_nonnegative:
            self.k1_hat = positive_clip(self.k1_hat)
            self.k2_hat = positive_clip(self.k2_hat)

        return {
            "k1_hat": self.k1_hat.copy(),
            "k2_hat": self.k2_hat.copy(),
            "err1": err1,
            "err2": err2,
            "denom1": denom1,
            "denom2": denom2,
        }
