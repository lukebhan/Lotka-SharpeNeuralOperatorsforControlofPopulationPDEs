from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def positive_clip(y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.maximum(y, eps)


def build_age_gain_profile(
    a: np.ndarray,
    mode: str,
    gamma_base: float,
    gamma_max: Optional[float] = None,
    rho: float = 2.0,
    custom_profile: Optional[np.ndarray] = None,
) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    A = max(a[-1] - a[0], 1e-12)
    an = (a - a[0]) / A

    if mode == "constant":
        return gamma_base * np.ones_like(a)

    if mode == "exp_increasing":
        raw = np.exp(rho * an)
        mean_raw = float(np.mean(raw))
        return gamma_base * raw / mean_raw

    if mode == "bounded_exp_increasing":
        if gamma_max is None:
            raise ValueError("gamma_max must be provided for bounded_exp_increasing.")
        shape = (np.exp(rho * an) - 1.0) / (np.exp(rho) - 1.0)
        return gamma_base + (gamma_max - gamma_base) * shape

    if mode == "custom":
        if custom_profile is None:
            raise ValueError("custom_profile must be provided when mode='custom'.")
        custom_profile = np.asarray(custom_profile, dtype=float)
        if custom_profile.shape != a.shape:
            raise ValueError("custom_profile must have same shape as a.")
        return custom_profile.copy()

    raise ValueError(f"Unknown gain profile mode: {mode}")


@dataclass
class MuEstimatorConfig:
    adapt: bool = True

    Gamma_1: float = 0.01
    Gamma_2: float = 0.01

    # These are the alpha_i filter rates in the swapping filters.
    sigma_1: float = 0.1
    sigma_2: float = 0.1

    gamma_profile_mode_1: str = "constant"
    gamma_profile_mode_2: str = "constant"

    Gamma_1_max: float | None = None
    Gamma_2_max: float | None = None

    gamma_rho_1: float = 2.0
    gamma_rho_2: float = 2.0

    Gamma_profile_1: np.ndarray | None = None
    Gamma_profile_2: np.ndarray | None = None

    eps_resource: float = 1e-4
    project_mu_to_nonnegative: bool = True

    mu_lower_1: np.ndarray | None = None
    mu_upper_1: np.ndarray | None = None
    mu_lower_2: np.ndarray | None = None
    mu_upper_2: np.ndarray | None = None


class MuEstimator:
    r"""
    Pointwise swapping-filter estimator for the age-dependent mortalities mu_i(a).

    For each species i and each age a:

        sigma_t = -alpha sigma + x
        rho_t   = -alpha rho + r

    where

        r_1(a,t) = -d_a x_1(a,t) - Q_1(t) x_1(a,t)
        r_2(a,t) = -d_a x_2(a,t) - Q_2(t) x_2(a,t)

        Q_1(t) = u(t) + ∫ g_1 x_2
        Q_2(t) = u(t) + 1 / (eps + ∫ g_2 x_1)

    The exact pointwise regression is

        Y(a,t) = mu(a) sigma(a,t)

    with

        Y(a,t) = rho(a,t) - x(a,t) + alpha sigma(a,t) + exp(-alpha t) x_0(a).

    The adaptive law is

        mu_hat_t(a,t) =
            Gamma(a) * sigma(a,t)/(1 + sigma(a,t)^2) * (Y(a,t) - mu_hat(a,t) sigma(a,t)).
    """

    def __init__(
        self,
        a: np.ndarray,
        mu1_init: np.ndarray,
        mu2_init: np.ndarray,
        x1_init: np.ndarray,
        x2_init: np.ndarray,
        config: MuEstimatorConfig,
    ):
        self.a = np.asarray(a, dtype=float)
        self.config = config

        self.Gamma1_profile = build_age_gain_profile(
            a=self.a,
            mode=config.gamma_profile_mode_1,
            gamma_base=config.Gamma_1,
            gamma_max=config.Gamma_1_max,
            rho=config.gamma_rho_1,
            custom_profile=config.Gamma_profile_1,
        )
        self.Gamma2_profile = build_age_gain_profile(
            a=self.a,
            mode=config.gamma_profile_mode_2,
            gamma_base=config.Gamma_2,
            gamma_max=config.Gamma_2_max,
            rho=config.gamma_rho_2,
            custom_profile=config.Gamma_profile_2,
        )

        self.mu1_hat = np.asarray(mu1_init, dtype=float).copy()
        self.mu2_hat = np.asarray(mu2_init, dtype=float).copy()

        self.x1_init = np.asarray(x1_init, dtype=float).copy()
        self.x2_init = np.asarray(x2_init, dtype=float).copy()

        self.sigma1 = np.zeros_like(self.a, dtype=float)
        self.sigma2 = np.zeros_like(self.a, dtype=float)
        self.rho1 = np.zeros_like(self.a, dtype=float)
        self.rho2 = np.zeros_like(self.a, dtype=float)

        self.Y1 = np.zeros_like(self.a, dtype=float)
        self.Y2 = np.zeros_like(self.a, dtype=float)
        self.regression_error1 = np.zeros_like(self.a, dtype=float)
        self.regression_error2 = np.zeros_like(self.a, dtype=float)

        self.time = 0.0

    def _known_loss_species_1(
        self,
        g1: np.ndarray,
        x2: np.ndarray,
        u: float,
    ) -> float:
        return float(u + np.trapezoid(g1 * x2, self.a))

    def _known_loss_species_2(
        self,
        g2: np.ndarray,
        x1: np.ndarray,
        u: float,
    ) -> float:
        resource = float(np.trapezoid(g2 * x1, self.a))
        return float(u + 1.0 / (self.config.eps_resource + resource))

    def _project_mu(self, mu: np.ndarray, species: int) -> np.ndarray:
        out = mu.copy()
        if self.config.project_mu_to_nonnegative:
            out = positive_clip(out)

        if species == 1:
            if self.config.mu_lower_1 is not None:
                out = np.maximum(out, self.config.mu_lower_1)
            if self.config.mu_upper_1 is not None:
                out = np.minimum(out, self.config.mu_upper_1)
        else:
            if self.config.mu_lower_2 is not None:
                out = np.maximum(out, self.config.mu_lower_2)
            if self.config.mu_upper_2 is not None:
                out = np.minimum(out, self.config.mu_upper_2)
        return out

    def _age_derivative(self, x: np.ndarray) -> np.ndarray:
        edge_order = 2 if len(self.a) >= 3 else 1
        return np.gradient(x, self.a, edge_order=edge_order)

    def _step_species(
        self,
        x: np.ndarray,
        x_init: np.ndarray,
        mu_hat: np.ndarray,
        sigma_state: np.ndarray,
        rho_state: np.ndarray,
        Gamma_profile: np.ndarray,
        alpha: float,
        Q: float,
        dt: float,
        species: int,
    ) -> dict:
        x = np.asarray(x, dtype=float)
        x_a = self._age_derivative(x)
        r = -x_a - Q * x

        sigma_dot = -alpha * sigma_state + x
        rho_dot = -alpha * rho_state + r
        sigma_new = sigma_state + dt * sigma_dot
        rho_new = rho_state + dt * rho_dot

        exp_term = np.exp(-alpha * self.time) * x_init
        Y = rho_new - x + alpha * sigma_new + exp_term
        regression_error = Y - mu_hat * sigma_new

        if self.config.adapt:
            mu_dot = (
                Gamma_profile
                * (sigma_new / (1.0 + sigma_new * sigma_new))
                * regression_error
            )
            mu_hat_new = mu_hat + dt * mu_dot
        else:
            mu_hat_new = mu_hat.copy()

        mu_hat_new = self._project_mu(mu_hat_new, species=species)
        regression_error_new = Y - mu_hat_new * sigma_new
        regression_error_norm = float(
            np.sqrt(np.trapezoid(regression_error_new * regression_error_new, self.a))
        )

        return {
            "mu_hat_new": mu_hat_new,
            "sigma_new": sigma_new,
            "rho_new": rho_new,
            "Y": Y,
            "r": r,
            "Q": Q,
            "regression_error": regression_error_new,
            "regression_error_norm": regression_error_norm,
        }

    def step(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        g1: np.ndarray,
        g2: np.ndarray,
        u: float,
        dt: float,
    ) -> dict:
        Q1 = self._known_loss_species_1(g1=g1, x2=x2, u=u)
        Q2 = self._known_loss_species_2(g2=g2, x1=x1, u=u)

        out1 = self._step_species(
            x=np.asarray(x1, dtype=float),
            x_init=self.x1_init,
            mu_hat=self.mu1_hat,
            sigma_state=self.sigma1,
            rho_state=self.rho1,
            Gamma_profile=self.Gamma1_profile,
            alpha=self.config.sigma_1,
            Q=Q1,
            dt=dt,
            species=1,
        )
        out2 = self._step_species(
            x=np.asarray(x2, dtype=float),
            x_init=self.x2_init,
            mu_hat=self.mu2_hat,
            sigma_state=self.sigma2,
            rho_state=self.rho2,
            Gamma_profile=self.Gamma2_profile,
            alpha=self.config.sigma_2,
            Q=Q2,
            dt=dt,
            species=2,
        )

        self.mu1_hat = out1["mu_hat_new"]
        self.mu2_hat = out2["mu_hat_new"]
        self.sigma1 = out1["sigma_new"]
        self.sigma2 = out2["sigma_new"]
        self.rho1 = out1["rho_new"]
        self.rho2 = out2["rho_new"]
        self.Y1 = out1["Y"]
        self.Y2 = out2["Y"]
        self.regression_error1 = out1["regression_error"]
        self.regression_error2 = out2["regression_error"]
        self.time += dt

        return {
            "mu1_hat": self.mu1_hat.copy(),
            "mu2_hat": self.mu2_hat.copy(),
            "sigma1": self.sigma1.copy(),
            "sigma2": self.sigma2.copy(),
            "rho1": self.rho1.copy(),
            "rho2": self.rho2.copy(),
            "Y1": self.Y1.copy(),
            "Y2": self.Y2.copy(),
            "r1": out1["r"].copy(),
            "r2": out2["r"].copy(),
            "Q1": Q1,
            "Q2": Q2,
            "regression_error1": self.regression_error1.copy(),
            "regression_error2": self.regression_error2.copy(),
            "regression_error1_norm": out1["regression_error_norm"],
            "regression_error2_norm": out2["regression_error_norm"],
            "Gamma1_profile": self.Gamma1_profile.copy(),
            "Gamma2_profile": self.Gamma2_profile.copy(),
        }
