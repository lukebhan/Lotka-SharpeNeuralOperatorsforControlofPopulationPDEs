from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def inner_product(f: np.ndarray, g: np.ndarray, a: np.ndarray) -> float:
    """
    Pairing over age:
        <f, g> = integral_0^A f(a) g(a) da
    """
    return float(np.trapezoid(f * g, a))


class NominalController:
    """
    Implements the nominal control law

        u_nom
        = zeta2 - 1/(x1_star_0 * gamma2)
          + beta * [
              (1+epsilon)(zeta2-zeta1)
              - epsilon/(x1_star_0 * gamma2)
              - kappa1/(gamma2 * <pi0_1, x1>)
              + (1+epsilon)(gamma1/kappa2) <pi0_2, x2>
            ]
    """

    def __init__(
        self,
        a: np.ndarray,
        zeta1: float,
        zeta2: float,
        gamma1: float,
        gamma2: float,
        kappa1: float,
        kappa2: float,
        pi0_1: np.ndarray,
        pi0_2: np.ndarray,
        x1_star_0: float,
        x2_star_0: float | None = None,
        beta: float = 1.0,
        epsilon: float = 0.1,
        denom_eps: float = 1e-8,
    ):
        self.a = a

        self.zeta1 = float(zeta1)
        self.zeta2 = float(zeta2)
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)
        self.kappa1 = float(kappa1)
        self.kappa2 = float(kappa2)

        self.pi0_1 = np.asarray(pi0_1, dtype=float)
        self.pi0_2 = np.asarray(pi0_2, dtype=float)

        self.x1_star_0 = float(x1_star_0)
        self.x2_star_0 = None if x2_star_0 is None else float(x2_star_0)

        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.denom_eps = float(denom_eps)

        self._validate()

    def _validate(self) -> None:
        if self.gamma1 <= 0.0:
            raise ValueError(f"gamma1 must be positive. Got {self.gamma1}.")
        if self.gamma2 <= 0.0:
            raise ValueError(f"gamma2 must be positive. Got {self.gamma2}.")
        if self.kappa1 <= 0.0:
            raise ValueError(f"kappa1 must be positive. Got {self.kappa1}.")
        if self.kappa2 <= 0.0:
            raise ValueError(f"kappa2 must be positive. Got {self.kappa2}.")
        if self.x1_star_0 <= 0.0:
            raise ValueError(f"x1_star_0 must be positive. Got {self.x1_star_0}.")
        if self.pi0_1.shape != self.a.shape:
            raise ValueError("pi0_1 must have the same shape as a.")
        if self.pi0_2.shape != self.a.shape:
            raise ValueError("pi0_2 must have the same shape as a.")

    def diagnostics(self, x1: np.ndarray, x2: np.ndarray) -> dict:
        """
        Return a detailed decomposition of the control law.

        Includes both raw and safeguarded versions of the inner-product terms.
        """
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        ip1 = inner_product(self.pi0_1, x1, self.a)
        ip2 = inner_product(self.pi0_2, x2, self.a)

        ip1_safe = max(ip1, self.denom_eps)
        ip2_safe = max(ip2, self.denom_eps)

        base_term = self.zeta2 - 1.0 / (self.x1_star_0 * self.gamma2)

        term_A = (1.0 + self.epsilon) * (self.zeta2 - self.zeta1)
        term_B = - self.epsilon / (self.x1_star_0 * self.gamma2)
        term_C_raw = - self.kappa1 / (self.gamma2 * ip1)
        term_C_safe = - self.kappa1 / (self.gamma2 * ip1_safe)
        term_D_raw = (1.0 + self.epsilon) * (self.gamma1 / self.kappa2) * ip2
        term_D_safe = (1.0 + self.epsilon) * (self.gamma1 / self.kappa2) * ip2_safe

        feedback_term_raw = term_A + term_B + term_C_raw + term_D_raw
        feedback_term = term_A + term_B + term_C_safe + term_D_safe

        u_nom_raw = base_term + self.beta * feedback_term_raw
        u_nom = base_term + self.beta * feedback_term

        out = {
            "ip_pi0_1_x1": ip1,
            "ip_pi0_2_x2": ip2,
            "ip_pi0_1_x1_safe": ip1_safe,
            "ip_pi0_2_x2_safe": ip2_safe,
            "base_term": base_term,
            "term_A_(1+eps)(z2-z1)": term_A,
            "term_B_-eps_over_x1star0gamma2": term_B,
            "term_C_raw_-kappa1_over_gamma2_ip1": term_C_raw,
            "term_C_safe_-kappa1_over_gamma2_ip1safe": term_C_safe,
            "term_D_raw_(1+eps)gamma1_over_kappa2_ip2": term_D_raw,
            "term_D_safe_(1+eps)gamma1_over_kappa2_ip2safe": term_D_safe,
            "feedback_term_raw": feedback_term_raw,
            "feedback_term": feedback_term,
            "u_nom_raw": u_nom_raw,
            "u_nom": u_nom,
        }

        # Extra equilibrium-targeted comparisons if x2_star_0 is known
        if self.x2_star_0 is not None:
            out.update({
                "target_ip1_x1star0_kappa1": self.x1_star_0 * self.kappa1,
                "target_ip2_x2star0_kappa2": self.x2_star_0 * self.kappa2,
                "termC_target_-1_over_gamma2_x1star0": -1.0 / (self.gamma2 * self.x1_star_0),
                "termD_target_(1+eps)gamma1_x2star0": (1.0 + self.epsilon) * self.gamma1 * self.x2_star_0,
            })

        return out

    def print_diagnostics(self, x1: np.ndarray, x2: np.ndarray, label: str = "controller diagnostics") -> None:
        """
        Pretty-print the decomposition so you can see which pieces cancel.
        """
        d = self.diagnostics(x1, x2)

        print(label)
        print("-" * len(label))
        print(f"<pi0_1, x1>                         = {d['ip_pi0_1_x1']:.12f}")
        print(f"<pi0_2, x2>                         = {d['ip_pi0_2_x2']:.12f}")
        print(f"<pi0_1, x1> safe                    = {d['ip_pi0_1_x1_safe']:.12f}")
        print(f"<pi0_2, x2> safe                    = {d['ip_pi0_2_x2_safe']:.12f}")
        print()
        print(f"base term                           = {d['base_term']:.12f}")
        print()
        print(f"term A = (1+eps)(zeta2-zeta1)       = {d['term_A_(1+eps)(z2-z1)']:.12f}")
        print(f"term B = -eps/(x1*(0) gamma2)       = {d['term_B_-eps_over_x1star0gamma2']:.12f}")
        print(f"term C raw                          = {d['term_C_raw_-kappa1_over_gamma2_ip1']:.12f}")
        print(f"term C safe                         = {d['term_C_safe_-kappa1_over_gamma2_ip1safe']:.12f}")
        print(f"term D raw                          = {d['term_D_raw_(1+eps)gamma1_over_kappa2_ip2']:.12f}")
        print(f"term D safe                         = {d['term_D_safe_(1+eps)gamma1_over_kappa2_ip2safe']:.12f}")
        print()
        print(f"feedback raw                        = {d['feedback_term_raw']:.12f}")
        print(f"feedback safe                       = {d['feedback_term']:.12f}")
        print(f"u_nom raw                           = {d['u_nom_raw']:.12f}")
        print(f"u_nom safe                          = {d['u_nom']:.12f}")

        if "target_ip1_x1star0_kappa1" in d:
            print()
            print("equilibrium consistency targets")
            print(f"x1*(0) * kappa1                     = {d['target_ip1_x1star0_kappa1']:.12f}")
            print(f"x2*(0) * kappa2                     = {d['target_ip2_x2star0_kappa2']:.12f}")
            print(f"target term C                       = {d['termC_target_-1_over_gamma2_x1star0']:.12f}")
            print(f"target term D                       = {d['termD_target_(1+eps)gamma1_x2star0']:.12f}")

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Evaluate the nominal control law at the current state (x1, x2).
        """
        info = self.diagnostics(x1, x2)
        return float(info["u_nom"])