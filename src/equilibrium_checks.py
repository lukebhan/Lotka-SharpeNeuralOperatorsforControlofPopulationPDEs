from __future__ import annotations

from dataclasses import dataclass

from plant import Plant


@dataclass
class EquilibriumCheckResult:
    u_star: float
    x1_star_age0: float
    x2_star_age0: float
    lower_bound_x1: float
    valid_u_positive: bool
    valid_u_less_than_zetas: bool
    valid_gamma_positive: bool
    valid_x1_star_positive: bool
    valid_x2_star_positive: bool
    valid_x1_lower_bound: bool


def compute_equilibrium_values(plant: Plant, u_star: float) -> EquilibriumCheckResult:
    gamma_ok = (plant.gamma1 > 0.0) and (plant.gamma2 > 0.0)
    zeta_ok = u_star > 0.0 and u_star < min(plant.zeta1, plant.zeta2)

    if gamma_ok and zeta_ok:
        x1_star_age0 = 1.0 / ((plant.zeta2 - u_star) * plant.gamma2)
        x2_star_age0 = (plant.zeta1 - u_star) / plant.gamma1
        lower_bound_x1 = 1.0 / (plant.zeta2 * plant.gamma2)
    else:
        x1_star_age0 = float("nan")
        x2_star_age0 = float("nan")
        lower_bound_x1 = float("nan")

    valid_x1_star_positive = gamma_ok and zeta_ok and (x1_star_age0 > 0.0)
    valid_x2_star_positive = gamma_ok and zeta_ok and (x2_star_age0 > 0.0)
    valid_x1_lower_bound = gamma_ok and zeta_ok and (x1_star_age0 > lower_bound_x1)

    return EquilibriumCheckResult(
        u_star=u_star,
        x1_star_age0=x1_star_age0,
        x2_star_age0=x2_star_age0,
        lower_bound_x1=lower_bound_x1,
        valid_u_positive=(u_star > 0.0),
        valid_u_less_than_zetas=(u_star < min(plant.zeta1, plant.zeta2)),
        valid_gamma_positive=gamma_ok,
        valid_x1_star_positive=valid_x1_star_positive,
        valid_x2_star_positive=valid_x2_star_positive,
        valid_x1_lower_bound=valid_x1_lower_bound,
    )


def validate_equilibrium_values(plant: Plant, u_star: float) -> EquilibriumCheckResult:
    result = compute_equilibrium_values(plant, u_star)

    if not result.valid_u_positive:
        raise ValueError(f"Need u_star > 0. Got u_star={u_star:.6f}.")

    if not result.valid_u_less_than_zetas:
        raise ValueError(
            f"Need u_star < min(zeta1, zeta2). Got u_star={u_star:.6f}, "
            f"zeta1={plant.zeta1:.6f}, zeta2={plant.zeta2:.6f}."
        )

    if not result.valid_gamma_positive:
        raise ValueError(
            f"Need gamma1, gamma2 > 0. Got gamma1={plant.gamma1:.6f}, gamma2={plant.gamma2:.6f}."
        )

    if not result.valid_x1_star_positive:
        raise ValueError(f"x1*(0) must be positive. Got {result.x1_star_age0:.12f}.")

    if not result.valid_x2_star_positive:
        raise ValueError(f"x2*(0) must be positive. Got {result.x2_star_age0:.12f}.")

    if not result.valid_x1_lower_bound:
        raise ValueError(
            "Need x1*(0) > 1/(zeta2*gamma2). "
            f"Got x1*(0)={result.x1_star_age0:.12f}, "
            f"1/(zeta2*gamma2)={result.lower_bound_x1:.12f}."
        )

    return result