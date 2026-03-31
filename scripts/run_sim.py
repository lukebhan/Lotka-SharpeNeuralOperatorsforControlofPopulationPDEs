from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for search_path in (ROOT, SRC):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

from plant import (
    EnvelopeGenerator,
    PlantSampler,
    Plant,
    positive_clip,
    reproductive_number,
    sample_initial_conditions,
    stable_age_profile,
)
from equilibrium_checks import validate_equilibrium_values
from operators import G_LS, G_gamma, compute_all_species_operators
from controller import NominalController
from estimate_k import KEstimator, KEstimatorConfig
from estimate_mu import MuEstimator, MuEstimatorConfig
from plotting import plot_simulation_results


# ============================================================
# User settings
# ============================================================

SEED = 123

N_AGE = 201
T_FINAL = 25.0
DT = 1.0 / (N_AGE - 1)
EPS_RESOURCE = 1e-4

USE_EQUILIBRIUM_INITIAL_CONDITIONS = False

# "constant" or "nominal"
CONTROL_MODE = "nominal"

# Controller parameters
BETA = 1.0
EPSILON = 0.1
CONTROL_DENOM_EPS = 1e-8

# ============================================================
# k-estimation settings
# ============================================================

ADAPT_K = True
USE_ESTIMATED_K_IN_CONTROLLER = True
GAMMA_K_1 = 0.5
GAMMA_K_2 = 0.5

# ============================================================
# mu-estimation settings
# ============================================================

ADAPT_MU = True
USE_ESTIMATED_MU_IN_CONTROLLER = True

MU_ADAPT_GAIN_1 = 0.5
MU_ADAPT_GAIN_2 = 0.5
MU_FILTER_ALPHA_1 = 2.0
MU_FILTER_ALPHA_2 = 2.0

# ============================================================
# CSV sample reuse
# ============================================================

USE_CSV_OPERATOR_SAMPLES = True
CSV_K_SAMPLES_PATH = ROOT / "figures" / "k_samples.csv"
CSV_MU_SAMPLES_PATH = ROOT / "figures" / "mu_samples.csv"
CSV_SAMPLE_INDEX_MODE = "one_based"

CSV_K1_SAMPLE_INDEX = 9
CSV_MU1_SAMPLE_INDEX = 9
CSV_K2_SAMPLE_INDEX = 10
CSV_MU2_SAMPLE_INDEX = 10

USE_CSV_ESTIMATOR_GUESSES = False
CSV_K1_HAT_SAMPLE_INDEX = 1
CSV_MU1_HAT_SAMPLE_INDEX = 1
CSV_K2_HAT_SAMPLE_INDEX = 2
CSV_MU2_HAT_SAMPLE_INDEX = 2

PLOT_RESULTS = True
PLOT_PREFIX = ROOT / "figures" / "sim"
ESTIMATOR_GUESS_SEED_OFFSET = 100


# ============================================================
# Controller builder
# ============================================================

def build_controller_from_estimates(
    plant: Plant,
    eq,
    k1_used: np.ndarray,
    mu1_used: np.ndarray,
    k2_used: np.ndarray,
    mu2_used: np.ndarray,
):
    ops = compute_all_species_operators(
        k1=k1_used,
        mu1=mu1_used,
        g1=plant.g1,
        k2=k2_used,
        mu2=mu2_used,
        g2=plant.g2,
        a=plant.a,
    )

    controller = NominalController(
        a=plant.a,
        zeta1=ops["zeta1"],
        zeta2=ops["zeta2"],
        gamma1=ops["gamma1"],
        gamma2=ops["gamma2"],
        kappa1=ops["kappa1"],
        kappa2=ops["kappa2"],
        pi0_1=ops["pi0_1"],
        pi0_2=ops["pi0_2"],
        x1_star_0=eq.x1_star_age0,
        x2_star_0=eq.x2_star_age0,
        beta=BETA,
        epsilon=EPSILON,
        denom_eps=CONTROL_DENOM_EPS,
    )
    return controller, ops


def evaluate_control(
    control_mode: str,
    constant_u_star: float,
    controller: NominalController | None,
    x1: np.ndarray,
    x2: np.ndarray,
) -> float:
    if control_mode == "constant":
        return float(constant_u_star)

    if control_mode == "nominal":
        if controller is None:
            raise ValueError("Nominal controller requested but controller is None.")
        return float(controller(x1, x2))

    raise ValueError(f"Unknown CONTROL_MODE: {control_mode}")


# ============================================================
# Simulator
# ============================================================


def _resample_to_age_grid(values: np.ndarray, source_grid: np.ndarray, target_grid: np.ndarray) -> np.ndarray:
    return np.interp(target_grid, source_grid, values).astype(float)


def _resolve_sample_row_index(index: int, n_available: int, label: str) -> int:
    if n_available <= 0:
        raise ValueError(f"No samples available for {label}.")

    if CSV_SAMPLE_INDEX_MODE == "zero_based":
        row_idx = index
    elif CSV_SAMPLE_INDEX_MODE == "one_based":
        row_idx = index - 1
    else:
        raise ValueError(
            f"Unsupported CSV_SAMPLE_INDEX_MODE={CSV_SAMPLE_INDEX_MODE!r}. "
            "Use 'zero_based' or 'one_based'."
        )

    if not 0 <= row_idx < n_available:
        raise IndexError(f"{label} sample index {index} is out of range.")
    return row_idx


def _load_csv_sample_profiles(
    csv_path: Path,
    sample_index: int,
    target_grid: np.ndarray,
    label: str,
) -> tuple[np.ndarray, int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"{label} sample CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{label} sample CSV is empty: {csv_path}")

    row_idx = _resolve_sample_row_index(sample_index, len(df), label)
    source_grid = np.linspace(0.0, 1.0, df.shape[1], dtype=float)
    values = df.iloc[row_idx].to_numpy(dtype=float)
    return _resample_to_age_grid(values, source_grid, target_grid), row_idx


def _apply_csv_samples_to_plant(plant: Plant) -> tuple[Plant, dict[str, float]]:
    k1, k1_row = _load_csv_sample_profiles(CSV_K_SAMPLES_PATH, CSV_K1_SAMPLE_INDEX, plant.a, "k1")
    mu1, mu1_row = _load_csv_sample_profiles(CSV_MU_SAMPLES_PATH, CSV_MU1_SAMPLE_INDEX, plant.a, "mu1")
    k2, k2_row = _load_csv_sample_profiles(CSV_K_SAMPLES_PATH, CSV_K2_SAMPLE_INDEX, plant.a, "k2")
    mu2, mu2_row = _load_csv_sample_profiles(CSV_MU_SAMPLES_PATH, CSV_MU2_SAMPLE_INDEX, plant.a, "mu2")

    zeta1 = float(G_LS(k1, mu1, plant.a))
    zeta2 = float(G_LS(k2, mu2, plant.a))
    if not np.isfinite(zeta1) or zeta1 <= 0.0:
        raise ValueError(f"Invalid zeta1 computed from selected CSV samples: {zeta1}")
    if not np.isfinite(zeta2) or zeta2 <= 0.0:
        raise ValueError(f"Invalid zeta2 computed from selected CSV samples: {zeta2}")

    n1 = stable_age_profile(mu1, zeta1, plant.a)
    n2 = stable_age_profile(mu2, zeta2, plant.a)
    gamma1 = G_gamma(plant.g1, zeta2, mu2, plant.a)
    gamma2 = G_gamma(plant.g2, zeta1, mu1, plant.a)
    updated = replace(
        plant,
        k1=k1,
        mu1=mu1,
        k2=k2,
        mu2=mu2,
        R0_1=reproductive_number(mu1, k1, plant.a),
        R0_2=reproductive_number(mu2, k2, plant.a),
        zeta1=zeta1,
        zeta2=zeta2,
        n1=n1,
        n2=n2,
        gamma1=gamma1,
        gamma2=gamma2,
    )
    info = {
        "k1_index": float(CSV_K1_SAMPLE_INDEX),
        "mu1_index": float(CSV_MU1_SAMPLE_INDEX),
        "k2_index": float(CSV_K2_SAMPLE_INDEX),
        "mu2_index": float(CSV_MU2_SAMPLE_INDEX),
        "k1_row": float(k1_row),
        "mu1_row": float(mu1_row),
        "k2_row": float(k2_row),
        "mu2_row": float(mu2_row),
    }
    return updated, info

def sample_estimator_initial_guesses(envelopes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if USE_CSV_ESTIMATOR_GUESSES:
        k1_guess, _ = _load_csv_sample_profiles(CSV_K_SAMPLES_PATH, CSV_K1_HAT_SAMPLE_INDEX, envelopes.a, "k1_hat")
        mu1_guess, _ = _load_csv_sample_profiles(CSV_MU_SAMPLES_PATH, CSV_MU1_HAT_SAMPLE_INDEX, envelopes.a, "mu1_hat")
        k2_guess, _ = _load_csv_sample_profiles(CSV_K_SAMPLES_PATH, CSV_K2_HAT_SAMPLE_INDEX, envelopes.a, "k2_hat")
        mu2_guess, _ = _load_csv_sample_profiles(CSV_MU_SAMPLES_PATH, CSV_MU2_HAT_SAMPLE_INDEX, envelopes.a, "mu2_hat")
        return k1_guess, mu1_guess, k2_guess, mu2_guess

    guess_sampler = PlantSampler(
        envelopes=envelopes,
        seed=SEED + ESTIMATOR_GUESS_SEED_OFFSET,
        target_R0_min=1.2,
    )
    k1_guess, mu1_guess, _, _, _, _ = guess_sampler.sample_one_species(guess_sampler.env.g1_ranges)
    k2_guess, mu2_guess, _, _, _, _ = guess_sampler.sample_one_species(guess_sampler.env.g2_ranges)
    return k1_guess, mu1_guess, k2_guess, mu2_guess


def simulate(
    plant: Plant,
    envelopes,
    eq,
    x1_0: np.ndarray,
    x2_0: np.ndarray,
    control_mode: str,
    constant_u_star: float,
    t_final: float,
    dt: float,
):
    a = plant.a
    da = a[1] - a[0]
    n_age = len(a)

    if abs(dt - da) > 1e-12:
        raise ValueError("This simulator assumes dt = da.")

    k1_guess, mu1_guess, k2_guess, mu2_guess = sample_estimator_initial_guesses(envelopes)

    # --------------------------------------------------------
    # Estimator initialization
    # --------------------------------------------------------
    k_est_cfg = KEstimatorConfig(
        adapt=ADAPT_K,
        Gamma_1=GAMMA_K_1,
        Gamma_2=GAMMA_K_2,
        project_to_nonnegative=True,
    )
    k_estimator = KEstimator(
        a=a,
        k1_init=k1_guess,
        k2_init=k2_guess,
        config=k_est_cfg,
    )

    mu_est_cfg = MuEstimatorConfig(
        adapt=ADAPT_MU,
        Gamma_1=MU_ADAPT_GAIN_1,
        Gamma_2=MU_ADAPT_GAIN_2,
        sigma_1=MU_FILTER_ALPHA_1,
        sigma_2=MU_FILTER_ALPHA_2,
        project_mu_to_nonnegative=True,
        eps_resource=EPS_RESOURCE,
    )
    mu_estimator = MuEstimator(
        a=a,
        mu1_init=mu1_guess,
        mu2_init=mu2_guess,
        x1_init=x1_0,
        x2_init=x2_0,
        config=mu_est_cfg,
    )

    def controller_inputs():
        k1_used = k_estimator.k1_hat if USE_ESTIMATED_K_IN_CONTROLLER else plant.k1
        k2_used = k_estimator.k2_hat if USE_ESTIMATED_K_IN_CONTROLLER else plant.k2
        mu1_used = mu_estimator.mu1_hat if USE_ESTIMATED_MU_IN_CONTROLLER else plant.mu1
        mu2_used = mu_estimator.mu2_hat if USE_ESTIMATED_MU_IN_CONTROLLER else plant.mu2
        return k1_used, mu1_used, k2_used, mu2_used

    if control_mode == "nominal":
        k1_used, mu1_used, k2_used, mu2_used = controller_inputs()
        controller, ops = build_controller_from_estimates(
            plant, eq, k1_used, mu1_used, k2_used, mu2_used
        )
    else:
        controller, ops = None, None

    n_steps = int(np.round(t_final / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)

    x1_hist = np.zeros((n_steps + 1, n_age))
    x2_hist = np.zeros((n_steps + 1, n_age))
    u_hist = np.zeros(n_steps + 1)

    births1_hist = np.zeros(n_steps + 1)
    births2_hist = np.zeros(n_steps + 1)
    pred_pressure_hist = np.zeros(n_steps + 1)
    resource_hist = np.zeros(n_steps + 1)

    # k-estimator histories
    k1_hat_hist = np.zeros((n_steps + 1, n_age))
    k2_hat_hist = np.zeros((n_steps + 1, n_age))
    k_err1_boundary = np.zeros(n_steps + 1)
    k_err2_boundary = np.zeros(n_steps + 1)

    # pointwise mu-estimator histories
    mu1_hat_hist = np.zeros((n_steps + 1, n_age))
    mu2_hat_hist = np.zeros((n_steps + 1, n_age))
    sigma1_hist = np.zeros((n_steps + 1, n_age))
    sigma2_hist = np.zeros((n_steps + 1, n_age))
    rho1_hist = np.zeros((n_steps + 1, n_age))
    rho2_hist = np.zeros((n_steps + 1, n_age))
    Y1_hist = np.zeros((n_steps + 1, n_age))
    Y2_hist = np.zeros((n_steps + 1, n_age))
    mu_regression_error1_hist = np.zeros((n_steps + 1, n_age))
    mu_regression_error2_hist = np.zeros((n_steps + 1, n_age))
    mu_regression_error1_norm_hist = np.zeros(n_steps + 1)
    mu_regression_error2_norm_hist = np.zeros(n_steps + 1)

    if control_mode == "nominal":
        ip_pi0_1_x1_hist = np.zeros(n_steps + 1)
        ip_pi0_2_x2_hist = np.zeros(n_steps + 1)
    else:
        ip_pi0_1_x1_hist = None
        ip_pi0_2_x2_hist = None

    x1_hist[0] = positive_clip(x1_0)
    x2_hist[0] = positive_clip(x2_0)

    k1_hat_hist[0] = k_estimator.k1_hat.copy()
    k2_hat_hist[0] = k_estimator.k2_hat.copy()

    mu1_hat_hist[0] = mu_estimator.mu1_hat.copy()
    mu2_hat_hist[0] = mu_estimator.mu2_hat.copy()
    sigma1_hist[0] = mu_estimator.sigma1.copy()
    sigma2_hist[0] = mu_estimator.sigma2.copy()
    rho1_hist[0] = mu_estimator.rho1.copy()
    rho2_hist[0] = mu_estimator.rho2.copy()
    Y1_hist[0] = mu_estimator.Y1.copy()
    Y2_hist[0] = mu_estimator.Y2.copy()

    for n in range(n_steps):
        x1 = x1_hist[n].copy()
        x2 = x2_hist[n].copy()

        if control_mode == "nominal":
            k1_used, mu1_used, k2_used, mu2_used = controller_inputs()
            controller, ops = build_controller_from_estimates(
                plant, eq, k1_used, mu1_used, k2_used, mu2_used
            )

        u = evaluate_control(control_mode, constant_u_star, controller, x1, x2)
        u_hist[n] = u

        pred_pressure = float(np.trapezoid(plant.g1 * x2, a))
        resource = float(np.trapezoid(plant.g2 * x1, a))

        pred_pressure_hist[n] = pred_pressure
        resource_hist[n] = resource

        births1 = float(np.trapezoid(plant.k1 * x1, a))
        births2 = float(np.trapezoid(plant.k2 * x2, a))

        births1_hist[n] = births1
        births2_hist[n] = births2

        if control_mode == "nominal":
            info = controller.diagnostics(x1, x2)
            ip_pi0_1_x1_hist[n] = info["ip_pi0_1_x1"]
            ip_pi0_2_x2_hist[n] = info["ip_pi0_2_x2"]

        # True plant dynamics
        lam1 = plant.mu1 + u + pred_pressure
        lam2 = plant.mu2 + u + 1.0 / (EPS_RESOURCE + resource)

        x1_next = np.zeros_like(x1)
        x2_next = np.zeros_like(x2)

        x1_next[0] = births1
        x2_next[0] = births2

        x1_next[1:] = x1[:-1] * np.exp(-dt * lam1[:-1])
        x2_next[1:] = x2[:-1] * np.exp(-dt * lam2[:-1])

        x1_next = positive_clip(x1_next)
        x2_next = positive_clip(x2_next)

        x1_hist[n + 1] = x1_next
        x2_hist[n + 1] = x2_next

        # ---- update k-estimator
        k_info = k_estimator.step(x1, x2, dt)
        k1_hat_hist[n + 1] = k_info["k1_hat"]
        k2_hat_hist[n + 1] = k_info["k2_hat"]
        k_err1_boundary[n] = k_info["err1"]
        k_err2_boundary[n] = k_info["err2"]

        # ---- update scalar mu-estimator
        mu_info = mu_estimator.step(
            x1=x1,
            x2=x2,
            g1=plant.g1,
            g2=plant.g2,
            u=u,
            dt=dt,
        )

        mu1_hat_hist[n + 1] = mu_info["mu1_hat"]
        mu2_hat_hist[n + 1] = mu_info["mu2_hat"]
        sigma1_hist[n + 1] = mu_info["sigma1"]
        sigma2_hist[n + 1] = mu_info["sigma2"]
        rho1_hist[n + 1] = mu_info["rho1"]
        rho2_hist[n + 1] = mu_info["rho2"]
        Y1_hist[n + 1] = mu_info["Y1"]
        Y2_hist[n + 1] = mu_info["Y2"]
        mu_regression_error1_hist[n] = mu_info["regression_error1"]
        mu_regression_error2_hist[n] = mu_info["regression_error2"]
        mu_regression_error1_norm_hist[n] = mu_info["regression_error1_norm"]
        mu_regression_error2_norm_hist[n] = mu_info["regression_error2_norm"]

    # final diagnostics
    x1_final = x1_hist[-1]
    x2_final = x2_hist[-1]

    if control_mode == "nominal":
        k1_used, mu1_used, k2_used, mu2_used = controller_inputs()
        controller, ops = build_controller_from_estimates(
            plant, eq, k1_used, mu1_used, k2_used, mu2_used
        )

    u_hist[-1] = evaluate_control(control_mode, constant_u_star, controller, x1_final, x2_final)
    pred_pressure_hist[-1] = float(np.trapezoid(plant.g1 * x2_final, a))
    resource_hist[-1] = float(np.trapezoid(plant.g2 * x1_final, a))
    births1_hist[-1] = float(np.trapezoid(plant.k1 * x1_final, a))
    births2_hist[-1] = float(np.trapezoid(plant.k2 * x2_final, a))

    k_final = k_estimator.step(x1_final, x2_final, 0.0)
    k_err1_boundary[-1] = k_final["err1"]
    k_err2_boundary[-1] = k_final["err2"]

    mu_final = mu_estimator.step(
        x1=x1_final,
        x2=x2_final,
        g1=plant.g1,
        g2=plant.g2,
        u=u_hist[-1],
        dt=0.0,
    )
    mu_regression_error1_hist[-1] = mu_final["regression_error1"]
    mu_regression_error2_hist[-1] = mu_final["regression_error2"]
    mu_regression_error1_norm_hist[-1] = mu_final["regression_error1_norm"]
    mu_regression_error2_norm_hist[-1] = mu_final["regression_error2_norm"]

    if control_mode == "nominal":
        info = controller.diagnostics(x1_final, x2_final)
        ip_pi0_1_x1_hist[-1] = info["ip_pi0_1_x1"]
        ip_pi0_2_x2_hist[-1] = info["ip_pi0_2_x2"]

    return {
        "a": a,
        "times": times,
        "x1": x1_hist,
        "x2": x2_hist,
        "u": u_hist,
        "births1": births1_hist,
        "births2": births2_hist,
        "pred_pressure": pred_pressure_hist,
        "resource": resource_hist,
        "ip_pi0_1_x1": ip_pi0_1_x1_hist,
        "ip_pi0_2_x2": ip_pi0_2_x2_hist,

        # k estimator
        "k1_hat": k1_hat_hist,
        "k2_hat": k2_hat_hist,
        "k1_true": plant.k1.copy(),
        "k2_true": plant.k2.copy(),
        "k_err1_boundary": k_err1_boundary,
        "k_err2_boundary": k_err2_boundary,

        # scalar mu estimator
        "mu1_hat": mu1_hat_hist,
        "mu2_hat": mu2_hat_hist,
        "mu1_true": plant.mu1.copy(),
        "mu2_true": plant.mu2.copy(),
        "sigma1": sigma1_hist,
        "sigma2": sigma2_hist,
        "rho1": rho1_hist,
        "rho2": rho2_hist,
        "Y1": Y1_hist,
        "Y2": Y2_hist,
        "mu_regression_error1": mu_regression_error1_hist,
        "mu_regression_error2": mu_regression_error2_hist,
        "mu_regression_error1_norm": mu_regression_error1_norm_hist,
        "mu_regression_error2_norm": mu_regression_error2_norm_hist,

        "adapt_k": ADAPT_K,
        "adapt_mu": ADAPT_MU,
        "Gamma_k_1": GAMMA_K_1,
        "Gamma_k_2": GAMMA_K_2,
        "mu_adapt_gain_1": MU_ADAPT_GAIN_1,
        "mu_adapt_gain_2": MU_ADAPT_GAIN_2,
        "mu_filter_alpha_1": MU_FILTER_ALPHA_1,
        "mu_filter_alpha_2": MU_FILTER_ALPHA_2,
        "use_estimated_k_in_controller": USE_ESTIMATED_K_IN_CONTROLLER,
        "use_estimated_mu_in_controller": USE_ESTIMATED_MU_IN_CONTROLLER,
    }


# ============================================================
# Main
# ============================================================

def main():
    env_gen = EnvelopeGenerator(seed=SEED, n_grid=N_AGE, target_R0_min=1.2)
    envelopes = env_gen.build()

    sampler = PlantSampler(envelopes=envelopes, seed=SEED + 1, target_R0_min=1.2)
    plant = sampler.sample_plant()
    csv_sample_info = None
    if USE_CSV_OPERATOR_SAMPLES:
        plant, csv_sample_info = _apply_csv_samples_to_plant(plant)

    u_star = 0.5 * min(plant.zeta1, plant.zeta2)

    eq = validate_equilibrium_values(plant, u_star)

    if USE_EQUILIBRIUM_INITIAL_CONDITIONS:
        x1_0 = eq.x1_star_age0 * plant.n1
        x2_0 = eq.x2_star_age0 * plant.n2
        ic_mode_str = "equilibrium"
    else:
        x1_0, x2_0 = sample_initial_conditions(
            plant,
            x1_star_age0=eq.x1_star_age0,
            x2_star_age0=eq.x2_star_age0,
            seed=SEED + 2,
        )
        ic_mode_str = "sampled"

    sim = simulate(
        plant=plant,
        envelopes=envelopes,
        eq=eq,
        x1_0=x1_0,
        x2_0=x2_0,
        control_mode=CONTROL_MODE,
        constant_u_star=u_star,
        t_final=T_FINAL,
        dt=DT,
    )

    print("Sampled plant:")
    print(f"  R0_1    = {plant.R0_1:.6f}")
    print(f"  R0_2    = {plant.R0_2:.6f}")
    print(f"  zeta1   = {plant.zeta1:.6f}")
    print(f"  zeta2   = {plant.zeta2:.6f}")
    print(f"  gamma1  = {plant.gamma1:.6f}")
    print(f"  gamma2  = {plant.gamma2:.6f}")
    if csv_sample_info is not None:
        print(f"  selected k1 sample index       = {int(csv_sample_info['k1_index'])}")
        print(f"  selected mu1 sample index      = {int(csv_sample_info['mu1_index'])}")
        print(f"  selected k2 sample index       = {int(csv_sample_info['k2_index'])}")
        print(f"  selected mu2 sample index      = {int(csv_sample_info['mu2_index'])}")
        print(f"  resolved k1 CSV row            = {int(csv_sample_info['k1_row'])}")
        print(f"  resolved mu1 CSV row           = {int(csv_sample_info['mu1_row'])}")
        print(f"  resolved k2 CSV row            = {int(csv_sample_info['k2_row'])}")
        print(f"  resolved mu2 CSV row           = {int(csv_sample_info['mu2_row'])}")
        print(f"  g1/g2 sampled from seed        = {SEED + 1}")
    print()
    print("Equilibrium checks:")
    print(f"  u_star  = {eq.u_star:.6f}")
    print(f"  x1*(0)  = {eq.x1_star_age0:.6f}")
    print(f"  x2*(0)  = {eq.x2_star_age0:.6f}")
    print(f"  1/(zeta2*gamma2) = {eq.lower_bound_x1:.6f}")
    print(f"  check x1*(0) > 1/(zeta2*gamma2): {eq.valid_x1_lower_bound}")
    print()
    print("Simulation choices:")
    print(f"  initial condition mode         = {ic_mode_str}")
    print(f"  control mode                   = {CONTROL_MODE}")
    print(f"  adapt k                        = {ADAPT_K}")
    print(f"  adapt mu                       = {ADAPT_MU}")
    print(f"  use k_hat in controller        = {USE_ESTIMATED_K_IN_CONTROLLER}")
    print(f"  use mu_hat in controller       = {USE_ESTIMATED_MU_IN_CONTROLLER}")
    print(f"  CSV plant samples              = {USE_CSV_OPERATOR_SAMPLES}")
    print(f"  CSV estimator init             = {USE_CSV_ESTIMATOR_GUESSES}")
    if USE_CSV_ESTIMATOR_GUESSES:
        print(f"  k1_hat sample index            = {CSV_K1_HAT_SAMPLE_INDEX}")
        print(f"  mu1_hat sample index           = {CSV_MU1_HAT_SAMPLE_INDEX}")
        print(f"  k2_hat sample index            = {CSV_K2_HAT_SAMPLE_INDEX}")
        print(f"  mu2_hat sample index           = {CSV_MU2_HAT_SAMPLE_INDEX}")
    print()
    print("Diagnostics:")
    print(f"  final k estimator error 1      = {sim['k_err1_boundary'][-1]:.6e}")
    print(f"  final k estimator error 2      = {sim['k_err2_boundary'][-1]:.6e}")
    print(f"  final mu regression error 1    = {sim['mu_regression_error1_norm'][-1]:.6e}")
    print(f"  final mu regression error 2    = {sim['mu_regression_error2_norm'][-1]:.6e}")

    if PLOT_RESULTS:
        plot_simulation_results(
            plant,
            sim,
            eq=eq,
            u_star=u_star,
            show=True,
            prefix=PLOT_PREFIX,
            include_heatmaps=True,
            include_eta_control=True,
            include_estimator=True,
        )


if __name__ == "__main__":
    main()
