from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for search_path in (ROOT, SRC):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

import run_sim as rs
from equilibrium_checks import validate_equilibrium_values
from operators import G_LS, compute_all_species_operators
from plant import EnvelopeGenerator, PlantSampler, positive_clip, sample_initial_conditions
from estimate_k import KEstimator, KEstimatorConfig
from estimate_mu import MuEstimator, MuEstimatorConfig
from plotting import compute_eta_series


# ============================================================
# Dataset settings
# ============================================================

N_ADAPTIVE_ROLLOUTS = 100
N_ESTIMATE_SNAPSHOTS_PER_ROLLOUT = 200

ADAPTIVE_DATASET_NPZ_PATH = ROOT / "datasets" / "adaptive_estimator_dataset.npz"
ADAPTIVE_DATASET_CSV_PATH = ROOT / "datasets" / "adaptive_estimator_summary.csv"
WORST_ROLLOUT_FIG_PATH = ROOT / "figures" / "adaptive_dataset_worst_rollout.png"
PARALLELIZE_ROLLOUTS = True
MAX_WORKERS = 10
print(MAX_WORKERS, "workers will be used for rollout parallelization.")
PROGRESS_EVERY = 5
ROLLOUT_STEP_PROGRESS_EVERY = 2000
DATASET_RANDOMIZE_HAT_INITIALIZATIONS = True


def sample_estimator_initial_guesses(envelopes, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not DATASET_RANDOMIZE_HAT_INITIALIZATIONS and rs.USE_CSV_ESTIMATOR_GUESSES:
        return rs.sample_estimator_initial_guesses(envelopes)

    guess_sampler = PlantSampler(
        envelopes=envelopes,
        seed=seed,
        target_R0_min=1.2,
    )
    k1_guess, mu1_guess, _, _, _, _ = guess_sampler.sample_one_species(guess_sampler.env.g1_ranges)
    k2_guess, mu2_guess, _, _, _, _ = guess_sampler.sample_one_species(guess_sampler.env.g2_ranges)
    return k1_guess, mu1_guess, k2_guess, mu2_guess


def choose_snapshot_indices(n_times: int, n_samples: int) -> np.ndarray:
    n_take = min(max(1, n_samples), n_times)
    idx = np.linspace(0, n_times - 1, n_take)
    idx = np.unique(np.round(idx).astype(int))
    if idx.size < n_take:
        missing = [j for j in range(n_times) if j not in set(idx.tolist())]
        idx = np.concatenate([idx, np.array(missing[: n_take - idx.size], dtype=int)])
        idx = np.sort(idx)
    return idx


def finite_estimator_zetas(
    a: np.ndarray,
    k1_hat: np.ndarray,
    mu1_hat: np.ndarray,
    k2_hat: np.ndarray,
    mu2_hat: np.ndarray,
) -> tuple[bool, float, float]:
    zeta1 = G_LS(k1_hat, mu1_hat, a)
    zeta2 = G_LS(k2_hat, mu2_hat, a)
    valid = np.isfinite(zeta1) and np.isfinite(zeta2)
    return bool(valid), float(zeta1), float(zeta2)


def simulate_with_initial_guesses(
    rollout_id: int,
    plant,
    envelopes,
    eq,
    x1_0: np.ndarray,
    x2_0: np.ndarray,
    control_mode: str,
    constant_u_star: float,
    t_final: float,
    dt: float,
    k1_guess: np.ndarray,
    mu1_guess: np.ndarray,
    k2_guess: np.ndarray,
    mu2_guess: np.ndarray,
):
    a = plant.a
    da = a[1] - a[0]
    n_age = len(a)

    if abs(dt - da) > 1e-12:
        raise ValueError("This simulator assumes dt = da.")

    k_est_cfg = KEstimatorConfig(
        adapt=rs.ADAPT_K,
        Gamma_1=rs.GAMMA_K_1,
        Gamma_2=rs.GAMMA_K_2,
        project_to_nonnegative=True,
    )
    k_estimator = KEstimator(
        a=a,
        k1_init=k1_guess,
        k2_init=k2_guess,
        config=k_est_cfg,
    )

    mu_est_cfg = MuEstimatorConfig(
        adapt=rs.ADAPT_MU,
        Gamma_1=rs.MU_ADAPT_GAIN_1,
        Gamma_2=rs.MU_ADAPT_GAIN_2,
        sigma_1=rs.MU_FILTER_ALPHA_1,
        sigma_2=rs.MU_FILTER_ALPHA_2,
        project_mu_to_nonnegative=True,
        eps_resource=rs.EPS_RESOURCE,
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
        k1_used = k_estimator.k1_hat if rs.USE_ESTIMATED_K_IN_CONTROLLER else plant.k1
        k2_used = k_estimator.k2_hat if rs.USE_ESTIMATED_K_IN_CONTROLLER else plant.k2
        mu1_used = mu_estimator.mu1_hat if rs.USE_ESTIMATED_MU_IN_CONTROLLER else plant.mu1
        mu2_used = mu_estimator.mu2_hat if rs.USE_ESTIMATED_MU_IN_CONTROLLER else plant.mu2
        return k1_used, mu1_used, k2_used, mu2_used

    if control_mode == "nominal":
        k1_used, mu1_used, k2_used, mu2_used = controller_inputs()
        controller, _ = rs.build_controller_from_estimates(
            plant, eq, k1_used, mu1_used, k2_used, mu2_used
        )
    else:
        controller = None

    n_steps = int(np.round(t_final / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)

    x1_hist = np.zeros((n_steps + 1, n_age))
    x2_hist = np.zeros((n_steps + 1, n_age))
    u_hist = np.zeros(n_steps + 1)
    k1_hat_hist = np.zeros((n_steps + 1, n_age))
    k2_hat_hist = np.zeros((n_steps + 1, n_age))
    mu1_hat_hist = np.zeros((n_steps + 1, n_age))
    mu2_hat_hist = np.zeros((n_steps + 1, n_age))

    x1_hist[0] = positive_clip(x1_0)
    x2_hist[0] = positive_clip(x2_0)
    k1_hat_hist[0] = k_estimator.k1_hat.copy()
    k2_hat_hist[0] = k_estimator.k2_hat.copy()
    mu1_hat_hist[0] = mu_estimator.mu1_hat.copy()
    mu2_hat_hist[0] = mu_estimator.mu2_hat.copy()

    valid, _, _ = finite_estimator_zetas(
        a,
        k_estimator.k1_hat,
        mu_estimator.mu1_hat,
        k_estimator.k2_hat,
        mu_estimator.mu2_hat,
    )
    if not valid:
        return {"valid": False, "failure_step": 0}

    for n in range(n_steps):
        if n % ROLLOUT_STEP_PROGRESS_EVERY == 0:
            print(
                f"[rollout {rollout_id}] step {n}/{n_steps} | "
                f"t={times[n]:.6f}"
            )

        x1 = x1_hist[n].copy()
        x2 = x2_hist[n].copy()

        if control_mode == "nominal":
            k1_used, mu1_used, k2_used, mu2_used = controller_inputs()
            try:
                controller, _ = rs.build_controller_from_estimates(
                    plant, eq, k1_used, mu1_used, k2_used, mu2_used
                )
            except ValueError:
                return {"valid": False, "failure_step": n}

        u = rs.evaluate_control(control_mode, constant_u_star, controller, x1, x2)
        u_hist[n] = u

        pred_pressure = float(np.trapezoid(plant.g1 * x2, a))
        resource = float(np.trapezoid(plant.g2 * x1, a))
        births1 = float(np.trapezoid(plant.k1 * x1, a))
        births2 = float(np.trapezoid(plant.k2 * x2, a))

        lam1 = plant.mu1 + u + pred_pressure
        lam2 = plant.mu2 + u + 1.0 / (rs.EPS_RESOURCE + resource)

        x1_next = np.zeros_like(x1)
        x2_next = np.zeros_like(x2)
        x1_next[0] = births1
        x2_next[0] = births2
        x1_next[1:] = x1[:-1] * np.exp(-dt * lam1[:-1])
        x2_next[1:] = x2[:-1] * np.exp(-dt * lam2[:-1])

        x1_hist[n + 1] = positive_clip(x1_next)
        x2_hist[n + 1] = positive_clip(x2_next)

        k_info = k_estimator.step(x1, x2, dt)
        k1_hat_hist[n + 1] = k_info["k1_hat"]
        k2_hat_hist[n + 1] = k_info["k2_hat"]

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

        valid, _, _ = finite_estimator_zetas(
            a,
            k_estimator.k1_hat,
            mu_estimator.mu1_hat,
            k_estimator.k2_hat,
            mu_estimator.mu2_hat,
        )
        if not valid:
            return {"valid": False, "failure_step": n + 1}

    if control_mode == "nominal":
        x1_final = x1_hist[-1]
        x2_final = x2_hist[-1]
        k1_used, mu1_used, k2_used, mu2_used = controller_inputs()
        try:
            controller, _ = rs.build_controller_from_estimates(
                plant, eq, k1_used, mu1_used, k2_used, mu2_used
            )
        except ValueError:
            return {"valid": False, "failure_step": n_steps}
        u_hist[-1] = rs.evaluate_control(control_mode, constant_u_star, controller, x1_final, x2_final)

    return {
        "valid": True,
        "a": a,
        "times": times,
        "x1": x1_hist,
        "x2": x2_hist,
        "u": u_hist,
        "k1_hat": k1_hat_hist,
        "k2_hat": k2_hat_hist,
        "mu1_hat": mu1_hat_hist,
        "mu2_hat": mu2_hat_hist,
        "k1_guess": k1_guess.copy(),
        "k2_guess": k2_guess.copy(),
        "mu1_guess": mu1_guess.copy(),
        "mu2_guess": mu2_guess.copy(),
        "plant_zeta1": float(plant.zeta1),
        "plant_zeta2": float(plant.zeta2),
    }


def run_single_rollout(
    rollout_id: int,
) -> dict:
    env_seed = rs.SEED + 1000 * rollout_id
    env_gen = EnvelopeGenerator(seed=env_seed, n_grid=rs.N_AGE, target_R0_min=1.2)
    envelopes = env_gen.build()

    plant_seed = env_seed + 1
    sampler = PlantSampler(envelopes=envelopes, seed=plant_seed, target_R0_min=1.2)
    plant = sampler.sample_plant()
    if rs.USE_CSV_OPERATOR_SAMPLES:
        plant, _ = rs._apply_csv_samples_to_plant(plant)

    u_star = 0.5 * min(plant.zeta1, plant.zeta2)
    eq = validate_equilibrium_values(plant, u_star)

    if rs.USE_EQUILIBRIUM_INITIAL_CONDITIONS:
        x1_0 = eq.x1_star_age0 * plant.n1
        x2_0 = eq.x2_star_age0 * plant.n2
    else:
        x1_0, x2_0 = sample_initial_conditions(
            plant,
            x1_star_age0=eq.x1_star_age0,
            x2_star_age0=eq.x2_star_age0,
            seed=env_seed + 2,
        )

    guess_seed = env_seed + rs.ESTIMATOR_GUESS_SEED_OFFSET
    k1_guess, mu1_guess, k2_guess, mu2_guess = sample_estimator_initial_guesses(envelopes, guess_seed)

    sim = simulate_with_initial_guesses(
        rollout_id=rollout_id,
        plant=plant,
        envelopes=envelopes,
        eq=eq,
        x1_0=x1_0,
        x2_0=x2_0,
        control_mode=rs.CONTROL_MODE,
        constant_u_star=u_star,
        t_final=rs.T_FINAL,
        dt=rs.DT,
        k1_guess=k1_guess,
        mu1_guess=mu1_guess,
        k2_guess=k2_guess,
        mu2_guess=mu2_guess,
    )
    if not sim["valid"]:
        return {
            "rollout_id": rollout_id,
            "valid": False,
            "failure_step": int(sim["failure_step"]),
        }

    eta1, eta2 = compute_eta_series(sim, plant, eq)
    sample_idx = choose_snapshot_indices(len(sim["times"]), N_ESTIMATE_SNAPSHOTS_PER_ROLLOUT)

    rows: list[dict[str, float | int]] = []
    k_hat_samples: list[np.ndarray] = []
    mu_hat_samples: list[np.ndarray] = []
    zeta_samples: list[float] = []

    for snap_order, idx in enumerate(sample_idx):
        t = float(sim["times"][idx])
        for species in (1, 2):
            k_hat = sim[f"k{species}_hat"][idx].copy()
            mu_hat = sim[f"mu{species}_hat"][idx].copy()
            zeta = G_LS(k_hat, mu_hat, sim["a"])
            if not np.isfinite(zeta):
                continue

            k_hat_samples.append(k_hat)
            mu_hat_samples.append(mu_hat)
            zeta_samples.append(float(zeta))
            rows.append(
                {
                    "rollout_id": rollout_id,
                    "snapshot_order": snap_order,
                    "time_index": int(idx),
                    "time": t,
                    "species": species,
                    "zeta": float(zeta),
                    "u_star": float(u_star),
                    "plant_zeta1": float(sim["plant_zeta1"]),
                    "plant_zeta2": float(sim["plant_zeta2"]),
                }
            )

    return {
        "rollout_id": rollout_id,
        "valid": True,
        "a": sim["a"].copy(),
        "rows": rows,
        "k_hat_samples": k_hat_samples,
        "mu_hat_samples": mu_hat_samples,
        "zeta_samples": zeta_samples,
        "final_eta1": float(eta1[-1]),
        "final_eta2": float(eta2[-1]),
        "times": sim["times"].copy(),
        "eta1": eta1.copy(),
        "eta2": eta2.copy(),
        "u": sim["u"].copy(),
    }


def build_adaptive_dataset() -> tuple[dict, pd.DataFrame]:
    rows: list[dict[str, float | int]] = []
    k_samples: list[np.ndarray] = []
    mu_samples: list[np.ndarray] = []
    zeta_samples: list[float] = []
    final_eta1_values: list[float] = []
    final_eta2_values: list[float] = []
    worst_rollout_plot_data: dict | None = None
    worst_rollout_score = -np.inf
    discarded_rollouts = 0
    start_time = time.perf_counter()
    a_grid = None

    def ingest_rollout_result(result: dict) -> None:
        nonlocal worst_rollout_plot_data, worst_rollout_score, a_grid, discarded_rollouts
        if not result["valid"]:
            discarded_rollouts += 1
            return
        if a_grid is None:
            a_grid = result["a"].copy()
        final_eta1_values.append(result["final_eta1"])
        final_eta2_values.append(result["final_eta2"])
        rollout_score = result["final_eta1"] + result["final_eta2"]
        if rollout_score > worst_rollout_score:
            worst_rollout_score = rollout_score
            worst_rollout_plot_data = {
                "rollout_id": result["rollout_id"],
                "times": result["times"],
                "eta1": result["eta1"],
                "eta2": result["eta2"],
                "u": result["u"],
            }
        k_samples.extend(result["k_hat_samples"])
        mu_samples.extend(result["mu_hat_samples"])
        zeta_samples.extend(result["zeta_samples"])
        rows.extend(result["rows"])

    if PARALLELIZE_ROLLOUTS and N_ADAPTIVE_ROLLOUTS > 1:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(run_single_rollout, rollout_id): rollout_id
                for rollout_id in range(N_ADAPTIVE_ROLLOUTS)
            }
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                ingest_rollout_result(result)
                completed += 1
                if completed % PROGRESS_EVERY == 0 or completed == N_ADAPTIVE_ROLLOUTS:
                    elapsed = time.perf_counter() - start_time
                    avg = elapsed / completed
                    remaining = avg * (N_ADAPTIVE_ROLLOUTS - completed)
                    print(
                        f"Completed {completed}/{N_ADAPTIVE_ROLLOUTS} rollouts | "
                        f"elapsed {elapsed:.1f}s | eta {remaining:.1f}s"
                    )
    else:
        for rollout_id in range(N_ADAPTIVE_ROLLOUTS):
            result = run_single_rollout(rollout_id)
            ingest_rollout_result(result)
            completed = rollout_id + 1
            if completed % PROGRESS_EVERY == 0 or completed == N_ADAPTIVE_ROLLOUTS:
                elapsed = time.perf_counter() - start_time
                avg = elapsed / completed
                remaining = avg * (N_ADAPTIVE_ROLLOUTS - completed)
                print(
                    f"Completed {completed}/{N_ADAPTIVE_ROLLOUTS} rollouts | "
                    f"elapsed {elapsed:.1f}s | eta {remaining:.1f}s"
                )

    if not rows:
        raise RuntimeError("No valid adaptive estimator samples were collected.")

    for sample_id, row in enumerate(rows):
        row["sample_id"] = sample_id

    arrays = {
        "a": a_grid,
        "k_hat": np.stack(k_samples, axis=0),
        "mu_hat": np.stack(mu_samples, axis=0),
        "zeta": np.array(zeta_samples, dtype=float),
        "final_eta1": np.array(final_eta1_values, dtype=float),
        "final_eta2": np.array(final_eta2_values, dtype=float),
    }
    df = pd.DataFrame(rows)
    if worst_rollout_plot_data is not None:
        arrays["worst_rollout_id"] = np.array([worst_rollout_plot_data["rollout_id"]], dtype=int)
        arrays["worst_rollout_times"] = worst_rollout_plot_data["times"]
        arrays["worst_rollout_eta1"] = worst_rollout_plot_data["eta1"]
        arrays["worst_rollout_eta2"] = worst_rollout_plot_data["eta2"]
        arrays["worst_rollout_u"] = worst_rollout_plot_data["u"]
    arrays["discarded_rollouts"] = np.array([discarded_rollouts], dtype=int)
    return arrays, df


def plot_worst_rollout_diagnostics(arrays: dict, save_path: str) -> None:
    times = arrays["worst_rollout_times"]
    eta1 = arrays["worst_rollout_eta1"]
    eta2 = arrays["worst_rollout_eta2"]
    u = arrays["worst_rollout_u"]
    rollout_id = int(arrays["worst_rollout_id"][0])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(times, eta1, label=r"$\eta_1$", color="tab:blue")
    ax1.plot(times, eta2, label=r"$\eta_2$", color="tab:orange")
    ax1.set_xlabel("time")
    ax1.set_ylabel(r"$\eta_i$")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(times, u, label=r"$u$", color="black", linestyle="--")
    ax2.set_ylabel(r"$u(t)$")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best", frameon=False)
    ax1.set_title(f"Worst rollout {rollout_id}: eta and dilution")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    arrays, df = build_adaptive_dataset()

    Path(ADAPTIVE_DATASET_NPZ_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(ADAPTIVE_DATASET_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ADAPTIVE_DATASET_NPZ_PATH, **arrays)
    df.to_csv(ADAPTIVE_DATASET_CSV_PATH, index=False)
    if "worst_rollout_times" in arrays:
        plot_worst_rollout_diagnostics(arrays, WORST_ROLLOUT_FIG_PATH)

    print(f"Saved adaptive dataset with {len(df)} samples.")
    print(f"NPZ dataset: {ADAPTIVE_DATASET_NPZ_PATH}")
    print(f"CSV summary: {ADAPTIVE_DATASET_CSV_PATH}")
    print(f"Discarded rollouts: {int(arrays['discarded_rollouts'][0])}")
    if "worst_rollout_times" in arrays:
        print(f"Worst-rollout diagnostics figure: {WORST_ROLLOUT_FIG_PATH}")
    print()
    print(df[["rollout_id", "time", "species", "zeta"]].describe(include="all"))
    print()
    print("Final eta statistics across rollouts:")
    print(
        f"  eta_1 final mean = {float(np.mean(arrays['final_eta1'])):.6e}, "
        f"max = {float(np.max(arrays['final_eta1'])):.6e}"
    )
    print(
        f"  eta_2 final mean = {float(np.mean(arrays['final_eta2'])):.6e}, "
        f"max = {float(np.max(arrays['final_eta2'])):.6e}"
    )


if __name__ == "__main__":
    main()
