from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# Internal helpers
# ============================================================


def _maybe_savefig(save_path: str | None) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")


def _finish_plot(show: bool) -> None:
    if show:
        plt.show()
    else:
        plt.close()


def _equilibrium_profiles(plant, eq) -> tuple[np.ndarray, np.ndarray]:
    x1_star = eq.x1_star_age0 * plant.n1
    x2_star = eq.x2_star_age0 * plant.n2
    return x1_star, x2_star


def compute_eta_series(sim: dict, plant, eq, denom_eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    a = sim["a"]
    x1 = sim["x1"]
    x2 = sim["x2"]

    x1_star, x2_star = _equilibrium_profiles(plant, eq)

    denom1 = max(float(np.trapezoid(x1_star * x1_star, a)), denom_eps)
    denom2 = max(float(np.trapezoid(x2_star * x2_star, a)), denom_eps)

    num1 = np.array([float(np.trapezoid(row * x1_star, a)) for row in x1])
    num2 = np.array([float(np.trapezoid(row * x2_star, a)) for row in x2])

    eta1 = np.log(np.maximum(num1 / denom1, denom_eps))
    eta2 = np.log(np.maximum(num2 / denom2, denom_eps))
    return eta1, eta2


def compute_k_error_series(sim: dict) -> tuple[np.ndarray, np.ndarray]:
    a = sim["a"]
    k1_hat = sim["k1_hat"]
    k2_hat = sim["k2_hat"]
    k1_true = sim["k1_true"]
    k2_true = sim["k2_true"]

    denom1 = max(float(np.sqrt(np.trapezoid(k1_true * k1_true, a))), 1e-12)
    denom2 = max(float(np.sqrt(np.trapezoid(k2_true * k2_true, a))), 1e-12)

    err1 = np.array([
        float(np.sqrt(np.trapezoid((row - k1_true) ** 2, a))) / denom1
        for row in k1_hat
    ])
    err2 = np.array([
        float(np.sqrt(np.trapezoid((row - k2_true) ** 2, a))) / denom2
        for row in k2_hat
    ])
    return err1, err2


def compute_mu_error_series(sim: dict) -> tuple[np.ndarray, np.ndarray]:
    a = sim["a"]
    mu1_hat = sim["mu1_hat"]
    mu2_hat = sim["mu2_hat"]
    mu1_true = sim["mu1_true"]
    mu2_true = sim["mu2_true"]

    denom1 = max(float(np.sqrt(np.trapezoid(mu1_true * mu1_true, a))), 1e-12)
    denom2 = max(float(np.sqrt(np.trapezoid(mu2_true * mu2_true, a))), 1e-12)

    err1 = np.array([
        float(np.sqrt(np.trapezoid((row - mu1_true) ** 2, a))) / denom1
        for row in mu1_hat
    ])
    err2 = np.array([
        float(np.sqrt(np.trapezoid((row - mu2_true) ** 2, a))) / denom2
        for row in mu2_hat
    ])
    return err1, err2


def compute_omega_norm_series(sim: dict) -> tuple[np.ndarray, np.ndarray]:
    a = sim["a"]
    sigma1 = sim["sigma1"]
    sigma2 = sim["sigma2"]

    norm1 = np.array([float(np.sqrt(np.trapezoid(row * row, a))) for row in sigma1])
    norm2 = np.array([float(np.sqrt(np.trapezoid(row * row, a))) for row in sigma2])
    return norm1, norm2


# ============================================================
# Envelope / sample plots
# ============================================================

def plot_samples(
    a: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    samples: np.ndarray,
    title: str,
    ylabel: str,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(8, 5))
    for j in range(samples.shape[0]):
        plt.plot(a, samples[j], alpha=0.30, linewidth=1.0)
    plt.plot(a, lower, linewidth=2.5, label="lower envelope")
    plt.plot(a, upper, linewidth=2.5, label="upper envelope")
    plt.xlabel("age")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    _maybe_savefig(save_path)
    _finish_plot(show)


def plot_r0_histogram(
    r0_values: np.ndarray,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(r0_values, bins=25, alpha=0.8)
    plt.xlabel("R0")
    plt.ylabel("count")
    plt.title("Distribution of sampled R0 values")
    plt.grid(True, alpha=0.3)
    _maybe_savefig(save_path)
    _finish_plot(show)


# ============================================================
# Plant coefficient plots
# ============================================================

def plot_plant_coefficients(
    plant,
    show: bool = True,
    prefix: str | None = None,
) -> None:
    a = plant.a

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(a, plant.k1, label="k1")
    axes[0].plot(a, plant.mu1, label="mu1")
    axes[0].plot(a, plant.g1, label="g1")
    axes[0].set_title("Species 1 coefficients")
    axes[0].set_ylabel("value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(a, plant.k2, label="k2")
    axes[1].plot(a, plant.mu2, label="mu2")
    axes[1].plot(a, plant.g2, label="g2")
    axes[1].set_title("Species 2 coefficients")
    axes[1].set_xlabel("age")
    axes[1].set_ylabel("value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_savefig(f"{prefix}_coeffs.png" if prefix else None)
    _finish_plot(show)


# ============================================================
# Simulation plots
# ============================================================

def plot_population_totals(
    sim: dict,
    show: bool = True,
    prefix: str | None = None,
) -> None:
    a = sim["a"]
    times = sim["times"]
    x1 = sim["x1"]
    x2 = sim["x2"]
    u = sim["u"]

    total_x1 = np.trapezoid(x1, a, axis=1)
    total_x2 = np.trapezoid(x2, a, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(times, total_x1, label="total x1")
    axes[0].plot(times, total_x2, label="total x2")
    axes[0].set_title("Total populations")
    axes[0].set_ylabel("total mass")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, u, label="u(t)")
    axes[1].set_title("Control")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("u")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_savefig(f"{prefix}_totals.png" if prefix else None)
    _finish_plot(show)


def plot_age_profiles(
    sim: dict,
    show: bool = True,
    prefix: str | None = None,
) -> None:
    a = sim["a"]
    times = sim["times"]
    x1 = sim["x1"]
    x2 = sim["x2"]

    mid = len(times) // 2

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(a, x1[0], label="x1(a,0)")
    axes[0].plot(a, x1[mid], label="x1(a,T/2)")
    axes[0].plot(a, x1[-1], label="x1(a,T)")
    axes[0].set_title("Species 1 age profiles")
    axes[0].set_ylabel("density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(a, x2[0], label="x2(a,0)")
    axes[1].plot(a, x2[mid], label="x2(a,T/2)")
    axes[1].plot(a, x2[-1], label="x2(a,T)")
    axes[1].set_title("Species 2 age profiles")
    axes[1].set_xlabel("age")
    axes[1].set_ylabel("density")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_savefig(f"{prefix}_profiles.png" if prefix else None)
    _finish_plot(show)


def plot_age_time_heatmaps(
    sim: dict,
    show: bool = True,
    prefix: str | None = None,
) -> None:
    a = sim["a"]
    times = sim["times"]
    x1 = sim["x1"]
    x2 = sim["x2"]

    extent = [a[0], a[-1], times[0], times[-1]]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    im1 = axes[0].imshow(x1, aspect="auto", origin="lower", extent=extent)
    axes[0].set_title("Species 1 density heatmap")
    axes[0].set_ylabel("time")
    fig.colorbar(im1, ax=axes[0], label="x1(a,t)")

    im2 = axes[1].imshow(x2, aspect="auto", origin="lower", extent=extent)
    axes[1].set_title("Species 2 density heatmap")
    axes[1].set_xlabel("age")
    axes[1].set_ylabel("time")
    fig.colorbar(im2, ax=axes[1], label="x2(a,t)")

    plt.tight_layout()
    _maybe_savefig(f"{prefix}_heatmaps.png" if prefix else None)
    _finish_plot(show)


def plot_eta_and_control(
    sim: dict,
    plant,
    eq,
    u_star: float,
    show: bool = True,
    prefix: str | None = None,
) -> None:
    times = sim["times"]
    u = sim["u"]

    eta1, eta2 = compute_eta_series(sim, plant, eq)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(times, eta1, label=r"$\eta_1(t)$")
    axes[0].plot(times, eta2, label=r"$\eta_2(t)$")
    axes[0].axhline(0.0, linestyle="--", linewidth=1.5, label="equilibrium level")
    axes[0].set_title(r"Log-amplitude coordinates $\eta_1,\eta_2$")
    axes[0].set_ylabel(r"$\eta_i$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, u, label=r"$u(t)$")
    axes[1].axhline(u_star, linestyle="--", linewidth=1.8, label=r"target $u^\ast$")
    axes[1].set_title("Control signal")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("dilution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_savefig(f"{prefix}_eta_control.png" if prefix else None)
    _finish_plot(show)


# ============================================================
# k-estimator plots
# ============================================================

def plot_k_estimates_profiles(
    sim: dict,
    show: bool = True,
    prefix: str | None = None,
) -> None:
    if "k1_hat" not in sim or "k2_hat" not in sim:
        return

    a = sim["a"]
    times = sim["times"]
    k1_hat = sim["k1_hat"]
    k2_hat = sim["k2_hat"]
    k1_true = sim["k1_true"]
    k2_true = sim["k2_true"]

    idx0 = 0
    idxm = len(times) // 2
    idxf = len(times) - 1

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(a, k1_true, linewidth=2.5, label="true k1")
    axes[0].plot(a, k1_hat[idx0], label="k1_hat(t=0)")
    axes[0].plot(a, k1_hat[idxm], label="k1_hat(t=T/2)")
    axes[0].plot(a, k1_hat[idxf], label="k1_hat(t=T)")
    axes[0].set_title("Species 1 fertility estimate")
    axes[0].set_ylabel("value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(a, k2_true, linewidth=2.5, label="true k2")
    axes[1].plot(a, k2_hat[idx0], label="k2_hat(t=0)")
    axes[1].plot(a, k2_hat[idxm], label="k2_hat(t=T/2)")
    axes[1].plot(a, k2_hat[idxf], label="k2_hat(t=T)")
    axes[1].set_title("Species 2 fertility estimate")
    axes[1].set_xlabel("age")
    axes[1].set_ylabel("value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_savefig(f"{prefix}_khat_profiles.png" if prefix else None)
    _finish_plot(show)


def plot_k_estimation_errors(
    sim: dict,
    show: bool = True,
    prefix: str | None = None,
) -> None:
    if "k1_hat" not in sim or "k2_hat" not in sim:
        return

    times = sim["times"]
    err1_rel, err2_rel = compute_k_error_series(sim)

    e1 = sim["k_err1_boundary"]
    e2 = sim["k_err2_boundary"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(times, err1_rel, label="relative error k1_hat")
    axes[0].plot(times, err2_rel, label="relative error k2_hat")
    axes[0].set_title("Fertility estimate errors")
    axes[0].set_ylabel("relative L2 error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, e1, label="boundary error species 1")
    axes[1].plot(times, e2, label="boundary error species 2")
    axes[1].axhline(0.0, linestyle="--", linewidth=1.2)
    axes[1].set_title("Boundary prediction errors")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_savefig(f"{prefix}_khat_errors.png" if prefix else None)
    _finish_plot(show)


def plot_estimator_results(
    sim: dict,
    show: bool = True,
    prefix: str | None = None,
) -> None:
    if "k1_hat" not in sim or "k2_hat" not in sim:
        return
    plot_k_estimates_profiles(sim, show=show, prefix=prefix)
    plot_k_estimation_errors(sim, show=show, prefix=prefix)


# ============================================================
# richer mu-estimator plots
# ============================================================

def plot_mu_estimates_profiles(
    sim: dict,
    show: bool = True,
    prefix: str | None = None,
) -> None:
    if "mu1_hat" not in sim or "mu2_hat" not in sim:
        return

    a = sim["a"]
    times = sim["times"]
    mu1_hat = sim["mu1_hat"]
    mu2_hat = sim["mu2_hat"]
    mu1_true = sim["mu1_true"]
    mu2_true = sim["mu2_true"]

    idx0 = 0
    idxm = len(times) // 2
    idxf = len(times) - 1

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(a, mu1_true, linewidth=2.5, label="true mu1")
    axes[0].plot(a, mu1_hat[idx0], label="mu1_hat(t=0)")
    axes[0].plot(a, mu1_hat[idxm], label="mu1_hat(t=T/2)")
    axes[0].plot(a, mu1_hat[idxf], label="mu1_hat(t=T)")
    axes[0].set_title("Species 1 mortality estimate")
    axes[0].set_ylabel("value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(a, mu2_true, linewidth=2.5, label="true mu2")
    axes[1].plot(a, mu2_hat[idx0], label="mu2_hat(t=0)")
    axes[1].plot(a, mu2_hat[idxm], label="mu2_hat(t=T/2)")
    axes[1].plot(a, mu2_hat[idxf], label="mu2_hat(t=T)")
    axes[1].set_title("Species 2 mortality estimate")
    axes[1].set_xlabel("age")
    axes[1].set_ylabel("value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_savefig(f"{prefix}_muhat_profiles.png" if prefix else None)
    _finish_plot(show)


def plot_mu_regression_norms(
    sim: dict,
    show: bool = True,
    prefix: str | None = None,
) -> None:
    if "regression_error1_norm" not in sim:
        return

    times = sim["times"]
    r1n = sim["regression_error1_norm"]
    r2n = sim["regression_error2_norm"]
    Y1 = sim["Y1"]
    Y2 = sim["Y2"]
    sigma1_norm, sigma2_norm = compute_omega_norm_series(sim)
    rho1_mean = np.mean(sim["rho1"], axis=1)
    rho2_mean = np.mean(sim["rho2"], axis=1)
    Y1_mean = np.mean(Y1, axis=1)
    Y2_mean = np.mean(Y2, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    axes[0].plot(times, r1n, label="species 1 regression norm")
    axes[0].plot(times, r2n, label="species 2 regression norm")
    axes[0].axhline(0.0, linestyle="--", linewidth=1.2)
    axes[0].set_title("Localized regression-error norms")
    axes[0].set_ylabel("norm")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, rho1_mean, label=r"mean $\rho_1$")
    axes[1].plot(times, rho2_mean, label=r"mean $\rho_2$")
    axes[1].plot(times, Y1_mean, label=r"mean $Y_1$", linestyle="--")
    axes[1].plot(times, Y2_mean, label=r"mean $Y_2$", linestyle="--")
    axes[1].axhline(0.0, linestyle="--", linewidth=1.2)
    axes[1].set_title("Mean pointwise filter signals")
    axes[1].set_ylabel("value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, sigma1_norm, label=r"$\|\sigma_1\|$")
    axes[2].plot(times, sigma2_norm, label=r"$\|\sigma_2\|$")
    axes[2].set_title("Swapping-filter state norms")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("norm")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_savefig(f"{prefix}_muhat_errors.png" if prefix else None)
    _finish_plot(show)


def plot_mu_regression_heatmaps(
    sim: dict,
    show: bool = True,
    prefix: str | None = None,
) -> None:
    if "mu_regression_error1" not in sim:
        return

    times = sim["times"]
    reg1 = sim["mu_regression_error1"]
    reg2 = sim["mu_regression_error2"]

    extent = [0.0, 1.0, times[0], times[-1]]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    im1 = axes[0].imshow(reg1, aspect="auto", origin="lower", extent=extent)
    axes[0].set_title("Species 1 localized regression errors")
    axes[0].set_ylabel("time")
    fig.colorbar(im1, ax=axes[0], label="error")

    im2 = axes[1].imshow(reg2, aspect="auto", origin="lower", extent=extent)
    axes[1].set_title("Species 2 localized regression errors")
    axes[1].set_xlabel("window index")
    axes[1].set_ylabel("time")
    fig.colorbar(im2, ax=axes[1], label="error")

    plt.tight_layout()
    _maybe_savefig(f"{prefix}_muhat_regression_heatmaps.png" if prefix else None)
    _finish_plot(show)


def plot_mu_regression_surfaces(
    sim: dict,
    show: bool = True,
    prefix: str | None = None,
) -> None:
    if "mu_regression_error1" not in sim:
        return

    a = sim["a"]
    times = sim["times"]
    reg1 = sim["mu_regression_error1"]
    reg2 = sim["mu_regression_error2"]

    aa, tt = np.meshgrid(a, times, indexing="xy")

    fig = plt.figure(figsize=(12, 5.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    surf1 = ax1.plot_surface(
        aa, tt, reg1,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rasterized=True,
    )
    surf2 = ax2.plot_surface(
        aa, tt, reg2,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rasterized=True,
    )

    ax1.set_title("Species 1 mu regression error")
    ax2.set_title("Species 2 mu regression error")
    for ax in (ax1, ax2):
        ax.set_xlabel("age")
        ax.set_ylabel("time")
        ax.set_zlabel("error")
        ax.view_init(elev=24, azim=-130)

    fig.colorbar(surf1, ax=ax1, shrink=0.72, pad=0.08)
    fig.colorbar(surf2, ax=ax2, shrink=0.72, pad=0.08)
    fig.tight_layout()

    _maybe_savefig(f"{prefix}_muhat_regression_surfaces.png" if prefix else None)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_mu_estimator_results(
    sim: dict,
    show: bool = True,
    prefix: str | None = None,
) -> None:
    if "mu1_hat" not in sim or "mu2_hat" not in sim:
        return
    plot_mu_estimates_profiles(sim, show=show, prefix=prefix)
    plot_mu_regression_norms(sim, show=show, prefix=prefix)
    plot_mu_regression_heatmaps(sim, show=show, prefix=prefix)
    plot_mu_regression_surfaces(sim, show=show, prefix=prefix)


# ============================================================
# target-equilibrium diagnostics
# ============================================================

def plot_target_vs_achieved_equilibrium(
    sim: dict,
    plant,
    eq,
    show: bool = True,
    prefix: str | None = None,
    ratio_eps: float = 1e-10,
) -> None:
    a = sim["a"]
    x1_final = sim["x1"][-1]
    x2_final = sim["x2"][-1]

    x1_star = eq.x1_star_age0 * plant.n1
    x2_star = eq.x2_star_age0 * plant.n2

    ratio1 = x1_final / np.maximum(x1_star, ratio_eps)
    ratio2 = x2_final / np.maximum(x2_star, ratio_eps)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex="col")

    axes[0, 0].plot(a, x1_star, label=r"target $x_1^\ast(a)$", linewidth=2.5)
    axes[0, 0].plot(a, x1_final, label=r"achieved $x_1(a,T)$", linewidth=2.0)
    axes[0, 0].set_title("Species 1: target vs achieved")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(a, ratio1, label=r"$x_1(a,T)/x_1^\ast(a)$")
    axes[1, 0].axhline(1.0, linestyle="--", linewidth=1.5, label="perfect match")
    axes[1, 0].set_xlabel("age")
    axes[1, 0].set_ylabel("ratio")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(a, x2_star, label=r"target $x_2^\ast(a)$", linewidth=2.5)
    axes[0, 1].plot(a, x2_final, label=r"achieved $x_2(a,T)$", linewidth=2.0)
    axes[0, 1].set_title("Species 2: target vs achieved")
    axes[0, 1].set_ylabel("density")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(a, ratio2, label=r"$x_2(a,T)/x_2^\ast(a)$")
    axes[1, 1].axhline(1.0, linestyle="--", linewidth=1.5, label="perfect match")
    axes[1, 1].set_xlabel("age")
    axes[1, 1].set_ylabel("ratio")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_savefig(f"{prefix}_target_vs_achieved.png" if prefix else None)
    _finish_plot(show)


def plot_equilibrium_tracking_errors(
    sim: dict,
    plant,
    eq,
    show: bool = True,
    prefix: str | None = None,
) -> None:
    a = sim["a"]
    times = sim["times"]
    x1_hist = sim["x1"]
    x2_hist = sim["x2"]

    x1_star = eq.x1_star_age0 * plant.n1
    x2_star = eq.x2_star_age0 * plant.n2

    denom1 = max(float(np.sqrt(np.trapezoid(x1_star * x1_star, a))), 1e-12)
    denom2 = max(float(np.sqrt(np.trapezoid(x2_star * x2_star, a))), 1e-12)

    err1 = np.array([
        float(np.sqrt(np.trapezoid((x - x1_star) ** 2, a))) / denom1
        for x in x1_hist
    ])
    err2 = np.array([
        float(np.sqrt(np.trapezoid((x - x2_star) ** 2, a))) / denom2
        for x in x2_hist
    ])

    plt.figure(figsize=(9, 5))
    plt.plot(times, err1, label=r"relative equilibrium error species 1")
    plt.plot(times, err2, label=r"relative equilibrium error species 2")
    plt.xlabel("time")
    plt.ylabel("relative error")
    plt.title("Distance to target equilibrium")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _maybe_savefig(f"{prefix}_equilibrium_errors.png" if prefix else None)
    _finish_plot(show)


# ============================================================
# Master plotting entry point
# ============================================================

def plot_simulation_results(
    plant,
    sim: dict,
    eq=None,
    u_star: float | None = None,
    show: bool = True,
    prefix: str | None = None,
    include_heatmaps: bool = True,
    include_eta_control: bool = True,
    include_estimator: bool = True,
) -> None:
    plot_plant_coefficients(plant, show=show, prefix=prefix)
    plot_population_totals(sim, show=show, prefix=prefix)
    plot_age_profiles(sim, show=show, prefix=prefix)

    if include_heatmaps:
        plot_age_time_heatmaps(sim, show=show, prefix=prefix)

    if include_eta_control:
        if eq is None or u_star is None:
            raise ValueError("Need eq and u_star to plot eta/control figure.")
        plot_eta_and_control(sim, plant, eq, u_star, show=show, prefix=prefix)

    if include_estimator:
        plot_estimator_results(sim, show=show, prefix=prefix)
        plot_mu_estimator_results(sim, show=show, prefix=prefix)

    if eq is not None:
        plot_target_vs_achieved_equilibrium(
            sim, plant, eq, show=show, prefix=prefix
        )
        plot_equilibrium_tracking_errors(
            sim, plant, eq, show=show, prefix=prefix
        )
