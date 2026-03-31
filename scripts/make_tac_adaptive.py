from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
from operators import G_LS, G_gamma, G_kappa, G_pi, compute_all_species_operators
from controller import NominalController
from estimate_k import KEstimator, KEstimatorConfig
from estimate_mu import MuEstimator, MuEstimatorConfig


mpl.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.7,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    }
)


# ============================================================
# User settings
# ============================================================

SEED = 123

N_AGE = 201
T_FINAL = 15
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
GAMMA_K_1 = 0.7
GAMMA_K_2 = 0.7

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
# FNO zeta settings
# ============================================================

USE_FNO_ZETA =True

# If True and zeta is coming from the FNO, recompute zeta1/zeta2 during the
# simulation from the current k/mu signals selected below. If False, compute
# the FNO zetas once at the beginning and keep them fixed.
ADAPT_FNO_ZETA_FROM_ESTIMATES = True

# Which signals to feed into the FNO zeta model.
USE_ESTIMATED_K_IN_FNO_ZETA = True
USE_ESTIMATED_MU_IN_FNO_ZETA = True

FNO_ZETA_CHECKPOINT = ROOT / "models" / "fno_zeta_adaptive" / "adaptive_run_001" / "best_model.pt"

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

USE_CSV_ESTIMATOR_GUESSES = True
CSV_K1_HAT_SAMPLE_INDEX = 1
CSV_MU1_HAT_SAMPLE_INDEX = 1
CSV_K2_HAT_SAMPLE_INDEX = 2
CSV_MU2_HAT_SAMPLE_INDEX = 2

PLOT_RESULTS = True
PLOT_PREFIX = ROOT / "figures" / "tac_adaptive_figure"
SIM_PROGRESS_EVERY_STEPS = 500


def set_size(width, fraction=1, subplots=(1, 1), height_add=0):
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = height_add + fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def _resolve_sample_row_index(index: int, n_available: int, label: str) -> int:
    if n_available <= 0:
        raise ValueError(f"No samples available for {label}.")

    if CSV_SAMPLE_INDEX_MODE == "zero_based":
        row_idx = index
        valid_range = f"0..{n_available - 1}"
    elif CSV_SAMPLE_INDEX_MODE == "one_based":
        row_idx = index - 1
        valid_range = f"1..{n_available}"
    else:
        raise ValueError(
            f"Unsupported CSV_SAMPLE_INDEX_MODE={CSV_SAMPLE_INDEX_MODE!r}. "
            "Use 'zero_based' or 'one_based'."
        )

    if not 0 <= row_idx < n_available:
        raise IndexError(
            f"{label} sample index {index} is out of range for mode "
            f"{CSV_SAMPLE_INDEX_MODE!r}; expected {valid_range}."
        )
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
        "k1_row": float(k1_row),
        "mu1_index": float(CSV_MU1_SAMPLE_INDEX),
        "mu1_row": float(mu1_row),
        "k2_index": float(CSV_K2_SAMPLE_INDEX),
        "k2_row": float(k2_row),
        "mu2_index": float(CSV_MU2_SAMPLE_INDEX),
        "mu2_row": float(mu2_row),
    }
    return updated, info


# ============================================================
# FNO zeta helpers
# ============================================================

class ZetaFNOPredictor:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self._model = None
        self._stats = None
        self._device = None
        self._predict_zeta_fno = None

    def _load(self) -> None:
        if self._model is not None:
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"FNO zeta checkpoint not found: {self.checkpoint_path}"
            )

        try:
            import torch
            from train_fno_adaptive import load_fno_checkpoint, predict_zeta_fno
        except ImportError as exc:
            raise ImportError(
                "USE_FNO_ZETA=True requires the FNO inference dependencies "
                "(e.g. torch and neuralop) to be installed."
            ) from exc

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model, stats, _ = load_fno_checkpoint(self.checkpoint_path, device=device)

        self._model = model
        self._stats = stats
        self._device = device
        self._predict_zeta_fno = predict_zeta_fno

    def predict(self, mu: np.ndarray, k: np.ndarray) -> float:
        self._load()
        zeta = float(
            self._predict_zeta_fno(
                self._model,
                self._stats,
                mu_vals=mu,
                k_vals=k,
                device=self._device,
            )
        )
        if not np.isfinite(zeta):
            raise ValueError("FNO predicted a non-finite zeta value.")
        return zeta


def compute_all_species_operators_with_optional_zeta(
    k1: np.ndarray,
    mu1: np.ndarray,
    g1: np.ndarray,
    k2: np.ndarray,
    mu2: np.ndarray,
    g2: np.ndarray,
    a: np.ndarray,
    zeta1: float | None = None,
    zeta2: float | None = None,
):
    if zeta1 is None and zeta2 is None:
        return compute_all_species_operators(
            k1=k1,
            mu1=mu1,
            g1=g1,
            k2=k2,
            mu2=mu2,
            g2=g2,
            a=a,
        )

    if zeta1 is None or zeta2 is None:
        raise ValueError("Either provide both zeta1 and zeta2 or neither.")
    if not np.isfinite(zeta1) or not np.isfinite(zeta2):
        raise ValueError("Non-finite zeta supplied for operator computation.")

    kappa1 = G_kappa(k1, mu1, zeta1, a)
    kappa2 = G_kappa(k2, mu2, zeta2, a)
    gamma1 = G_gamma(g1, zeta2, mu2, a)
    gamma2 = G_gamma(g2, zeta1, mu1, a)
    pi0_1 = G_pi(k1, mu1, zeta1, a)
    pi0_2 = G_pi(k2, mu2, zeta2, a)

    return {
        "zeta1": float(zeta1),
        "zeta2": float(zeta2),
        "kappa1": kappa1,
        "kappa2": kappa2,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "pi0_1": pi0_1,
        "pi0_2": pi0_2,
    }


def initial_fno_inputs(plant: Plant, envelopes):
    k1 = envelopes.k_lower.copy() if USE_ESTIMATED_K_IN_FNO_ZETA else plant.k1
    k2 = envelopes.k_lower.copy() if USE_ESTIMATED_K_IN_FNO_ZETA else plant.k2
    mu1 = envelopes.mu_lower.copy() if USE_ESTIMATED_MU_IN_FNO_ZETA else plant.mu1
    mu2 = envelopes.mu_lower.copy() if USE_ESTIMATED_MU_IN_FNO_ZETA else plant.mu2
    return k1, mu1, k2, mu2


def estimated_fno_inputs(
    plant: Plant,
    k_estimator: KEstimator,
    mu_estimator: MuEstimator,
):
    k1 = k_estimator.k1_hat if USE_ESTIMATED_K_IN_FNO_ZETA else plant.k1
    k2 = k_estimator.k2_hat if USE_ESTIMATED_K_IN_FNO_ZETA else plant.k2
    mu1 = mu_estimator.mu1_hat if USE_ESTIMATED_MU_IN_FNO_ZETA else plant.mu1
    mu2 = mu_estimator.mu2_hat if USE_ESTIMATED_MU_IN_FNO_ZETA else plant.mu2
    return k1, mu1, k2, mu2


def predict_zeta_pair(
    predictor: ZetaFNOPredictor,
    k1: np.ndarray,
    mu1: np.ndarray,
    k2: np.ndarray,
    mu2: np.ndarray,
) -> tuple[float, float]:
    zeta1 = predictor.predict(mu1, k1)
    zeta2 = predictor.predict(mu2, k2)
    return zeta1, zeta2


def plant_with_overridden_zetas(plant: Plant, zeta1: float, zeta2: float) -> Plant:
    gamma1 = G_gamma(plant.g1, zeta2, plant.mu2, plant.a)
    gamma2 = G_gamma(plant.g2, zeta1, plant.mu1, plant.a)
    n1 = stable_age_profile(plant.mu1, zeta1, plant.a)
    n2 = stable_age_profile(plant.mu2, zeta2, plant.a)
    return replace(
        plant,
        zeta1=float(zeta1),
        zeta2=float(zeta2),
        gamma1=gamma1,
        gamma2=gamma2,
        n1=n1,
        n2=n2,
    )


def _resample_to_age_grid(values: np.ndarray, source_grid: np.ndarray, target_grid: np.ndarray) -> np.ndarray:
    return np.interp(target_grid, source_grid, values).astype(float)


def initial_estimator_guesses(plant: Plant, envelopes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if USE_CSV_ESTIMATOR_GUESSES:
        k1_guess, _ = _load_csv_sample_profiles(
            CSV_K_SAMPLES_PATH,
            CSV_K1_HAT_SAMPLE_INDEX,
            plant.a,
            "k1_hat",
        )
        mu1_guess, _ = _load_csv_sample_profiles(
            CSV_MU_SAMPLES_PATH,
            CSV_MU1_HAT_SAMPLE_INDEX,
            plant.a,
            "mu1_hat",
        )
        k2_guess, _ = _load_csv_sample_profiles(
            CSV_K_SAMPLES_PATH,
            CSV_K2_HAT_SAMPLE_INDEX,
            plant.a,
            "k2_hat",
        )
        mu2_guess, _ = _load_csv_sample_profiles(
            CSV_MU_SAMPLES_PATH,
            CSV_MU2_HAT_SAMPLE_INDEX,
            plant.a,
            "mu2_hat",
        )
        return k1_guess, mu1_guess, k2_guess, mu2_guess

    return (
        envelopes.k_lower.copy(),
        envelopes.mu_lower.copy(),
        envelopes.k_lower.copy(),
        envelopes.mu_lower.copy(),
    )


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
    zeta1_override: float | None = None,
    zeta2_override: float | None = None,
):
    ops = compute_all_species_operators_with_optional_zeta(
        k1=k1_used,
        mu1=mu1_used,
        g1=plant.g1,
        k2=k2_used,
        mu2=mu2_used,
        g2=plant.g2,
        a=plant.a,
        zeta1=zeta1_override,
        zeta2=zeta2_override,
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

def _style_3d_axis(ax) -> None:
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["axisline"]["linewidth"] = 1.0
        axis._axinfo["axisline"]["color"] = "b"
        axis._axinfo["grid"]["linewidth"] = 0.2
        axis._axinfo["grid"]["linestyle"] = "--"
        axis._axinfo["grid"]["color"] = "#d1d1d1"
        axis.set_pane_color((1, 1, 1, 1))


def _plot_surface_with_final_slice(
    ax,
    a: np.ndarray,
    times: np.ndarray,
    surface: np.ndarray,
    final_target: np.ndarray | None = None,
    final_actual: np.ndarray | None = None,
    zlabel: str = "",
) -> None:
    aa, tt = np.meshgrid(a, times, indexing="xy")
    ax.plot_surface(
        aa,
        tt,
        surface,
        edgecolor="black",
        lw=0.2,
        rstride=max(1, len(times) // 50),
        cstride=max(1, len(a) // 50),
        alpha=1.0,
        color="white",
        shade=False,
        antialiased=True,
        rasterized=True,
    )
    if final_target is not None:
        ax.plot(a, np.full_like(a, times[-1]), final_target, color="blue", lw=1.3, antialiased=True)
    if final_actual is not None:
        ax.plot(
            a,
            np.full_like(a, times[-1]),
            final_actual,
            color="red",
            lw=1.3,
            linestyle="--",
            antialiased=True,
        )
    ax.set_xlabel(r"Age $a$", labelpad=-2)
    ax.set_ylabel(r"Time $t$", labelpad=-2)
    ax.set_zlabel(zlabel)
    ax.view_init(10, 15)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="x", pad=-2)
    ax.tick_params(axis="y", pad=-2)
    ax.tick_params(axis="z", pad=-3)


def compute_exact_estimator_zeta_series(sim: dict) -> tuple[np.ndarray, np.ndarray]:
    a = sim["a"]
    zeta1_exact = np.full(len(sim["times"]), np.nan)
    zeta2_exact = np.full(len(sim["times"]), np.nan)

    for idx in range(len(sim["times"])):
        zeta1_exact[idx] = G_LS(sim["k1_hat"][idx], sim["mu1_hat"][idx], a)
        zeta2_exact[idx] = G_LS(sim["k2_hat"][idx], sim["mu2_hat"][idx], a)

    return zeta1_exact, zeta2_exact


def plot_tac_adaptive_figure(
    plant: Plant,
    eq,
    sim_adaptive: dict,
    sim_frozen: dict,
    u_star: float,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    times = sim_adaptive["times"]
    a = sim_adaptive["a"]
    x1 = sim_adaptive["x1"]
    x2 = sim_adaptive["x2"]
    u_adaptive = sim_adaptive["u"]
    u_frozen = sim_frozen["u"]
    x1_target = eq.x1_star_age0 * plant.n1
    x2_target = eq.x2_star_age0 * plant.n2

    zeta1_exact, zeta2_exact = compute_exact_estimator_zeta_series(sim_adaptive)
    zeta1_fno = sim_adaptive["zeta1_used"]
    zeta2_fno = sim_adaptive["zeta2_used"]
    zeta1_error = np.abs(zeta1_fno - zeta1_exact)
    zeta2_error = np.abs(zeta2_fno - zeta2_exact)

    save_root = Path(save_path) if save_path else None
    if save_root is not None:
        save_root.parent.mkdir(parents=True, exist_ok=True)

    def _shrink_square_axis(ax, scale: float = 0.72) -> None:
        pos = ax.get_position()
        new_width = pos.width * scale
        new_height = pos.height * scale
        new_x0 = pos.x0 + 0.5 * (pos.width - new_width)
        new_y0 = pos.y0 + 0.5 * (pos.height - new_height)
        ax.set_position([new_x0, new_y0, new_width, new_height])

    def _draw_top_row(ax1, ax2, ax3) -> None:
        for ax in (ax1, ax2):
            _style_3d_axis(ax)

        _plot_surface_with_final_slice(
            ax1,
            a,
            times,
            x1,
            final_target=x1_target,
            final_actual=x1[-1],
            zlabel="",
        )
        _plot_surface_with_final_slice(
            ax2,
            a,
            times,
            x2,
            final_target=x2_target,
            final_actual=x2[-1],
            zlabel="",
        )
        ax3.plot(times, u_adaptive, color="black", label=r"adaptive $u(t)$")
        ax3.plot(times, u_frozen, "--", color="tab:blue", label=r"no adaptation $u(t)$")
        ax3.axhline(u_star, linestyle="--", color="gray", linewidth=1.1, label=r"$u^\ast$")
        ax3.set_xlabel(r"Time $t$")
        ax3.set_ylabel(r"$u(t)$")
        ax3.grid(True, alpha=0.22, linewidth=0.4, linestyle="--", color="#d1d1d1")
        ax3.legend(
            frameon=False,
            loc="lower right",
            bbox_to_anchor=(1.05, 0.1),
            fontsize=8,
            labelspacing=0.2,
            handlelength=1.4,
            handletextpad=0.35,
            borderaxespad=0.15,
        )
        ax3.set_box_aspect(1)
        _shrink_square_axis(ax3)

    def _draw_bottom_row(ax4, ax5, ax6) -> None:
        for ax in (ax4, ax5):
            _style_3d_axis(ax)

        _plot_surface_with_final_slice(
            ax4,
            a,
            times,
            sim_adaptive["k1_hat"],
            final_target=plant.k1,
            final_actual=sim_adaptive["k1_hat"][-1],
            zlabel=r"$\hat{k}_1$",
        )
        _plot_surface_with_final_slice(
            ax5,
            a,
            times,
            sim_adaptive["mu1_hat"],
            final_target=plant.mu1,
            final_actual=sim_adaptive["mu1_hat"][-1],
            zlabel=r"$\hat{\mu}_1$",
        )
        ax6.plot(times, zeta1_error, color="tab:red", label=r"$|\hat{\zeta}_1^{\mathrm{FNO}}-\hat{\zeta}_1^{\mathrm{num}}|$")
        ax6.plot(times, zeta2_error, color="tab:blue", label=r"$|\hat{\zeta}_2^{\mathrm{FNO}}-\hat{\zeta}_2^{\mathrm{num}}|$")
        ax6.set_xlabel(r"Time $t$")
        ax6.set_ylabel(r"$\zeta$ error")
        ax6.set_yscale("log")
        ax6.grid(True, alpha=0.22, linewidth=0.4, linestyle="--", color="#d1d1d1")
        ax6.legend(frameon=False, loc="best")
        ax6.set_box_aspect(1)
        _shrink_square_axis(ax6)

    def _label_row(fig, axes, labels, y_offset: float = 0.08) -> None:
        positions = [ax.get_position() for ax in axes]
        title_y = max(pos.y1 for pos in positions) - y_offset
        for pos, label in zip(positions, labels, strict=False):
            fig.text(pos.x0, title_y, label, ha="left", va="bottom")

    top_fig = plt.figure(figsize=set_size(522, 1.0, (1, 3), height_add=1.1))
    top_gs = top_fig.add_gridspec(
        1,
        3,
        left=0.04,
        bottom=0.12,
        right=0.985,
        top=0.92,
        wspace=0.26,
    )
    top_axes = (
        top_fig.add_subplot(top_gs[0, 0], projection="3d", computed_zorder=False),
        top_fig.add_subplot(top_gs[0, 1], projection="3d", computed_zorder=False),
        top_fig.add_subplot(top_gs[0, 2]),
    )
    _draw_top_row(*top_axes)
    _label_row(
        top_fig,
        top_axes,
        (
            r"(a) State $x_1(a, t)$",
            r"(b) State $x_2(a, t)$",
            r"(c) Dilution $u(t)$",
        ),
    )

    bottom_fig = plt.figure(figsize=set_size(522, 1.0, (1, 3), height_add=1.1))
    bottom_gs = bottom_fig.add_gridspec(
        1,
        3,
        left=0.04,
        bottom=0.12,
        right=0.985,
        top=0.92,
        wspace=0.26,
    )
    bottom_axes = (
        bottom_fig.add_subplot(bottom_gs[0, 0], projection="3d", computed_zorder=False),
        bottom_fig.add_subplot(bottom_gs[0, 1], projection="3d", computed_zorder=False),
        bottom_fig.add_subplot(bottom_gs[0, 2]),
    )
    _draw_bottom_row(*bottom_axes)
    _label_row(
        bottom_fig,
        bottom_axes,
        (
            r"(d) Estimate $\hat{k}_1(a, t)$",
            r"(e) Estimate $\hat{\mu}_1(a, t)$",
            r"(f) FNO $\zeta$ error",
        ),
    )

    combined_fig = plt.figure(figsize=set_size(522, 1.0, (2, 3), height_add=1.5))
    combined_gs = combined_fig.add_gridspec(
        2,
        3,
        left=0.04,
        bottom=0.08,
        right=0.985,
        top=0.96,
        hspace=0.26,
        wspace=0.26,
    )
    combined_top_axes = (
        combined_fig.add_subplot(combined_gs[0, 0], projection="3d", computed_zorder=False),
        combined_fig.add_subplot(combined_gs[0, 1], projection="3d", computed_zorder=False),
        combined_fig.add_subplot(combined_gs[0, 2]),
    )
    combined_bottom_axes = (
        combined_fig.add_subplot(combined_gs[1, 0], projection="3d", computed_zorder=False),
        combined_fig.add_subplot(combined_gs[1, 1], projection="3d", computed_zorder=False),
        combined_fig.add_subplot(combined_gs[1, 2]),
    )
    _draw_top_row(*combined_top_axes)
    _draw_bottom_row(*combined_bottom_axes)
    _label_row(
        combined_fig,
        combined_top_axes,
        (
            r"(a) State $x_1(a, t)$",
            r"(b) State $x_2(a, t)$",
            r"(c) Dilution $u(t)$",
        ),
        y_offset=0.065,
    )
    _label_row(
        combined_fig,
        combined_bottom_axes,
        (
            r"(d) Estimate $\hat{k}_1(a, t)$",
            r"(e) Estimate $\hat{\mu}_1(a, t)$",
            r"(f) FNO $\zeta$ error",
        ),
        y_offset=0.065,
    )

    if save_root is not None:
        top_path = save_root.with_name(f"{save_root.stem}_top{save_root.suffix}")
        bottom_path = save_root.with_name(f"{save_root.stem}_bottom{save_root.suffix}")
        combined_path = save_root
        top_fig.savefig(top_path, bbox_inches="tight")
        bottom_fig.savefig(bottom_path, bbox_inches="tight")
        combined_fig.savefig(combined_path, bbox_inches="tight")
        print(f"Saved figure to: {top_path}")
        print(f"Saved figure to: {bottom_path}")
        print(f"Saved figure to: {combined_path}")

    if show:
        plt.show()
    else:
        plt.close(top_fig)
        plt.close(bottom_fig)
        plt.close(combined_fig)


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
    k1_init_guess: np.ndarray | None = None,
    mu1_init_guess: np.ndarray | None = None,
    k2_init_guess: np.ndarray | None = None,
    mu2_init_guess: np.ndarray | None = None,
    freeze_controller_estimates: bool = False,
):
    a = plant.a
    da = a[1] - a[0]
    n_age = len(a)

    if abs(dt - da) > 1e-12:
        raise ValueError("This simulator assumes dt = da.")

    if k1_init_guess is None:
        k1_init_guess = envelopes.k_lower
    if k2_init_guess is None:
        k2_init_guess = envelopes.k_lower
    if mu1_init_guess is None:
        mu1_init_guess = envelopes.mu_lower
    if mu2_init_guess is None:
        mu2_init_guess = envelopes.mu_lower

    fixed_k1 = np.asarray(k1_init_guess, dtype=float).copy()
    fixed_k2 = np.asarray(k2_init_guess, dtype=float).copy()
    fixed_mu1 = np.asarray(mu1_init_guess, dtype=float).copy()
    fixed_mu2 = np.asarray(mu2_init_guess, dtype=float).copy()

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
        k1_init=fixed_k1,
        k2_init=fixed_k2,
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
        mu1_init=fixed_mu1,
        mu2_init=fixed_mu2,
        x1_init=x1_0,
        x2_init=x2_0,
        config=mu_est_cfg,
    )

    zeta_predictor = ZetaFNOPredictor(FNO_ZETA_CHECKPOINT) if USE_FNO_ZETA else None
    adaptive_fno_zeta_active = (
        USE_FNO_ZETA
        and ADAPT_FNO_ZETA_FROM_ESTIMATES
        and (ADAPT_K or ADAPT_MU)
    )
    static_fno_zetas = None

    def controller_inputs():
        if freeze_controller_estimates:
            return fixed_k1, fixed_mu1, fixed_k2, fixed_mu2
        k1_used = k_estimator.k1_hat if USE_ESTIMATED_K_IN_CONTROLLER else plant.k1
        k2_used = k_estimator.k2_hat if USE_ESTIMATED_K_IN_CONTROLLER else plant.k2
        mu1_used = mu_estimator.mu1_hat if USE_ESTIMATED_MU_IN_CONTROLLER else plant.mu1
        mu2_used = mu_estimator.mu2_hat if USE_ESTIMATED_MU_IN_CONTROLLER else plant.mu2
        return k1_used, mu1_used, k2_used, mu2_used

    def current_fno_zetas():
        nonlocal static_fno_zetas

        if not USE_FNO_ZETA:
            return None, None

        if freeze_controller_estimates:
            return predict_zeta_pair(zeta_predictor, fixed_k1, fixed_mu1, fixed_k2, fixed_mu2)

        if adaptive_fno_zeta_active:
            k1_z, mu1_z, k2_z, mu2_z = estimated_fno_inputs(
                plant,
                k_estimator,
                mu_estimator,
            )
            return predict_zeta_pair(zeta_predictor, k1_z, mu1_z, k2_z, mu2_z)

        if static_fno_zetas is None:
            k1_z, mu1_z, k2_z, mu2_z = initial_fno_inputs(plant, envelopes)
            static_fno_zetas = predict_zeta_pair(zeta_predictor, k1_z, mu1_z, k2_z, mu2_z)
        return static_fno_zetas

    if control_mode == "nominal":
        k1_used, mu1_used, k2_used, mu2_used = controller_inputs()
        zeta1_used, zeta2_used = current_fno_zetas()
        controller, ops = build_controller_from_estimates(
            plant,
            eq,
            k1_used,
            mu1_used,
            k2_used,
            mu2_used,
            zeta1_override=zeta1_used,
            zeta2_override=zeta2_used,
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
    r1_hist = np.zeros((n_steps + 1, n_age))
    r2_hist = np.zeros((n_steps + 1, n_age))
    Q1_hist = np.zeros(n_steps + 1)
    Q2_hist = np.zeros(n_steps + 1)
    mu_regression_error1_hist = np.zeros((n_steps + 1, n_age))
    mu_regression_error2_hist = np.zeros((n_steps + 1, n_age))
    mu_regression_error1_norm_hist = np.zeros(n_steps + 1)
    mu_regression_error2_norm_hist = np.zeros(n_steps + 1)
    zeta1_used_hist = np.full(n_steps + 1, np.nan)
    zeta2_used_hist = np.full(n_steps + 1, np.nan)

    if control_mode == "nominal":
        ip_pi0_1_x1_hist = np.zeros(n_steps + 1)
        ip_pi0_2_x2_hist = np.zeros(n_steps + 1)
        zeta1_used_hist[0] = ops["zeta1"]
        zeta2_used_hist[0] = ops["zeta2"]
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
    mu_regression_error1_hist[0] = mu_estimator.regression_error1.copy()
    mu_regression_error2_hist[0] = mu_estimator.regression_error2.copy()

    for n in range(n_steps):
        x1 = x1_hist[n].copy()
        x2 = x2_hist[n].copy()

        if SIM_PROGRESS_EVERY_STEPS > 0 and (n % SIM_PROGRESS_EVERY_STEPS == 0 or n == n_steps - 1):
            print(
                f"[simulate{' frozen' if freeze_controller_estimates else ''}] "
                f"step {n + 1}/{n_steps} "
                f"(t={times[n]:.3f})"
            )

        if control_mode == "nominal":
            k1_used, mu1_used, k2_used, mu2_used = controller_inputs()
            zeta1_used, zeta2_used = current_fno_zetas()
            controller, ops = build_controller_from_estimates(
                plant,
                eq,
                k1_used,
                mu1_used,
                k2_used,
                mu2_used,
                zeta1_override=zeta1_used,
                zeta2_override=zeta2_used,
            )
            zeta1_used_hist[n] = ops["zeta1"]
            zeta2_used_hist[n] = ops["zeta2"]

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

        # ---- update pointwise mu-estimator
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
        r1_hist[n + 1] = mu_info["r1"]
        r2_hist[n + 1] = mu_info["r2"]
        Q1_hist[n + 1] = mu_info["Q1"]
        Q2_hist[n + 1] = mu_info["Q2"]
        mu_regression_error1_hist[n] = mu_info["regression_error1"]
        mu_regression_error2_hist[n] = mu_info["regression_error2"]
        mu_regression_error1_norm_hist[n] = mu_info["regression_error1_norm"]
        mu_regression_error2_norm_hist[n] = mu_info["regression_error2_norm"]

    # final diagnostics
    x1_final = x1_hist[-1]
    x2_final = x2_hist[-1]

    if control_mode == "nominal":
        k1_used, mu1_used, k2_used, mu2_used = controller_inputs()
        zeta1_used, zeta2_used = current_fno_zetas()
        controller, ops = build_controller_from_estimates(
            plant,
            eq,
            k1_used,
            mu1_used,
            k2_used,
            mu2_used,
            zeta1_override=zeta1_used,
            zeta2_override=zeta2_used,
        )
        zeta1_used_hist[-1] = ops["zeta1"]
        zeta2_used_hist[-1] = ops["zeta2"]

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
    sigma1_hist[-1] = mu_final["sigma1"]
    sigma2_hist[-1] = mu_final["sigma2"]
    rho1_hist[-1] = mu_final["rho1"]
    rho2_hist[-1] = mu_final["rho2"]
    Y1_hist[-1] = mu_final["Y1"]
    Y2_hist[-1] = mu_final["Y2"]
    r1_hist[-1] = mu_final["r1"]
    r2_hist[-1] = mu_final["r2"]
    Q1_hist[-1] = mu_final["Q1"]
    Q2_hist[-1] = mu_final["Q2"]

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

        # pointwise mu estimator
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
        "r1": r1_hist,
        "r2": r2_hist,
        "Q1": Q1_hist,
        "Q2": Q2_hist,
        "mu_regression_error1": mu_regression_error1_hist,
        "mu_regression_error2": mu_regression_error2_hist,
        "mu_regression_error1_norm": mu_regression_error1_norm_hist,
        "mu_regression_error2_norm": mu_regression_error2_norm_hist,
        "zeta1_used": zeta1_used_hist,
        "zeta2_used": zeta2_used_hist,

        "adapt_k": ADAPT_K,
        "adapt_mu": ADAPT_MU,
        "use_fno_zeta": USE_FNO_ZETA,
        "adapt_fno_zeta_from_estimates": adaptive_fno_zeta_active,
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
    true_zeta1 = float(plant.zeta1)
    true_zeta2 = float(plant.zeta2)
    u_star = 0.5 * min(true_zeta1, true_zeta2)

    eq = validate_equilibrium_values(plant, u_star)

    print("\n" + "="*60)
    print("Plant coefficients:")
    print(f"  k1  = {plant.k1}")
    print(f"  k2  = {plant.k2}")
    print(f"  mu1 = {plant.mu1}")
    print(f"  mu2 = {plant.mu2}")
    print(f"  g1  = {plant.g1}")
    print(f"  g2  = {plant.g2}")
    print("="*60 + "\n")
    if csv_sample_info is not None:
        print("CSV-selected samples:")
        print(f"  selected k1 sample index         = {int(csv_sample_info['k1_index'])}")
        print(f"  selected mu1 sample index        = {int(csv_sample_info['mu1_index'])}")
        print(f"  selected k2 sample index         = {int(csv_sample_info['k2_index'])}")
        print(f"  selected mu2 sample index        = {int(csv_sample_info['mu2_index'])}")
        print(f"  resolved k1 CSV row              = {int(csv_sample_info['k1_row'])}")
        print(f"  resolved mu1 CSV row             = {int(csv_sample_info['mu1_row'])}")
        print(f"  resolved k2 CSV row              = {int(csv_sample_info['k2_row'])}")
        print(f"  resolved mu2 CSV row             = {int(csv_sample_info['mu2_row'])}")
        print(f"  g1/g2 sampled from seed          = {SEED + 1}")
        print()

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

    k1_init_guess, mu1_init_guess, k2_init_guess, mu2_init_guess = initial_estimator_guesses(
        plant,
        envelopes,
    )

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
        k1_init_guess=k1_init_guess,
        mu1_init_guess=mu1_init_guess,
        k2_init_guess=k2_init_guess,
        mu2_init_guess=mu2_init_guess,
        freeze_controller_estimates=False,
    )

    sim_frozen = simulate(
        plant=plant,
        envelopes=envelopes,
        eq=eq,
        x1_0=x1_0,
        x2_0=x2_0,
        control_mode=CONTROL_MODE,
        constant_u_star=u_star,
        t_final=T_FINAL,
        dt=DT,
        k1_init_guess=k1_init_guess,
        mu1_init_guess=mu1_init_guess,
        k2_init_guess=k2_init_guess,
        mu2_init_guess=mu2_init_guess,
        freeze_controller_estimates=True,
    )

    zeta1_used_for_u = (
        float(sim["zeta1_used"][-1]) if CONTROL_MODE == "nominal" else float(plant.zeta1)
    )
    zeta2_used_for_u = (
        float(sim["zeta2_used"][-1]) if CONTROL_MODE == "nominal" else float(plant.zeta2)
    )

    print("Sampled plant:")
    print(f"  R0_1    = {plant.R0_1:.6f}")
    print(f"  R0_2    = {plant.R0_2:.6f}")
    print(f"  gamma1  = {plant.gamma1:.6f}")
    print(f"  gamma2  = {plant.gamma2:.6f}")
    print()
    print("Zeta summary:")
    print(f"  exact operator zeta1 (root solve)   = {true_zeta1:.6f}")
    print(f"  exact operator zeta2 (root solve)   = {true_zeta2:.6f}")
    print("  FNO zeta outputs                    = online controller estimates only")
    print(f"  zeta1 used in u_star calculation    = {true_zeta1:.6f}")
    print(f"  zeta2 used in u_star calculation    = {true_zeta2:.6f}")
    print(f"  zeta1 used in u calculation         = {zeta1_used_for_u:.6f}")
    print(f"  zeta2 used in u calculation         = {zeta2_used_for_u:.6f}")
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
    print(f"  use FNO zeta                   = {USE_FNO_ZETA}")
    print(f"  adapt FNO zeta from estimates  = {ADAPT_FNO_ZETA_FROM_ESTIMATES}")
    print(f"  use k_hat in FNO zeta          = {USE_ESTIMATED_K_IN_FNO_ZETA}")
    print(f"  use mu_hat in FNO zeta         = {USE_ESTIMATED_MU_IN_FNO_ZETA}")
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
    if CONTROL_MODE == "nominal":
        print(f"  final zeta1 used by controller = {sim['zeta1_used'][-1]:.6e}")
        print(f"  final zeta2 used by controller = {sim['zeta2_used'][-1]:.6e}")

    if PLOT_RESULTS:
        plot_tac_adaptive_figure(
            plant=plant,
            eq=eq,
            sim_adaptive=sim,
            sim_frozen=sim_frozen,
            u_star=u_star,
            save_path=f"{PLOT_PREFIX}.pdf",
            show=True,
        )


if __name__ == "__main__":
    main()
