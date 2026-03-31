# make_parametric_dataset.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for search_path in (ROOT, SRC):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

import config as cfg
from core import generate_dataset_from_parametric_families


def plot_sample_families(arrays: dict, n_plot: int = 20, save_path: str | None = None) -> None:
    a = arrays["a"]
    k = arrays["k"]
    mu = arrays["mu"]
    g = arrays["g"]

    n_available = k.shape[0]
    n_show = min(n_plot, n_available)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for j in range(n_show):
        axes[0].plot(a, k[j], alpha=0.8, lw=1.2)
        axes[1].plot(a, mu[j], alpha=0.8, lw=1.2)
        axes[2].plot(a, g[j], alpha=0.8, lw=1.2)

    axes[0].set_title(r"Samples of $k(a)$")
    axes[1].set_title(r"Samples of $\mu(a)$")
    axes[2].set_title(r"Samples of $g(a)$")

    for ax in axes:
        ax.set_xlabel(r"$a$")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"$k(a)$")
    axes[1].set_ylabel(r"$\mu(a)$")
    axes[2].set_ylabel(r"$g(a)$")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if cfg.SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def get_example_indices(zeta: np.ndarray) -> list[int]:
    order = np.argsort(zeta)
    n = len(order)

    if n == 0:
        return []
    if n == 1:
        return [int(order[0])]

    idxs = [
        int(order[max(0, n // 10)]),
        int(order[n // 2]),
        int(order[min(n - 1, (9 * n) // 10)]),
    ]
    return idxs


def plot_diagnostics(
    arrays: dict,
    df,
    save_path: str | None = None,
) -> None:
    a = arrays["a"]
    k = arrays["k"]
    mu = arrays["mu"]
    g = arrays["g"]
    zeta = arrays["zeta"]
    R0 = arrays["R0"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()

    # 1. Distribution of zeta
    axes[0].hist(zeta, bins=40)
    axes[0].set_title(r"Distribution of $\zeta$")
    axes[0].set_xlabel(r"$\zeta$")
    axes[0].set_ylabel("Count")

    # 2. Distribution of R0
    axes[1].hist(R0, bins=40)
    axes[1].set_title(r"Distribution of $R_0$")
    axes[1].set_xlabel(r"$R_0$")
    axes[1].set_ylabel("Count")

    # 3. zeta vs K_amp
    axes[2].scatter(df["K_amp"], df["zeta"], s=12, alpha=0.5)
    axes[2].set_title(r"$\zeta$ vs $K_{\mathrm{amp}}$")
    axes[2].set_xlabel(r"$K_{\mathrm{amp}}$")
    axes[2].set_ylabel(r"$\zeta$")

    # 4. zeta vs MU_min
    axes[3].scatter(df["MU_min"], df["zeta"], s=12, alpha=0.5)
    axes[3].set_title(r"$\zeta$ vs $\mu_{\min}$")
    axes[3].set_xlabel(r"$\mu_{\min}$")
    axes[3].set_ylabel(r"$\zeta$")

    # 5. zeta vs MU_sen
    axes[4].scatter(df["MU_sen"], df["zeta"], s=12, alpha=0.5)
    axes[4].set_title(r"$\zeta$ vs $\mu_{\mathrm{sen}}$")
    axes[4].set_xlabel(r"$\mu_{\mathrm{sen}}$")
    axes[4].set_ylabel(r"$\zeta$")

    # 6. zeta vs G_amp
    axes[5].scatter(df["G_amp"], df["zeta"], s=12, alpha=0.5)
    axes[5].set_title(r"$\zeta$ vs $G_{\mathrm{amp}}$")
    axes[5].set_xlabel(r"$G_{\mathrm{amp}}$")
    axes[5].set_ylabel(r"$\zeta$")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if cfg.SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    # Representative low/mid/high-zeta function examples
    example_indices = get_example_indices(zeta)
    labels = ["low", "mid", "high"][: len(example_indices)]

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for idx, label in zip(example_indices, labels):
        axes2[0].plot(a, k[idx], lw=2, label=fr"{label}, $\zeta={zeta[idx]:.3f}$")
        axes2[1].plot(a, mu[idx], lw=2, label=fr"{label}, $\zeta={zeta[idx]:.3f}$")
        axes2[2].plot(a, g[idx], lw=2, label=fr"{label}, $\zeta={zeta[idx]:.3f}$")

    axes2[0].set_title(r"Representative $k(a)$")
    axes2[1].set_title(r"Representative $\mu(a)$")
    axes2[2].set_title(r"Representative $g(a)$")

    axes2[0].set_ylabel(r"$k(a)$")
    axes2[1].set_ylabel(r"$\mu(a)$")
    axes2[2].set_ylabel(r"$g(a)$")

    for ax in axes2:
        ax.set_xlabel(r"$a$")
        ax.grid(True, alpha=0.25)
        ax.legend()

    if save_path is not None:
        example_path = save_path.replace(".png", "_examples.png")
        Path(example_path).parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(example_path, dpi=200, bbox_inches="tight")

    if cfg.SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig2)


def main(n_plot: int = 20) -> None:
    arrays, df = generate_dataset_from_parametric_families(
        n_families=cfg.N_FAMILIES,
        seed=cfg.SEED,
        n_grid=cfg.N_GRID,
        target_R0_min=cfg.TARGET_R0_MIN,
        config_module=cfg,
    )

    Path(cfg.NPZ_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.NPZ_PATH, **arrays)
    df.to_csv(cfg.CSV_PATH, index=False)

    sample_fig_path = f"{cfg.PLOT_PREFIX}_samples.png"
    diag_fig_path = f"{cfg.PLOT_PREFIX}_diagnostics.png"

    plot_sample_families(arrays, n_plot=n_plot, save_path=sample_fig_path)
    plot_diagnostics(arrays, df, save_path=diag_fig_path)

    print(f"Saved dataset to {cfg.NPZ_PATH}")
    print(f"Saved summary to {cfg.CSV_PATH}")
    print(f"Saved sample plot to {sample_fig_path}")
    print(f"Saved diagnostics plot to {diag_fig_path}")
    print()
    print(df[["family_id", "R0", "zeta"]].describe())


if __name__ == "__main__":
    main(n_plot=20)
