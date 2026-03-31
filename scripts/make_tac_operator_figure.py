from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for search_path in (ROOT, SRC):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

from train_fno import (
    collect_predictions,
    load_fno_checkpoint,
    make_loader,
    normalize_data,
    prepare_training_arrays,
    split_dataset,
)


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
# Default settings (matching make_tac_sim_figure)
# ============================================================

DATA_PATH = ROOT / "datasets" / "simple_parametric_dataset.npz"
CHECKPOINT_PATH = ROOT / "models" / "fno_zeta" / "run_001" / "best_model.pt"
SAVE_PATH = ROOT / "figures" / "tac_operator_figure.pdf"

SEED = 123
N_SHOW = 10
SELECTION_MODE = "quantile"
EVAL_BATCH_SIZE = 256


def set_size(width, fraction=1, subplots=(1, 1), height_add=0):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
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



def _denormalize_x(x_norm: np.ndarray, stats: dict) -> np.ndarray:
    x_mean = np.asarray(stats["x_mean"], dtype=np.float32).reshape(1, 1, 2)
    x_std = np.asarray(stats["x_std"], dtype=np.float32).reshape(1, 1, 2)
    return x_norm * x_std + x_mean



def _select_representative_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_show: int,
    mode: str = "quantile",
) -> np.ndarray:
    n = len(y_true)
    n_show = min(max(1, n_show), n)

    if mode == "best":
        order = np.argsort(np.abs(y_pred - y_true))
        return np.sort(order[:n_show])

    order = np.argsort(y_true)
    quantile_positions = np.linspace(0, n - 1, n_show)
    picked = np.unique(np.round(quantile_positions).astype(int))

    idx = list(order[picked])
    if len(idx) < n_show:
        remaining = [i for i in order if i not in idx]
        remaining = sorted(remaining, key=lambda i: abs(y_pred[i] - y_true[i]))
        idx.extend(remaining[: n_show - len(idx)])
    return np.array(idx[:n_show], dtype=int)



def make_operator_figure(
    data_path: str | Path,
    checkpoint_path: str | Path,
    save_path: str | Path,
    seed: int = 0,
    n_show: int = 10,
    eval_batch_size: int = 512,
    selection_mode: str = "quantile",
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = prepare_training_arrays(data_path, seed=seed)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(x, y)

    x_train_n, y_train_n, x_val_n, y_val_n, x_test_n, y_test_n, stats = normalize_data(
        x_train, y_train, x_val, y_val, x_test, y_test
    )

    model, ckpt_stats, _ = load_fno_checkpoint(checkpoint_path, device=device)
    stats = ckpt_stats or stats

    test_loader = make_loader(
        x_test_n,
        y_test_n,
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )
    y_true, y_pred = collect_predictions(model, test_loader, stats["y_mean"], stats["y_std"])
    x_test_denorm = _denormalize_x(x_test_n, stats)
    mu_test = x_test_denorm[:, :, 0]
    k_test = x_test_denorm[:, :, 1]
    n_grid = mu_test.shape[1]
    grid = np.linspace(0.0, 1.0, n_grid)

    chosen = _select_representative_indices(y_true, y_pred, n_show=n_show, mode=selection_mode)
    chosen = chosen[np.argsort(y_true[chosen])]
    errors = y_true - y_pred

    # Save only the chosen k and mu values that are plotted in the figure
    sample_dir = Path(save_path).parent
    
    # Save k samples for chosen indices (one row per chosen sample, one column per grid point)
    k_chosen_df = pd.DataFrame(
        k_test[chosen],
        columns=[f"grid_{i:.3f}" for i in grid]
    )
    k_csv_path = sample_dir / "k_samples.csv"
    k_chosen_df.to_csv(k_csv_path, index=False)
    
    # Save mu samples for chosen indices (one row per chosen sample, one column per grid point)
    mu_chosen_df = pd.DataFrame(
        mu_test[chosen],
        columns=[f"grid_{i:.3f}" for i in grid]
    )
    mu_csv_path = sample_dir / "mu_samples.csv"
    mu_chosen_df.to_csv(mu_csv_path, index=False)
    
    # Save summary data (predictions, errors for all samples and marker for chosen)
    summary_df = pd.DataFrame({
        "sample_id": np.arange(len(y_true)),
        "y_true": y_true,
        "y_pred": y_pred,
        "error": errors,
        "is_chosen": np.isin(np.arange(len(y_true)), chosen),
    })
    summary_csv_path = sample_dir / "operator_predictions.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"Saved {len(chosen)} chosen samples to: {k_csv_path}, {mu_csv_path}, {summary_csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=set_size(516, subplots=(1, 3), height_add=0.2), constrained_layout=False)
    ax_k, ax_mu, ax_resid = axes

    # (a) representative k(x) fields
    sample_handles = []
    sample_labels = []
    for rank, idx in enumerate(chosen, start=1):
        line_k, = ax_k.plot(grid, k_test[idx], label=str(rank), linewidth=1.35)
        ax_mu.plot(grid, mu_test[idx], label=str(rank), color=line_k.get_color(), linewidth=1.35)
        sample_handles.append(line_k)
        sample_labels.append(str(rank-1))
    ax_k.set_xlim(0.0, 1.0)
    ax_k.set_xlabel(r"Age $a$")
    ax_k.set_ylabel(r"$k(a)$")
    ax_k.set_title(r"(a) Example Fertilities", loc="left")
    ax_k.grid(True, alpha=0.22)

    # (b) representative mu(x) fields
    ax_mu.set_xlim(0.0, 1.0)
    ax_mu.set_xlabel(r"Age $a$")
    ax_mu.set_ylabel(r"$\mu(a)$")
    ax_mu.set_title(r"(b) Example Mortalities", loc="left")
    ax_mu.grid(True, alpha=0.22)

    # (d) residuals
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    pad = 0.04 * (hi - lo + 1e-12)
    lo -= pad
    hi += pad

    ax_resid.axhline(0.0, linestyle="--", linewidth=1.1)
    ax_resid.scatter(y_true, errors, s=22, alpha=0.55, rasterized=True)
    ax_resid.scatter(
        y_true[chosen],
        errors[chosen],
        s=54,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    for rank, idx in enumerate(chosen, start=1):
        ax_resid.annotate(
            str(rank-1),
            (y_true[idx], errors[idx]),
            xytext=(-2, -2.2),
            textcoords="offset points",
            fontsize=8,
        )
    ax_resid.set_xlim(lo, hi)
    ax_resid.set_xlabel(r"True $\zeta$")
    ax_resid.set_ylabel(r"$\zeta - \hat{\zeta}$", labelpad=2)
    ax_resid.set_title(r"(c) FNO Residuals $\mathcal{\hat{G}}_{\rm LS}$", loc="left")
    ax_resid.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax_resid.grid(True, alpha=0.22)

    fig.legend(
        sample_handles,
        sample_labels,
        loc="lower center",
        bbox_to_anchor=(0.57, -0.1),
        ncol=min(len(sample_labels), 10),
        frameon=False,
        handlelength=1.8,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    fig.text(0.29, 0, "Samples", ha="right", va="center", fontsize=10)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.30, wspace=0.42)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    make_operator_figure(
        data_path=DATA_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        save_path=SAVE_PATH,
        seed=SEED,
        n_show=N_SHOW,
        eval_batch_size=EVAL_BATCH_SIZE,
        selection_mode=SELECTION_MODE,
    )
