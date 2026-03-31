from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for search_path in (ROOT, SRC):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

from neuralop.models import FNO


if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

torch.set_float32_matmul_precision("high")


class ZetaFNORegressor(nn.Module):
    def __init__(
        self,
        n_modes: int = 16,
        hidden_channels: int = 64,
        in_channels: int = 2,
        fno_out_channels: int = 32,
        projection_hidden_channels: int = 64,
        lifting_channels: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()

        self.fno = FNO(
            n_modes=(n_modes,),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=fno_out_channels,
            projection_channels=projection_hidden_channels,
            lifting_channels=lifting_channels,
            n_layers=n_layers,
        )

        self.regressor = nn.Sequential(
            nn.Linear(fno_out_channels, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_grid, channels)
        x = x.permute(0, 2, 1)   # -> (batch, channels, n_grid)
        y = self.fno(x)          # -> expected: (batch, fno_out_channels, n_grid)
        y = y.mean(dim=-1)       # global average pooling over grid
        y = self.regressor(y)
        return y.squeeze(-1)



def prepare_training_arrays(npz_path: str | Path, seed: int = 0):
    data = np.load(npz_path)

    mu = data["mu"].astype(np.float32)
    k = data["k"].astype(np.float32)
    zeta = data["zeta"].astype(np.float32)

    if mu.ndim != 2 or k.ndim != 2:
        raise ValueError("Expected 'mu' and 'k' to have shape (n_samples, n_grid).")
    if zeta.ndim != 1:
        raise ValueError("Expected 'zeta' to have shape (n_samples,).")

    n_samples, n_grid = mu.shape
    if k.shape != (n_samples, n_grid):
        raise ValueError("Shapes of 'k' and 'mu' do not match.")
    if zeta.shape[0] != n_samples:
        raise ValueError("Length of 'zeta' does not match number of samples.")

    x = np.stack([mu, k], axis=-1)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)

    return x[perm], zeta[perm]



def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    n = x.shape[0]
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
    x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)



def normalize_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
):
    x_mean = x_train.mean(axis=(0, 1), keepdims=True)
    x_std = x_train.std(axis=(0, 1), keepdims=True) + 1e-8
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    stats = {
        "x_mean": x_mean.squeeze().tolist(),
        "x_std": x_std.squeeze().tolist(),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
    }

    return (
        (x_train - x_mean) / x_std,
        (y_train - y_mean) / y_std,
        (x_val - x_mean) / x_std,
        (y_val - y_mean) / y_std,
        (x_test - x_mean) / x_std,
        (y_test - y_mean) / y_std,
        stats,
    )



def to_cpu_tensor(x: np.ndarray, y: np.ndarray):
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )



def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
    num_workers: int,
):
    x_t, y_t = to_cpu_tensor(x, y)
    return DataLoader(
        TensorDataset(x_t, y_t),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )



def evaluate(model, loader, y_mean: float, y_std: float):
    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb)

            batch_count = xb.size(0)
            total_mse += torch.sum((pred - yb) ** 2).item()

            pred_denorm = pred * y_std + y_mean
            y_denorm = yb * y_std + y_mean
            total_mae += torch.sum(torch.abs(pred_denorm - y_denorm)).item()
            total_count += batch_count

    mse = total_mse / total_count
    mae = total_mae / total_count
    return mse, mae



def collect_predictions(model, loader, y_mean: float, y_std: float):
    model.eval()

    y_true_parts = []
    y_pred_parts = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb)

            pred_denorm = (pred * y_std + y_mean).detach().cpu().numpy()
            y_denorm = (yb * y_std + y_mean).detach().cpu().numpy()

            y_pred_parts.append(pred_denorm)
            y_true_parts.append(y_denorm)

    return np.concatenate(y_true_parts), np.concatenate(y_pred_parts)



def plot_loss_curves(history: dict, save_path: str | Path) -> None:
    epochs = history["epoch"]
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_mse"], label="Train MSE")
    plt.plot(epochs, history["val_mse"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training and validation loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()



def plot_test_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path) -> None:
    errors = y_pred - y_true
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    plt.hist(errors, bins=30)
    plt.xlabel(r"Prediction error $(\hat{\zeta}-\zeta)$")
    plt.ylabel("Count")
    plt.title("Test-set error distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()



def plot_test_parity(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_true, y_pred, s=20, alpha=0.7)
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel(r"True $\zeta$")
    plt.ylabel(r"Predicted $\hat{\zeta}$")
    plt.title("Test-set parity plot")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()



def train_model(
    data_path,
    save_dir,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    n_modes: int = 16,
    hidden_channels: int = 64,
    fno_out_channels: int = 32,
    n_layers: int = 4,
    batch_size: int = 16,
    eval_batch_size: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    x, y = prepare_training_arrays(data_path, seed=seed)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(x, y)

    x_train, y_train, x_val, y_val, x_test, y_test, stats = normalize_data(
        x_train, y_train, x_val, y_val, x_test, y_test
    )

    eval_batch_size = eval_batch_size or batch_size
    pin_memory = device.type == "cuda"

    train_loader = make_loader(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    val_loader = make_loader(
        x_val,
        y_val,
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    test_loader = make_loader(
        x_test,
        y_test,
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    model = ZetaFNORegressor(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=2,
        fno_out_channels=fno_out_channels,
        n_layers=n_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    y_mean = stats["y_mean"]
    y_std = stats["y_std"]

    best_val_mae = float("inf")
    best_path = save_dir / "best_model.pt"

    history = {
        "epoch": [],
        "train_mse": [],
        "val_mse": [],
        "val_mae": [],
    }

    print(f"Using device: {device}")
    print(f"Train size: {x_train.shape[0]}, Val size: {x_val.shape[0]}, Test size: {x_test.shape[0]}")
    print(f"Batch size: {batch_size}, Eval batch size: {eval_batch_size}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_train_examples = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            batch_count = xb.size(0)
            running_loss += loss.item() * batch_count
            num_train_examples += batch_count

        scheduler.step()

        train_mse = running_loss / num_train_examples
        val_mse, val_mae = evaluate(model, val_loader, y_mean, y_std)

        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "stats": stats,
                    "args": {
                        "n_modes": n_modes,
                        "hidden_channels": hidden_channels,
                        "fno_out_channels": fno_out_channels,
                        "n_layers": n_layers,
                        "epochs": epochs,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "batch_size": batch_size,
                        "eval_batch_size": eval_batch_size,
                        "num_workers": num_workers,
                        "seed": seed,
                    },
                },
                best_path,
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"train_mse={train_mse:.6f} | "
                f"val_mse={val_mse:.6f} | "
                f"val_mae={val_mae:.6f}"
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    test_mse, test_mae = evaluate(model, test_loader, y_mean, y_std)
    y_true_test, y_pred_test = collect_predictions(model, test_loader, y_mean, y_std)
    test_errors = y_pred_test - y_true_test

    with (save_dir / "normalization.json").open("w") as f:
        json.dump(stats, f, indent=2)

    with (save_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    # Create figure directory in figures/ with same structure as models/
    rel_path = save_dir.relative_to(ROOT)
    figure_dir = ROOT / "figures" / rel_path
    plot_loss_curves(history, figure_dir / "loss_curves.png")
    plot_test_error_distribution(y_true_test, y_pred_test, figure_dir / "test_error_distribution.png")
    plot_test_parity(y_true_test, y_pred_test, figure_dir / "test_parity.png")

    metrics = {
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
        "test_error_mean": float(np.mean(test_errors)),
        "test_error_std": float(np.std(test_errors)),
        "test_error_max_abs": float(np.max(np.abs(test_errors))),
    }

    with (save_dir / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nBest model saved to: {best_path}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test error mean: {metrics['test_error_mean']:.6f}")
    print(f"Test error std: {metrics['test_error_std']:.6f}")
    print(f"Test max abs error: {metrics['test_error_max_abs']:.6f}")

    return {
        "model": model,
        "stats": stats,
        "history": history,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "y_true_test": y_true_test,
        "y_pred_test": y_pred_test,
        "best_path": best_path,
    }



def load_fno_checkpoint(checkpoint_path, device=device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint["args"]
    stats = checkpoint["stats"]

    model = ZetaFNORegressor(
        n_modes=ckpt_args.get("n_modes", 16),
        hidden_channels=ckpt_args.get("hidden_channels", 64),
        in_channels=2,
        fno_out_channels=ckpt_args.get("fno_out_channels", 32),
        n_layers=ckpt_args.get("n_layers", 4),
    ).to(device)

    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_")}
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    return model, stats, checkpoint



def predict_zeta_fno(model, stats, mu_vals, k_vals, device=device):
    mu_vals = np.asarray(mu_vals, dtype=np.float32)
    k_vals = np.asarray(k_vals, dtype=np.float32)

    if mu_vals.shape != k_vals.shape:
        raise ValueError("mu_vals and k_vals must have the same shape.")

    x = np.stack([mu_vals, k_vals], axis=-1)[None, :, :]

    x_mean = np.asarray(stats["x_mean"], dtype=np.float32).reshape(1, 1, 2)
    x_std = np.asarray(stats["x_std"], dtype=np.float32).reshape(1, 1, 2)
    y_mean = float(stats["y_mean"])
    y_std = float(stats["y_std"])

    x_norm = (x - x_mean) / x_std
    x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device)

    with torch.no_grad():
        zeta_norm = model(x_tensor).item()

    return zeta_norm * y_std + y_mean


if __name__ == "__main__":
    train_model(
        data_path=ROOT / "datasets" / "simple_parametric_dataset.npz",
        save_dir=ROOT / "models" / "fno_zeta" / "run_001",
        epochs=100,
        lr=4e-3,
        weight_decay=1e-6,
        n_modes=32,
        hidden_channels=64,
        fno_out_channels=32,
        n_layers=4,
        batch_size=64,
        eval_batch_size=64,
        num_workers=0,
        seed=0,
    )
