"""
Fine-tune UWnet underwater image enhancer on paired murky/clear images.
Run from project root: python train_enhancer.py
"""
# 1. THIS MUST BE FIRST: __future__ imports have to be at the absolute top
from __future__ import annotations

# 2. Standard system imports
import os
import sys
from pathlib import Path

# 3. Path modifications (This helps PyTorch find your model.py file)
sys.path.append(os.getcwd())
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 4. Third-party library imports
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    # TorchMetrics provides standard, well-tested restoration metrics.
    # We use image-aware implementations rather than hand-rolled formulas.
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "torchmetrics is required for PSNR/SSIM. Install it with:\n"
        "  pip install torchmetrics\n"
        f"Original import error: {e}"
    )

# 5. Your local model import
from models.stage1_enhancer import UWnet
import models.stage1_enhancer as original_model_module

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _normalize_state_dict_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip common distributed / Lightning prefixes from parameter names."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        k = key
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("model."):
            k = k[len("model.") :]
        out[k] = value
    return out


def load_checkpoint_into_model(
    model: nn.Module,
    ckpt_path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    strict: bool = False,
) -> nn.Module:
    """
    Load weights from a .ckpt / .pth file into ``model``.
    """
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")

    if map_location is None:
        map_location = "cpu"

    # Namespace redirect for old checkpoints:
    # If the checkpoint was saved when UWnet lived in a top-level `model.py`,
    # unpickling may try to import `model`. We map that name to the current module.
    sys.modules["model"] = original_model_module
    blob = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(blob, nn.Module):
        state = blob.state_dict()
    elif isinstance(blob, dict):
        if "state_dict" in blob and isinstance(blob["state_dict"], dict):
            state = blob["state_dict"]
        elif "model_state_dict" in blob and isinstance(blob["model_state_dict"], dict):
            state = blob["model_state_dict"]
        elif "model" in blob and isinstance(blob["model"], dict):
            state = blob["model"]
        else:
            tensor_vals = {k: v for k, v in blob.items() if isinstance(v, torch.Tensor)}
            if tensor_vals:
                state = tensor_vals
            else:
                raise ValueError(
                    f"Could not find weights in {path.name}: "
                    "expected 'state_dict', tensor-valued dict, or nn.Module."
                )
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(blob).__name__}")

    state = _normalize_state_dict_keys(state)
    incompatible = model.load_state_dict(state, strict=strict)
    if incompatible.missing_keys:
        print(f"[load_checkpoint] missing keys ({len(incompatible.missing_keys)}): "
              f"{incompatible.missing_keys[:5]}{'...' if len(incompatible.missing_keys) > 5 else ''}")
    if incompatible.unexpected_keys:
        print(f"[load_checkpoint] unexpected keys ({len(incompatible.unexpected_keys)}): "
              f"{incompatible.unexpected_keys[:5]}{'...' if len(incompatible.unexpected_keys) > 5 else ''}")
    return model


class PairedEnhancementDataset(Dataset):
    """Pairs murky images with clean targets; filenames must match in both folders."""

    def __init__(
        self,
        murky_dir: str | Path,
        clear_dir: str | Path,
        *,
        transform_murky,
        transform_clear,
    ) -> None:
        self.murky_dir = Path(murky_dir)
        self.clear_dir = Path(clear_dir)
        if not self.murky_dir.is_dir():
            raise FileNotFoundError(f"Murky dir not found: {self.murky_dir}")
        if not self.clear_dir.is_dir():
            raise FileNotFoundError(f"Clear dir not found: {self.clear_dir}")

        murky_files = sorted(p for p in self.murky_dir.iterdir() if p.is_file() and _is_image_file(p))
        self.pairs: list[tuple[Path, Path]] = []
        missing_clear: list[str] = []
        for m in murky_files:
            c = self.clear_dir / m.name
            if c.is_file():
                self.pairs.append((m, c))
            else:
                missing_clear.append(m.name)

        if not self.pairs:
            raise RuntimeError(
                f"No paired images found under {self.murky_dir} / {self.clear_dir}. "
                "Use matching filenames in both folders."
            )
        if missing_clear:
            print(f"[Dataset] Warning: {len(missing_clear)} murky file(s) have no clear twin (skipped).")

        self.transform_murky = transform_murky
        self.transform_clear = transform_clear

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        murky_path, clear_path = self.pairs[idx]
        murky = Image.open(murky_path).convert("RGB")
        clear = Image.open(clear_path).convert("RGB")
        return self.transform_murky(murky), self.transform_clear(clear)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA not available; training will run on CPU.")
    else:
        print(f"🚀 Training on GPU: {torch.cuda.get_device_name(0)}")

    murky_root = PROJECT_ROOT / "datasets" / "train_murky"
    clear_root = PROJECT_ROOT / "datasets" / "train_clear"
    ckpt_path = PROJECT_ROOT / "weights" / "model.ckpt"
    weights_dir = PROJECT_ROOT / "weights"
    save_path = weights_dir / "best_enhancer_finetuned.pth"
    weights_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    dataset = PairedEnhancementDataset(
        murky_root,
        clear_root,
        transform_murky=transform,
        transform_clear=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = UWnet(num_layers=3).to(device)
    load_checkpoint_into_model(model, ckpt_path, map_location=device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50
    model.train()

    print("\nStarting Training Loop...")

    # -------------------------
    # Epoch-level tracking
    # -------------------------
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_psnr: list[float] = []
    val_ssim: list[float] = []

    # TorchMetrics modules handle internal state and device placement cleanly.
    # Since inputs are ToTensor() in [0, 1], we set data_range=1.0.
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # For now we validate on the training loader to keep it simple.
    # Swap `val_loader` to a real validation loader when you add val paths.
    val_loader = loader

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (murky, clear) in enumerate(loader, start=1):
            murky = murky.to(device, non_blocking=True)
            clear = clear.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(murky)
            loss = criterion(out, clear)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch}/{num_epochs} | Batch {batch_idx}/{len(loader)} | "
                    f"loss={loss_val:.6f}"
                )

        mean_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(mean_loss)

        # -------------------------
        # Validation (eval + no_grad)
        # -------------------------
        model.eval()
        psnr_metric.reset()
        ssim_metric.reset()

        val_epoch_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for murky, clear in val_loader:
                murky = murky.to(device, non_blocking=True)
                clear = clear.to(device, non_blocking=True)

                out = model(murky)

                # Clamp for metric stability (model outputs may slightly exceed [0, 1]).
                out_for_metrics = out.clamp(0.0, 1.0)
                clear_for_metrics = clear.clamp(0.0, 1.0)

                vloss = criterion(out, clear)
                val_epoch_loss += float(vloss.item())
                val_batches += 1

                # TorchMetrics modules are on `device`, so updates happen without CPU hops.
                psnr_metric.update(out_for_metrics, clear_for_metrics)
                ssim_metric.update(out_for_metrics, clear_for_metrics)

        mean_val_loss = val_epoch_loss / max(val_batches, 1)
        mean_val_psnr = float(psnr_metric.compute().detach().item())
        mean_val_ssim = float(ssim_metric.compute().detach().item())

        val_losses.append(mean_val_loss)
        val_psnr.append(mean_val_psnr)
        val_ssim.append(mean_val_ssim)

        print(
            f"Epoch {epoch}/{num_epochs} done | "
            f"train_loss={mean_loss:.6f} | val_loss={mean_val_loss:.6f} | "
            f"PSNR={mean_val_psnr:.3f} dB | SSIM={mean_val_ssim:.4f}"
        )

        # Back to train mode for next epoch.
        model.train()

        torch.save(model.state_dict(), save_path)
    
    print(f"\n🎉 Training Complete! Weights saved to: {save_path}")

    # -------------------------
    # Plot training curves
    # -------------------------
    metrics_plot_path = weights_dir / "training_metrics.png"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    epochs = list(range(1, len(train_losses) + 1))

    # Subplot 1: Loss curves
    ax = axes[0]
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=2)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Subplot 2: PSNR curve
    ax = axes[1]
    ax.plot(epochs, val_psnr, label="PSNR", color="tab:green", linewidth=2)
    ax.set_title("PSNR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("dB")
    ax.grid(True, alpha=0.3)

    # Subplot 3: SSIM curve
    ax = axes[2]
    ax.plot(epochs, val_ssim, label="SSIM", color="tab:purple", linewidth=2)
    ax.set_title("SSIM")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Training / Validation Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(metrics_plot_path, dpi=200)
    plt.close(fig)

    print(f"📈 Saved training curves to: {metrics_plot_path}")

if __name__ == "__main__":
    main()