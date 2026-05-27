#!/usr/bin/env python3
"""Train a tiny CNN to classify LCD digits (0-9) from 12×18 px grayscale crops.

Architecture:
    Input (1×18×12)
    Conv(16, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  →  (16×9×6)
    Conv(32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  →  (32×4×3)
    Flatten(384) → Dropout(0.4) → Linear(64) → ReLU → Dropout(0.2) → Linear(10)
    ~30 k parameters — runs on any hardware including RPi 2.

Usage:
    conda run -n gs2026 python scripts/train_cnn.py
    conda run -n gs2026 python scripts/train_cnn.py --epochs 200 --lr 5e-4
"""

from __future__ import annotations

import argparse
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

ROOT = pathlib.Path(__file__).parent.parent
DATASET_DIR = ROOT / "data" / "dataset_balanced"
MODELS_DIR = ROOT / "models"

# ── Model ─────────────────────────────────────────────────────────────────────


class DigitCNN(nn.Module):
    """Tiny CNN for 12×18 grayscale LCD digit classification."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # → 16×9×6
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # → 32×4×3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(32 * 4 * 3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Dataset ───────────────────────────────────────────────────────────────────


class DigitDataset(Dataset):
    """Loads PNG crops from data/dataset_balanced/{0-9}/."""

    def __init__(self, root: pathlib.Path, transform=None) -> None:
        self.samples: list[tuple[pathlib.Path, int]] = []
        for label in range(10):
            for p in sorted((root / str(label)).glob("*.png")):
                self.samples.append((p, label))
        random.shuffle(self.samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label


# ── Training helpers ──────────────────────────────────────────────────────────


def build_transforms(augment: bool):
    base = [
        transforms.ToTensor(),          # → (1, H, W) float32 in [0,1]
        transforms.Normalize((0.5,), (0.5,)),
    ]
    if augment:
        return transforms.Compose([
            transforms.RandomAffine(
                degrees=4,
                translate=(0.1, 0.1),
                fill=0,
            ),
            *base,
        ])
    return transforms.Compose(base)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total_acc = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy(logits, labels)
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = total_acc = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += F.cross_entropy(logits, labels).item()
        total_acc += accuracy(logits, labels)
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def confusion_matrix(model, loader, device, num_classes=10):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        for t, p in zip(labels.numpy(), preds):
            cm[t, p] += 1
    return cm


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    full_dataset = DigitDataset(DATASET_DIR)
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Override transforms: augment only training set
    train_ds.dataset = DigitDataset(DATASET_DIR, transform=build_transforms(augment=True))
    # Re-split with same indices to keep augmented train / clean val
    train_ds_aug = torch.utils.data.Subset(
        DigitDataset(DATASET_DIR, transform=build_transforms(augment=True)),
        train_ds.indices,
    )
    val_ds_clean = torch.utils.data.Subset(
        DigitDataset(DATASET_DIR, transform=build_transforms(augment=False)),
        val_ds.indices,
    )

    train_loader = DataLoader(train_ds_aug, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds_clean, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds_aug)}  Val: {len(val_ds_clean)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DigitCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    MODELS_DIR.mkdir(exist_ok=True)
    best_path = MODELS_DIR / "digit_cnn_best.pt"

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch",
                     dynamic_ncols=True)

    for epoch in epoch_bar:
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)
            marker = " ✓"
        else:
            marker = ""

        epoch_bar.set_postfix({
            "tr_loss": f"{tr_loss:.3f}",
            "tr_acc":  f"{tr_acc:.3f}",
            "va_loss": f"{va_loss:.3f}",
            "va_acc":  f"{va_acc:.3f}",
            "best":    f"{best_val_acc:.3f}",
            "mark":    marker.strip() or "-",
        })

    epoch_bar.close()
    print(f"\nBest val acc: {best_val_acc:.4f}")

    # ── Final evaluation with best weights ────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=device))
    cm = confusion_matrix(model, val_loader, device)

    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "     " + "".join(f"{i:5}" for i in range(10))
    print(header)
    print("    " + "-" * (5 * 10 + 1))
    per_class_acc = []
    for i in range(10):
        row_str = f"  {i} |" + "".join(f"{cm[i,j]:5}" for j in range(10))
        n = cm[i].sum()
        acc_i = cm[i, i] / n if n > 0 else 0.0
        per_class_acc.append(acc_i)
        print(row_str + f"   acc={acc_i:.2f}")

    overall = cm.diagonal().sum() / cm.sum()
    print(f"\nOverall val accuracy: {overall:.4f}  ({cm.diagonal().sum()}/{cm.sum()})")
    print(f"Mean per-class acc:   {np.mean(per_class_acc):.4f}")

    # ── Export ────────────────────────────────────────────────────────────────
    dummy = torch.zeros(1, 1, 18, 12, device=device)
    model.eval()

    # TorchScript — works on any platform that has PyTorch installed
    ts_path = MODELS_DIR / "digit_cnn.torchscript.pt"
    traced = torch.jit.trace(model, dummy)
    traced.save(str(ts_path))
    print(f"\nTorchScript model → {ts_path.relative_to(ROOT)}")

    # ONNX — preferred for onnxruntime deployment (lighter than full PyTorch)
    onnx_path = MODELS_DIR / "digit_cnn.onnx"
    try:
        torch.onnx.export(
            traced,
            dummy,
            str(onnx_path),
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=12,  # opset 12: widely supported by onnxruntime on ARMv7
            dynamo=False,      # use legacy TorchScript-based exporter
        )
        print(f"ONNX model        → {onnx_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"ONNX export skipped ({e}); use TorchScript instead.")

    print(f"PyTorch weights   → {best_path.relative_to(ROOT)}")
    print("\nTo run on RPi 2 (pick one):")
    print("  # Option A — onnxruntime (lightest):")
    print("  pip install onnxruntime")
    print("  sess = onnxruntime.InferenceSession('models/digit_cnn.onnx')")
    print("  digit = sess.run(['logits'], {'image': arr})[0].argmax()")
    print("  # Option B — TorchScript:")
    print("  model = torch.jit.load('models/digit_cnn.torchscript.pt')")
    print("  digit = model(tensor).argmax().item()")


if __name__ == "__main__":
    main()
