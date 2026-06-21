#!/usr/bin/env python3
"""Train a tiny CNN to classify LCD digits (0-9) from 12×18 px grayscale crops.

Architecture:
    Input (1×18×12)
    Conv(16, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  →  (16×9×6)
    Conv(32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  →  (32×4×3)
    Flatten(384) → Dropout(0.4) → Linear(64) → ReLU → Dropout(0.2) → Linear(10)
    ~30 k parameters — runs on any hardware including RPi 2.

Validation strategy:
    Images are split at the SOURCE IMAGE level (not crop level) so that the
    validation set measures end-to-end (E2E) accuracy on complete LCD images —
    the same metric that matters in production.  The best model checkpoint is
    selected based on E2E "both PH and Redux exact" accuracy.

Usage:
    conda run -n gs2026 python scripts/train_cnn.py
    conda run -n gs2026 python scripts/train_cnn.py --epochs 200 --lr 5e-4
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.cam_image_processor.lcd_detector import extract_digit_rois

DATASET_DIR  = ROOT / "data" / "dataset"        # unbalanced, all crops
CAPTURED_DIR = ROOT / "data" / "captured"
CONFIG_PATH  = ROOT / "config.yaml"
MODELS_DIR   = ROOT / "models"

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
    """Loads PNG crops from data/dataset/{0-9}/.

    Each sample filename encodes its source image stem, e.g.
    ``20260524_203415_D0.png``  →  source ``20260524_203415``.
    """

    def __init__(self, samples: list[tuple[pathlib.Path, int]], transform=None) -> None:
        self.samples = samples
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


def load_all_crops() -> list[tuple[pathlib.Path, int]]:
    """Return all (path, label) pairs from data/dataset/{0-9}/."""
    samples = []
    for label in range(10):
        for p in sorted((DATASET_DIR / str(label)).glob("*.png")):
            samples.append((p, label))
    return samples


def source_stem(crop_path: pathlib.Path) -> str:
    """Extract the source image stem from a crop filename.

    e.g. '20260524_203415_D0.png' → '20260524_203415'
    """
    name = crop_path.stem          # e.g. '20260524_203415_D0'
    parts = name.rsplit("_D", 1)   # split on last '_D'
    return parts[0] if len(parts) == 2 else name


def image_level_split(
    samples: list[tuple[pathlib.Path, int]],
    val_fraction: float,
    seed: int,
) -> tuple[list, list]:
    """Split samples by source image so no image appears in both train and val.

    Returns (train_samples, val_samples).
    """
    stems = sorted({source_stem(p) for p, _ in samples})
    rng = random.Random(seed)
    rng.shuffle(stems)
    n_val = max(1, int(len(stems) * val_fraction))
    val_stems = set(stems[:n_val])

    train = [(p, l) for p, l in samples if source_stem(p) not in val_stems]
    val   = [(p, l) for p, l in samples if source_stem(p) in val_stems]
    return train, val


def balance_samples(
    samples: list[tuple[pathlib.Path, int]],
    seed: int,
) -> list[tuple[pathlib.Path, int]]:
    """Downsample to the smallest class count (balanced dataset)."""
    by_class: dict[int, list] = {i: [] for i in range(10)}
    for p, l in samples:
        by_class[l].append((p, l))
    min_count = min(len(v) for v in by_class.values() if v)
    rng = random.Random(seed)
    balanced = []
    for lst in by_class.values():
        balanced.extend(rng.sample(lst, min(min_count, len(lst))))
    rng.shuffle(balanced)
    return balanced


# ── E2E validation helpers ────────────────────────────────────────────────────


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_digit_labels(ph: float, redux: int) -> dict[int, int]:
    ph_str    = f"{ph:.2f}".replace(".", "")
    redux_str = f"{int(redux):03d}"
    return {0: int(ph_str[0]), 2: int(ph_str[1]), 3: int(ph_str[2]),
            5: int(redux_str[0]), 6: int(redux_str[1]), 7: int(redux_str[2])}


def build_e2e_val_set(val_stems: set[str]) -> list[tuple[pathlib.Path, dict]]:
    """Return (png_path, meta) pairs for captured images whose stems are in val_stems."""
    records = []
    for yf in sorted(CAPTURED_DIR.glob("*.yaml")):
        if yf.stem not in val_stems:
            continue
        meta = yaml.safe_load(open(yf)) or {}
        ph = meta.get("PH"); rx = meta.get("Redux")
        if ph is None or rx is None or float(ph) == 0.0:
            continue
        for ext in (".png", ".jpg"):
            img_path = yf.with_suffix(ext)
            if img_path.exists():
                records.append((img_path, meta))
                break
    return records


@torch.no_grad()
def eval_e2e(
    model: nn.Module,
    val_images: list[tuple[pathlib.Path, dict]],
    digit_config: dict,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run model on full captured LCD images; return (ph_acc, rx_acc, both_acc)."""
    model.eval()
    ph_ok = rx_ok = both_ok = total = 0

    norm = transforms.Normalize((0.5,), (0.5,))

    for img_path, meta in val_images:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if (img > 240).mean() > 0.5:   # overexposed
            continue
        try:
            rois = extract_digit_rois(img, digit_config)
        except Exception:
            continue

        digits = []
        for i in (0, 2, 3, 5, 6, 7):
            roi = torch.tensor(rois[i].astype(np.float32) / 255.0).unsqueeze(0)
            roi = norm(roi).unsqueeze(0).to(device)   # (1,1,H,W)
            logits = model(roi)
            digits.append(logits.argmax(dim=1).item())

        ph_pred = round(digits[0] + digits[1] * 0.1 + digits[2] * 0.01, 2)
        rx_pred = digits[3] * 100 + digits[4] * 10 + digits[5]

        ph_gt = float(meta["PH"])
        rx_gt = int(meta["Redux"])

        p = abs(ph_pred - ph_gt) < 0.005
        r = rx_pred == rx_gt
        if p: ph_ok  += 1
        if r: rx_ok  += 1
        if p and r: both_ok += 1
        total += 1

    if total == 0:
        return 0.0, 0.0, 0.0
    return ph_ok / total, rx_ok / total, both_ok / total


# ── Training helpers ──────────────────────────────────────────────────────────


def build_transforms(augment: bool):
    base = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
    if augment:
        return transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), fill=0),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            *base,
        ])
    return transforms.Compose(base)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--val-split",  type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--no-cuda",    action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")

    config       = load_config()
    digit_config = config["lcd"]["digits"]

    # ── Image-level train / val split ─────────────────────────────────────────
    all_samples = load_all_crops()
    if not all_samples:
        print(f"No crops found in {DATASET_DIR}. Run build_dataset.py first.")
        return

    train_samples, val_crop_samples = image_level_split(
        all_samples, val_fraction=args.val_split, seed=args.seed
    )

    # Balanced training set (downsample to smallest class)
    train_balanced = balance_samples(train_samples, seed=args.seed)

    # E2E val: only captured images (already-unwarped PNGs)
    val_stems = {source_stem(p) for p, _ in val_crop_samples}
    val_images = build_e2e_val_set(val_stems)

    train_ds = DigitDataset(train_balanced,    transform=build_transforms(augment=True))
    val_ds   = DigitDataset(val_crop_samples,  transform=build_transforms(augment=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2)

    print(f"\nDataset split (image-level):")
    print(f"  Train crops (balanced) : {len(train_balanced)}")
    print(f"  Val crops              : {len(val_crop_samples)}")
    print(f"  E2E val images         : {len(val_images)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model   = DigitCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters             : {n_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_e2e   = 0.0
    best_digit = 0.0
    MODELS_DIR.mkdir(exist_ok=True)
    best_path  = MODELS_DIR / "digit_cnn_best.pt"

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch",
                     dynamic_ncols=True)

    for epoch in epoch_bar:
        # -- train --
        model.train()
        tr_loss = tr_acc = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_acc  += (logits.argmax(1) == labels).float().mean().item()
        tr_loss /= len(train_loader)
        tr_acc  /= len(train_loader)

        # -- per-digit val (fast, every epoch) --
        model.eval()
        va_loss = va_acc = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                va_loss += F.cross_entropy(logits, labels).item()
                va_acc  += (logits.argmax(1) == labels).float().mean().item()
        va_loss /= max(len(val_loader), 1)
        va_acc  /= max(len(val_loader), 1)

        # -- E2E val (every 5 epochs to keep training fast) --
        if epoch % 5 == 0 or epoch == args.epochs:
            ph_acc, rx_acc, e2e_acc = eval_e2e(model, val_images, digit_config, device)
        else:
            e2e_acc = best_e2e   # carry forward

        scheduler.step()

        # Save best model based on E2E both-exact accuracy
        marker = ""
        if e2e_acc > best_e2e or (e2e_acc == best_e2e and va_acc > best_digit):
            best_e2e   = e2e_acc
            best_digit = va_acc
            torch.save(model.state_dict(), best_path)
            marker = " ✓"

        epoch_bar.set_postfix({
            "tr_acc":  f"{tr_acc:.3f}",
            "va_acc":  f"{va_acc:.3f}",
            "e2e":     f"{e2e_acc:.3f}",
            "best_e2e":f"{best_e2e:.3f}",
            "mark":    marker.strip() or "-",
        })

    epoch_bar.close()
    print(f"\nBest E2E both-exact: {best_e2e:.4f}")

    # ── Final evaluation with best weights ────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=device))

    # Per-digit confusion matrix on val crops
    cm = np.zeros((10, 10), dtype=int)
    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            for t, p in zip(labels.numpy(), preds):
                cm[t, p] += 1

    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "     " + "".join(f"{i:5}" for i in range(10))
    print(header)
    print("    " + "-" * 51)
    per_class_acc = []
    for i in range(10):
        row_str = f"  {i} |" + "".join(f"{cm[i,j]:5}" for j in range(10))
        n = cm[i].sum()
        acc_i = cm[i, i] / n if n > 0 else 0.0
        per_class_acc.append(acc_i)
        print(row_str + f"   acc={acc_i:.3f}")

    overall = cm.diagonal().sum() / cm.sum()
    print(f"\nOverall val digit acc : {overall:.4f}  ({cm.diagonal().sum()}/{cm.sum()})")
    print(f"Mean per-class acc    : {np.mean(per_class_acc):.4f}")

    # Full E2E report on val images
    print(f"\n── E2E validation on {len(val_images)} held-out full images ──")
    ph_acc, rx_acc, e2e_acc = eval_e2e(model, val_images, digit_config, device)
    total_e2e = len(val_images)
    print(f"  PH exact match   : {ph_acc*total_e2e:.0f}/{total_e2e}  ({100*ph_acc:.1f}%)")
    print(f"  Redux exact match: {rx_acc*total_e2e:.0f}/{total_e2e}  ({100*rx_acc:.1f}%)")
    print(f"  Both exact       : {e2e_acc*total_e2e:.0f}/{total_e2e}  ({100*e2e_acc:.1f}%)")

    # ── Export ────────────────────────────────────────────────────────────────
    dummy = torch.zeros(1, 1, 18, 12, device=device)
    model.eval()

    ts_path = MODELS_DIR / "digit_cnn.torchscript.pt"
    traced  = torch.jit.trace(model, dummy)
    traced.save(str(ts_path))
    print(f"\nTorchScript → {ts_path.relative_to(ROOT)}")

    onnx_path = MODELS_DIR / "digit_cnn.onnx"
    try:
        # opset_version=11 → IR version 6, supported by all onnxruntime builds
        # including old RPi packages.  dynamo=False forces the legacy
        # TorchScript-based exporter which produces a single self-contained file.
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["image"], output_names=["logits"],
            opset_version=11,
            dynamo=False,
        )
        print(f"ONNX        → {onnx_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"ONNX export failed: {e}")

    print(f"PyTorch weights → {best_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
