#!/usr/bin/env python3
"""Export digit_cnn_best.pt to ONNX format for onnxruntime deployment."""

from __future__ import annotations

import pathlib
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"


class DigitCNN(nn.Module):
    """Tiny CNN for 12x18 grayscale LCD digit classification."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
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


def main() -> None:
    weights_path = MODELS_DIR / "digit_cnn_best.pt"
    onnx_path = MODELS_DIR / "digit_cnn.onnx"

    model = DigitCNN()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, 1, 18, 12)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,  # opset 12 is widely supported; avoids dynamo path
        dynamo=False,
    )
    print(f"Exported ONNX model -> {onnx_path.relative_to(ROOT)}  ({onnx_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
