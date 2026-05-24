# Copyright (c) 2026 Lukas Silberbauer. All rights reserved.

"""Shared pytest fixtures for the rpi_pool_cam test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_DATA_DIR = Path("data")
_CONFIG_PATH = Path("config.yaml")


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Return the parsed config.yaml as a dictionary."""
    with _CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


def _collect_pairs() -> list[tuple[dict[str, Any] | None, Path, Path]]:
    """Collect all (ground_truth, yaml_path, jpg_path) tuples from data/."""
    pairs = []
    for yaml_path in sorted(_DATA_DIR.glob("*.yaml")):
        jpg_path = yaml_path.with_suffix(".jpg")
        if not jpg_path.is_file():
            continue
        with yaml_path.open() as fh:
            ground_truth = yaml.safe_load(fh)  # None when YAML is empty
        pairs.append((ground_truth, yaml_path, jpg_path))
    return pairs


_DATA_PAIRS = _collect_pairs()


@pytest.fixture(
    params=_DATA_PAIRS,
    ids=[p[1].stem for p in _DATA_PAIRS],
)
def image_yaml_pair(request: pytest.FixtureRequest) -> tuple[dict[str, Any] | None, Path]:
    """Parametrized fixture yielding (ground_truth_dict_or_None, jpg_path) for each data image."""
    ground_truth, _yaml_path, jpg_path = request.param
    return ground_truth, jpg_path
