#!/usr/bin/env bash
# One-time setup for Raspberry Pi (ARMv7, Raspbian Bullseye).
#
# onnxruntime has no official PyPI wheel for ARMv7. This script installs the
# community-built wheel from nknytk/built-onnxruntime-for-raspberrypi-linux
# before running pip install -r requirements.txt so that the requirements step
# sees onnxruntime as already satisfied and does not attempt a PyPI resolution.
#
# After this script completes, run the test suite to confirm everything works:
#   pytest tests/ -v

set -euo pipefail

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
OS_CODENAME=$(. /etc/os-release && echo "${VERSION_CODENAME}")
ARCH=$(uname -m)

echo "Detected: Python cp${PYTHON_VERSION}, OS ${OS_CODENAME}, arch ${ARCH}"

if [[ "${ARCH}" != "armv7l" ]]; then
    echo "Not an ARMv7 system — running plain pip install -r requirements.txt"
    pip install -r requirements.txt
    exit 0
fi

if [[ "${OS_CODENAME}" != "bullseye" ]]; then
    echo "ERROR: This script targets Raspbian Bullseye. Found: ${OS_CODENAME}" >&2
    echo "Adapt the wheel URL for your OS codename from:" >&2
    echo "  https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux/tree/master/wheels" >&2
    exit 1
fi

if ! [[ "${PYTHON_VERSION}" =~ ^3[89]$|^31[0-9]$ ]]; then
    echo "ERROR: No community onnxruntime wheel known for cp${PYTHON_VERSION} on ARMv7 Bullseye." >&2
    exit 1
fi

ONNX_VERSION="1.16.0"
WHEEL_BASE="https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux/raw/master/wheels"
WHEEL_URL="${WHEEL_BASE}/${OS_CODENAME}/onnxruntime-${ONNX_VERSION}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_${ARCH}.whl"

echo "Installing onnxruntime ${ONNX_VERSION} from community wheel..."
pip install "${WHEEL_URL}"

# onnxruntime 1.16.0 was built against NumPy 1.x; NumPy 2.x causes an import
# error at runtime. Pin it here so that subsequent pip installs do not upgrade.
echo "Pinning numpy<2 for onnxruntime 1.16.0 compatibility..."
pip install "numpy<2"

echo "Installing remaining dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete. Run 'pytest tests/ -v' to verify."
