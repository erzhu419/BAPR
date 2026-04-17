#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  BAPR Server Setup — create conda env and install dependencies
#
#  Usage:
#    bash setup_env.sh              # creates/uses conda env 'bapr-jax'
#    ENV_NAME=mybapr bash setup_env.sh
# ═══════════════════════════════════════════════════════════════════════════
set -e

ENV_NAME=${ENV_NAME:-bapr-jax}
PY_VERSION=${PY_VERSION:-3.11}

echo "═══════════════════════════════════════════════════════════════"
echo "  BAPR environment setup → $ENV_NAME (python=$PY_VERSION)"
echo "═══════════════════════════════════════════════════════════════"

# ── Detect conda ──────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Please install Miniconda/Anaconda first." >&2
    exit 1
fi

# Activate conda in this shell
eval "$(conda shell.bash hook)"

# ── Create env if missing ────────────────────────────────────────────
if conda info --envs | grep -qE "^${ENV_NAME}\s"; then
    echo "Env '$ENV_NAME' exists, using it."
else
    echo "Creating env '$ENV_NAME' with python=$PY_VERSION..."
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi

conda activate "$ENV_NAME"

# ── Detect CUDA version ──────────────────────────────────────────────
CUDA_TAG=""
if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    if [ -n "$CUDA_VERSION" ]; then
        CUDA_MAJOR=${CUDA_VERSION%.*}
        if [ "$CUDA_MAJOR" -ge 12 ]; then
            CUDA_TAG="[cuda12]"
        elif [ "$CUDA_MAJOR" -eq 11 ]; then
            CUDA_TAG="[cuda11_pip]"
        fi
        echo "Detected CUDA $CUDA_VERSION → jax$CUDA_TAG"
    fi
fi

# ── Install packages ─────────────────────────────────────────────────
echo "Installing JAX (+optax/flax/brax) ..."
pip install --upgrade pip

# JAX with matching CUDA
if [ -n "$CUDA_TAG" ]; then
    pip install -U "jax${CUDA_TAG}==0.9.2"
else
    pip install -U "jax==0.9.2" "jaxlib==0.9.2"
fi

pip install \
    "flax==0.12.6" \
    "optax==0.2.8" \
    "brax==0.14.2" \
    "numpy>=2.0" \
    "scipy>=1.10" \
    "matplotlib>=3.7" \
    "scikit-learn>=1.3"

# ── Verify ───────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Verification"
echo "═══════════════════════════════════════════════════════════════"
python -c "
import jax, jaxlib, flax, brax, optax, numpy, scipy, matplotlib, sklearn
print(f'jax      = {jax.__version__}')
print(f'jaxlib   = {jaxlib.__version__}')
print(f'flax     = {flax.__version__}')
print(f'brax     = {brax.__version__}')
print(f'optax    = {optax.__version__}')
print(f'numpy    = {numpy.__version__}')
print(f'scipy    = {scipy.__version__}')
print(f'sklearn  = {sklearn.__version__}')
print(f'devices  = {jax.devices()}')
"

echo ""
echo "✓ Environment ready! Activate with:"
echo "    conda activate $ENV_NAME"
echo ""
echo "Then run experiments:"
echo "    bash jax_experiments/run_server.sh"
