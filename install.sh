#!/bin/bash
# LatentFrag installation script
# Creates a conda environment with all dependencies.
#
# Usage:
#   bash install.sh                  # defaults: env=latentfrag, python=3.11
#   bash install.sh myenv 3.12       # custom env name and python version
set -e

ENV_NAME="${1:-latentfrag}"
PYTHON_VERSION="${2:-3.11}"

echo "=== Installing LatentFrag ==="
echo "  Environment: $ENV_NAME"
echo "  Python:      $PYTHON_VERSION"
echo ""

# Create conda environment
conda create --name "$ENV_NAME" "python=$PYTHON_VERSION" -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# PyTorch + geometric
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.3.1
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Core scientific dependencies
pip install "numpy<2" "setuptools<81"
pip install pytorch-lightning==2.1.0 wandb biopython pdb-tools ProDy plyfile pyvtk
if python -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)" 2>/dev/null; then
    pip install rdkit
else
    pip install rdkit-pypi
fi
pip install pandas matplotlib jupyter
pip install posebusters fcd useful-rdkit-utils

# Fix C++ standard library compatibility
conda install -c conda-forge libstdcxx-ng -y
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/libstdcxx.sh" << 'ACTIVATE'
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ACTIVATE
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/libstdcxx.sh" << 'DEACTIVATE'
export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
unset _OLD_LD_LIBRARY_PATH
DEACTIVATE

# Re-activate to pick up the LD_LIBRARY_PATH
conda deactivate
conda activate "$ENV_NAME"

# Install LatentFrag itself
pip install -e .

echo ""
echo "=== Installation complete ==="
echo "Activate with:  conda activate $ENV_NAME"
echo "Verify with:    python test_setup.py"
