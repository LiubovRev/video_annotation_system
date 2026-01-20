# 1️⃣ Make sure venv is active
source ~/Bureau/new/.venv_build/bin/activate

# 2️⃣ Upgrade pip/setuptools/wheel
pip install --upgrade pip setuptools wheel

# 3️⃣ Install PyTorch + torchvision + torchaudio for CUDA 12
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu12

# 4️⃣ Install psifx from PyPI
pip install psifx

# 5️⃣ Optional: check versions and CUDA availability
python -c "import torch, torchvision; print('torch:', torch.__version__, 'torchvision:', torchvision.__version__, 'CUDA:', torch.version.cuda)"
python -c "from psifx.video.tracking.samurai.tool import SamuraiTrackingTool; print('psifx import OK')"
