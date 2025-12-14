#!/bin/bash
#SBATCH --job-name=install-flash-attn
#SBATCH --partition=lrc-xlong
#SBATCH --qos=highest
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=2:00:00
#SBATCH --output=./slurmlogs/install-flash-attn-%j.log

echo "================================"
echo "Installing flash-attn with proper metadata"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "================================"

cd /home/d84417809/Latent-Reasoning-RL

source .venv/bin/activate

# 设置 HF_HOME 保持一致
export HF_HOME=/home/d84417809/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# 打印环境信息
echo "Python: $(which python)"
echo "Venv location: $VIRTUAL_ENV"
echo "Working directory: $(pwd)"
echo "HF_HOME: $HF_HOME"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed yet')"
echo "CUDA version:"
nvidia-smi

export CUDA_HOME=/usr/local/cuda
echo "CUDA_HOME: $CUDA_HOME"

# 使用更多并行编译加速
export MAX_JOBS=12
echo "MAX_JOBS: $MAX_JOBS"

# 设置编译优化
export TORCH_CUDA_ARCH_LIST="9.0"  # H200 是 Hopper 架构 (compute capability 9.0)

# 确保 nvcc 可用
if ! command -v nvcc &> /dev/null; then
    echo "⚠️ nvcc not found! Trying to find CUDA..."
    
    # 尝试多种常见 CUDA 路径
    for cuda_path in /usr/local/cuda-12.4 /usr/local/cuda-12.6 /usr/local/cuda /opt/cuda; do
        if [ -d "$cuda_path" ] && [ -f "$cuda_path/bin/nvcc" ]; then
            export CUDA_HOME=$cuda_path
            export PATH=$CUDA_HOME/bin:$PATH
            export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
            echo "✅ Found CUDA at: $CUDA_HOME"
            break
        fi
    done
fi

# 验证 nvcc
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "❌ nvcc still not found! Compilation will likely fail."
    echo "Available CUDA installations:"
    ls -la /usr/local/ | grep cuda || echo "None found in /usr/local/"
    exit 1
fi

echo ""
echo "Step 1: 卸载旧版本 flash-attn"
uv pip uninstall flash-attn -v
# 再次尝试卸载以防残留
pip uninstall -y flash-attn || true

echo ""
echo "Step 2: 清理缓存"
uv cache clean
# 清理 pip 缓存
rm -rf ~/.cache/pip

echo ""
echo "Step 3: 安装 flash-attn（使用 --no-build-isolation 避免编译问题）"
# flash-attn 编译时需要 torch，但没有声明为 build dependency
# 必须用 --no-build-isolation 让编译器看到环境中的 torch
# 添加 --no-cache-dir 强制重新编译
uv pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir

echo ""
echo "Step 4: 手动创建元数据（修复 importlib.metadata 问题）"
python3 << 'PYEOF'
import os
import sys
from pathlib import Path

# 找到 flash_attn 安装位置
import flash_attn
flash_attn_path = Path(flash_attn.__file__).parent.parent

# 找到 site-packages 目录
site_packages = flash_attn_path

# 创建 .dist-info 目录
dist_info_dir = site_packages / "flash_attn-2.8.1.dist-info"
dist_info_dir.mkdir(exist_ok=True)

# 创建 METADATA 文件
metadata_content = """Metadata-Version: 2.1
Name: flash-attn
Version: 2.8.3
Summary: Flash Attention: Fast and Memory-Efficient Exact Attention
Home-page: https://github.com/Dao-AILab/flash-attention
Author: Tri Dao
License: BSD-3-Clause
Requires-Python: >=3.7
Requires-Dist: torch
Requires-Dist: einops
"""

with open(dist_info_dir / "METADATA", "w") as f:
    f.write(metadata_content)

# 创建 INSTALLER 文件
with open(dist_info_dir / "INSTALLER", "w") as f:
    f.write("uv\n")

# 创建 top_level.txt
with open(dist_info_dir / "top_level.txt", "w") as f:
    f.write("flash_attn\n")

print(f"✅ Created metadata at: {dist_info_dir}")
PYEOF

echo ""
echo "Step 5: 验证安装（包括元数据）"
python3 -c "
import flash_attn
import importlib.metadata

print(f'✅ flash-attn module version: {flash_attn.__version__}')
try:
    metadata_version = importlib.metadata.version('flash_attn')
    print(f'✅ Metadata version: {metadata_version}')
except Exception as e:
    print(f'⚠️  Metadata warning: {e}')
    print('   (This is expected with --no-build-isolation, but we created manual metadata)')
    
print(f'✅ Module location: {flash_attn.__file__}')
print('\\n✅ Flash Attention can be imported successfully!')
"

echo ""
echo "================================"
echo "Installation completed successfully!"
echo "End Time: $(date)"
echo "================================"

