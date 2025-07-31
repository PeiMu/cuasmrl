#!/bin/bash
# CuASMRL 环境自动配置脚本（在 cuasmrl 根目录内运行）
set -e

echo "[INFO] 当前路径: $PWD"
if [[ ! -f "requirement.txt" ]]; then
    echo "[ERROR] 脚本必须在 cuasmrl 根目录下运行。"
    exit 1
fi

# 1. 加载预设环境变量（如果有）
source ./pre_setting.sh || true

# 2. 安装系统依赖
apt-get update
apt-get install -y clang lld ccache zlib1g-dev

# 3. 链接 CUDA 工具路径（conda 环境）
if [ ! -d "/usr/local/cuda/bin" ]; then
    mkdir -p /usr/local/cuda/bin
fi
if [ ! -L /usr/local/cuda/bin/python ]; then
    ln -s /opt/conda/envs/workflow/bin /usr/local/cuda/bin || true
fi

# 4. Python 虚拟环境
if [ ! -d ".venv" ]; then
    python -m venv .venv --prompt triton
fi
source .venv/bin/activate

# 5. 安装 Python 依赖
pip install -U pip
pip install ninja cmake wheel
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --no-build-isolation -r requirement.txt
pip install flash-attn==2.3.3
pip install pyelftools tensorboard
pip install nvidia-cutlass==3.5

# 6. 设置自定义 triton（如启用）
# pip uninstall triton -y
# pip install -e python --no-build-isolation
export PATH=$PWD/python/triton/third_party/cuda/bin:$PATH

# 7. 安装 CuAssembler 到本地子目录
if [ ! -d "CuAssembler" ]; then
    git clone https://github.com/hgl71964/CuAssembler.git CuAssembler
fi

export PATH=$PATH:$PWD/CuAssembler/bin
export PYTHONPATH=$PYTHONPATH:$PWD/CuAssembler:$PWD/CuAssembler/bin:$PWD/CuAssembler/CuAsm

cd CuAssembler/bin
if [ ! -f "cuasm" ]; then
    ln -s cuasm.py cuasm
    chmod a+x cuasm
fi
cd ../../

# 8. 安装 cuasmrl 为可编辑模式
pip install -e .

