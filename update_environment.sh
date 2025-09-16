#!/bin/bash

echo "正在更新 htfllib 环境的 PyTorch 版本..."

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate htfllib

# 升级 PyTorch 到兼容版本
echo "升级 PyTorch 到 2.1.0 或更高版本..."
# 首先卸载冲突的包
pip uninstall -y torchtext torchdata
# 安装兼容的 PyTorch 套件
pip install --upgrade torch>=2.1.0 torchaudio torchvision
# 安装兼容的 torchtext 和 torchdata 版本
pip install torchtext torchdata

# 验证版本
echo "验证 PyTorch 版本..."
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')"

# 测试 diffusers 是否正常工作
echo "测试 diffusers 库..."
python -c "
try:
    from diffusers import StableDiffusionPipeline
    print('✅ diffusers 库可以正常导入')
except Exception as e:
    print(f'❌ diffusers 导入失败: {e}')
"

echo "环境更新完成！"
