#!/usr/bin/env python
"""
生成 500,000 个单细胞图像
使用已训练好的 innov_cell_pixpix 模型
"""

import os
import sys
import runpy

# 设置工作目录
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pix2pix'))

# 输出目录
output_dir = "./result_A/innov_cell_pixpix/test_latest/images"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("🚀 开始生成 500,000 个单细胞图像")
print("=" * 80)
print(f"📁 输出目录: {output_dir}")
print(f"🎯 目标数量: 500,000")
print(f"📦 模型: innov_cell_pixpix (latest)")
print(f"🔧 Batch Size: 1024 (4 GPU)")
print("=" * 80)

# 使用训练集作为数据源（包含所有增强的mask）
train_dataroot = "./datasets/cell_dataset/train"

# 检查训练集是否存在
if not os.path.exists(train_dataroot):
    print(f"❌ 错误：训练集目录不存在: {train_dataroot}")
    sys.exit(1)

# 统计训练集文件数量
train_files = [f for f in os.listdir(train_dataroot) if f.endswith('.jpg') or f.endswith('.png')]
num_train_files = len(train_files)
print(f"📊 训练集文件数: {num_train_files}")

# 计算需要生成的数量
num_to_generate = 500000

# 构建测试命令
test_cmd = [
    "python", "test.py",
    "--dataroot", train_dataroot,
    "--name", "cell_pix2pix_v200",  # 使用 v200 模型
    "--model", "test",
    "--netG", "unet_256",  # 标准 UNet 架构
    "--norm", "batch",  # 使用 BatchNorm（与训练时一致）
    "--direction", "AtoB",
    "--num_test", str(num_to_generate),
    "--results_dir", output_dir,
    "--batch_size", "256",
    "--gpu_ids", "0,1,2,3",
    "--no_dropout"
]

print("\n🔥 开始生成...")
print("=" * 80)
sys.stdout.flush()

# 设置命令行参数
sys.argv = ['test.py'] + test_cmd[2:]

# 禁用输出缓冲
class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def flush(self):
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

# 运行 test.py
try:
    runpy.run_path('test.py', run_name='__main__')
    print("\n" + "=" * 80)
    print("✅ 生成完成！")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 80)
except Exception as e:
    print(f"\n❌ 生成出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

