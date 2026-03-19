#!/usr/bin/env python3
"""
准备训练数据：从 mask 目录中选择 10000 张图片用于训练
并进行二值化处理，确保只有纯黑白像素
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

def binarize_image(image, threshold=128):
    """
    将图像二值化为纯黑白

    Args:
        image: PIL Image 对象
        threshold: 二值化阈值 (0-255)

    Returns:
        二值化后的 PIL Image
    """
    # 转换为灰度图
    gray = image.convert('L')
    # 转换为 numpy 数组
    img_array = np.array(gray)
    # 二值化：大于阈值的设为255(白)，小于等于阈值的设为0(黑)
    binary_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
    # 转回 PIL Image
    return Image.fromarray(binary_array, mode='L')

def prepare_training_data(source_dir, output_dir, num_samples=10000, binarize=True, threshold=128):
    """
    从源目录中随机选择指定数量的图片，二值化后保存到输出目录

    Args:
        source_dir: 源图片目录
        output_dir: 输出目录
        num_samples: 要选择的图片数量
        binarize: 是否进行二值化处理
        threshold: 二值化阈值 (0-255)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件（包括 tif 格式）
    image_files = (list(source_path.glob("*.png")) +
                   list(source_path.glob("*.jpg")) +
                   list(source_path.glob("*.tif")) +
                   list(source_path.glob("*.tiff")))

    print(f"找到 {len(image_files)} 张图片")

    if len(image_files) < num_samples:
        print(f"警告：只有 {len(image_files)} 张图片，少于请求的 {num_samples} 张")
        num_samples = len(image_files)

    # 随机选择图片
    selected_files = random.sample(image_files, num_samples)

    print(f"随机选择 {num_samples} 张图片...")
    if binarize:
        print(f"⚙️  二值化设置: 阈值={threshold}, 输出纯黑白图像")

    # 处理并保存文件
    gray_count = 0
    binary_count = 0

    for img_file in tqdm(selected_files, desc="处理图片"):
        try:
            # 读取图像
            img = Image.open(img_file)

            # 检查原始图像是否包含灰度值
            gray_img = img.convert('L')
            unique_values = np.unique(np.array(gray_img))
            if not set(unique_values.tolist()).issubset({0, 255}):
                gray_count += 1
            else:
                binary_count += 1

            # 二值化处理
            if binarize:
                img = binarize_image(img, threshold)
            else:
                img = gray_img

            # 保存为 PNG 格式（无损压缩）
            output_file = output_path / f"{img_file.stem}.png"
            img.save(output_file)

        except Exception as e:
            print(f"\n❌ 处理失败 {img_file.name}: {e}")

    print(f"\n✅ 完成！已处理 {num_samples} 张图片到 {output_dir}")
    print(f"📊 原始数据统计:")
    print(f"   - 纯黑白图像: {binary_count}")
    print(f"   - 包含灰度值: {gray_count}")
    if binarize:
        print(f"   - 所有图像已二值化为纯黑白 (0 或 255)")

    # 验证输出
    print(f"\n🔍 验证输出数据...")
    output_files = list(output_path.glob("*.png"))[:10]
    all_binary = True
    for out_file in output_files:
        img = Image.open(out_file).convert('L')
        unique_vals = set(np.unique(np.array(img)).tolist())
        if unique_vals != {0, 255}:
            all_binary = False
            print(f"   ⚠️  {out_file.name} 包含灰度值: {unique_vals}")

    if all_binary:
        print(f"   ✅ 验证通过！所有输出图像都是纯黑白的")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='准备训练数据（自动二值化为纯黑白）')
    parser.add_argument('--source_dir', type=str, required=True, help='源图片目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10000, help='选择的图片数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no_binarize', action='store_true', help='禁用二值化（保留原始灰度）')
    parser.add_argument('--threshold', type=int, default=128, help='二值化阈值 (0-255，默认128)')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    prepare_training_data(
        args.source_dir,
        args.output_dir,
        args.num_samples,
        binarize=not args.no_binarize,
        threshold=args.threshold
    )
