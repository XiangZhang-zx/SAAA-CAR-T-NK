#!/usr/bin/env python3
"""
将已生成的 sample 图像二值化为纯黑白
"""

import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

def binarize_image(image_path, output_path, threshold=128):
    """
    将图像二值化为纯黑白
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        threshold: 二值化阈值 (0-255)
    """
    # 读取图像
    image = Image.open(image_path)
    
    # 转换为灰度图
    gray = image.convert('L')
    
    # 转换为 numpy 数组
    img_array = np.array(gray)
    
    # 二值化：大于阈值的设为255(白)，小于等于阈值的设为0(黑)
    binary_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
    
    # 转回 PIL Image
    binary_image = Image.fromarray(binary_array, mode='L')
    
    # 保存
    binary_image.save(output_path)

def binarize_directory(input_dir, output_dir=None, threshold=128, recursive=True):
    """
    批量二值化目录中的所有图像
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录（如果为None，则覆盖原文件）
        threshold: 二值化阈值
        recursive: 是否递归处理子目录
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_path = input_path
        print(f"⚠️  警告：将覆盖原文件！")
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"输出目录: {output_path}")
    
    # 查找所有图像文件
    if recursive:
        image_files = list(input_path.rglob("*.png")) + list(input_path.rglob("*.jpg"))
    else:
        image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像
    for img_file in tqdm(image_files, desc="二值化图像"):
        # 计算相对路径
        rel_path = img_file.relative_to(input_path)
        
        # 输出路径
        if output_dir is None:
            out_file = img_file
        else:
            out_file = output_path / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 二值化
        try:
            binarize_image(str(img_file), str(out_file), threshold)
        except Exception as e:
            print(f"❌ 处理失败 {img_file}: {e}")
    
    print(f"✅ 完成！已处理 {len(image_files)} 个图像")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将图像二值化为纯黑白')
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（默认覆盖原文件）')
    parser.add_argument('--threshold', type=int, default=128, help='二值化阈值 (0-255，默认128)')
    parser.add_argument('--no_recursive', action='store_true', help='不递归处理子目录')
    
    args = parser.parse_args()
    
    binarize_directory(
        args.input_dir,
        args.output_dir,
        args.threshold,
        recursive=not args.no_recursive
    )

