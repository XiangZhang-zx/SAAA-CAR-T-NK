#!/usr/bin/env python3
"""
生成纯噪声图像（用于展示 diffusion 模型的起点）
"""

import torch
import numpy as np
from PIL import Image
import argparse

def generate_pure_noise(size=256, seed=None):
    """
    生成纯高斯噪声图像
    
    Args:
        size: 图像尺寸
        seed: 随机种子
    
    Returns:
        PIL Image
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # 生成标准高斯噪声 (mean=0, std=1)
    noise = torch.randn(3, size, size)
    
    # 归一化到 [0, 255] 范围
    # 将 [-3, 3] 的范围映射到 [0, 255]
    noise = torch.clamp(noise, -3, 3)  # 裁剪极端值
    noise = (noise + 3) / 6 * 255  # 映射到 [0, 255]
    
    # 转换为 numpy 数组
    noise_np = noise.permute(1, 2, 0).numpy().astype(np.uint8)
    
    # 转换为 PIL Image
    image = Image.fromarray(noise_np, mode='RGB')
    
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成纯噪声图像')
    parser.add_argument('--output', type=str, default='pure_noise.png', help='输出文件名')
    parser.add_argument('--size', type=int, default=256, help='图像尺寸')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--num_images', type=int, default=1, help='生成图像数量')
    
    args = parser.parse_args()
    
    if args.num_images == 1:
        # 生成单张图像
        noise_image = generate_pure_noise(args.size, args.seed)
        noise_image.save(args.output)
        print(f"✅ 已生成纯噪声图像: {args.output}")
    else:
        # 生成多张图像
        for i in range(args.num_images):
            noise_image = generate_pure_noise(args.size, args.seed + i)
            output_name = args.output.replace('.png', f'_{i}.png')
            noise_image.save(output_name)
            print(f"✅ 已生成: {output_name}")
        print(f"✅ 共生成 {args.num_images} 张纯噪声图像")

