#!/usr/bin/env python3
"""
使用训练好的模型生成 mask 图像
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from diffusers import DDPMPipeline
from tqdm import tqdm

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

def generate_masks(model_path, output_dir, num_images=100, batch_size=4, num_inference_steps=50, binarize=True, threshold=128, base_seed=0):
    """
    生成 mask 图像

    Args:
        model_path: 训练好的模型路径
        output_dir: 输出目录
        num_images: 要生成的图像数量
        batch_size: 批次大小
        num_inference_steps: 推理步数
        binarize: 是否二值化为纯黑白
        threshold: 二值化阈值 (0-255)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"加载模型: {model_path}")
    pipeline = DDPMPipeline.from_pretrained(model_path)
    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"开始生成 {num_images} 张图像...")
    
    num_batches = (num_images + batch_size - 1) // batch_size
    image_count = 0
    
    for batch_idx in tqdm(range(num_batches), desc="生成图像"):
        # 计算当前批次大小
        current_batch_size = min(batch_size, num_images - image_count)
        
        # 生成图像
        generator = torch.Generator(device=pipeline.device).manual_seed(base_seed * 10000 + batch_idx)
        images = pipeline(
            generator=generator,
            batch_size=current_batch_size,
            num_inference_steps=num_inference_steps,
            output_type="pil",
        ).images
        
        # 保存图像
        for i, image in enumerate(images):
            # 二值化处理（如果启用）
            if binarize:
                image = binarize_image(image, threshold)

            image_path = os.path.join(output_dir, f"generated_mask_{image_count:05d}.png")
            image.save(image_path)
            image_count += 1
    
    print(f"\n完成！已生成 {image_count} 张图像到 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成纯黑白 mask 图像')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--num_images', type=int, default=100, help='生成的图像数量')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='推理步数')
    parser.add_argument('--no_binarize', action='store_true', help='禁用二值化（保留灰度）')
    parser.add_argument('--threshold', type=int, default=128, help='二值化阈值 (0-255，默认128)')
    parser.add_argument('--base_seed', type=int, default=0, help='基础随机种子（用于 multi-run FID/KID）')

    args = parser.parse_args()

    generate_masks(
        args.model_path,
        args.output_dir,
        args.num_images,
        args.batch_size,
        args.num_inference_steps,
        binarize=not args.no_binarize,
        threshold=args.threshold,
        base_seed=args.base_seed,
    )
