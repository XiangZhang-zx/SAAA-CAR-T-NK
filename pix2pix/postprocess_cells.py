#!/usr/bin/env python3
"""
后处理脚本：将 mask 调整到生成的细胞实际范围
只修改 mask，不修改细胞图像
确保 mask 和生成的细胞图像完全匹配
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

def extract_mask_from_combined(combined_img):
    """从组合图像中提取 mask（左半部分）"""
    h, w = combined_img.shape[:2]
    mask = combined_img[:, :w//2]
    return mask

def extract_cell_from_combined(combined_img):
    """从组合图像中提取生成的细胞图像（右半部分）"""
    h, w = combined_img.shape[:2]
    cell = combined_img[:, w//2:]
    return cell

def get_mask_region(mask):
    """获取 mask 的有效区域（非黑色部分）"""
    # 转换为灰度图
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask
    
    # 二值化：找到非黑色区域
    _, binary = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
    
    return binary

def match_mask_to_cell(mask, cell_img, margin=0):
    """
    将 mask 调整到完全匹配生成的细胞实际范围
    - 如果 mask 比细胞大，缩小 mask
    - 如果 mask 比细胞小，扩大 mask

    Parameters:
        mask: 原始 mask 图像 (RGB)
        cell_img: 生成的细胞图像 (RGB)
        margin: 边距调整（正数=扩大mask，负数=缩小mask，单位：像素）

    Returns:
        处理后的 mask 图像
    """
    # 获取细胞图像的非黑色区域
    cell_gray = cv2.cvtColor(cell_img, cv2.COLOR_RGB2GRAY)
    _, cell_binary = cv2.threshold(cell_gray, 10, 255, cv2.THRESH_BINARY)

    # 获取原始 mask 的区域
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    _, mask_binary = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)

    # 使用细胞的实际范围作为新的 mask 形状
    new_mask_binary = cell_binary.copy()

    # 如果需要调整边距
    if margin != 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if margin > 0:
            # 扩大 mask
            new_mask_binary = cv2.dilate(new_mask_binary, kernel, iterations=margin)
        else:
            # 缩小 mask
            new_mask_binary = cv2.erode(new_mask_binary, kernel, iterations=abs(margin))

    # 使用原始 mask 的颜色信息，但应用新的形状
    # 方法：在新的 mask 区域内，使用 inpainting 填充颜色
    new_mask = np.zeros_like(mask)

    # 找到新旧 mask 的交集区域（可以直接复制的部分）
    overlap = cv2.bitwise_and(new_mask_binary, mask_binary)
    new_mask[overlap > 0] = mask[overlap > 0]

    # 找到需要填充的区域（新 mask 有但旧 mask 没有的部分）
    need_fill = cv2.bitwise_and(new_mask_binary, cv2.bitwise_not(mask_binary))

    if np.any(need_fill > 0):
        # 使用 inpainting 填充新区域
        # 创建一个临时图像，包含已有的颜色信息
        temp_mask = mask.copy()
        temp_mask[new_mask_binary == 0] = 0  # 清除不需要的区域

        # 使用 inpainting 填充缺失区域
        new_mask = cv2.inpaint(temp_mask, need_fill, 3, cv2.INPAINT_TELEA)

        # 确保只保留新 mask 区域
        new_mask[new_mask_binary == 0] = 0

    return new_mask

def ensure_edge_clearance(cell_img, mask_binary, border_size=2):
    """
    确保细胞边缘不触碰图像边界
    
    Parameters:
        cell_img: 细胞图像
        mask_binary: mask 二值图
        border_size: 边界清除大小（像素）
    
    Returns:
        处理后的细胞图像
    """
    h, w = cell_img.shape[:2]
    
    # 创建边界 mask（图像边缘区域）
    border_mask = np.zeros((h, w), dtype=np.uint8)
    border_mask[:border_size, :] = 255  # 上边界
    border_mask[-border_size:, :] = 255  # 下边界
    border_mask[:, :border_size] = 255  # 左边界
    border_mask[:, -border_size:] = 255  # 右边界
    
    # 找到触碰边界的细胞区域
    cell_gray = cv2.cvtColor(cell_img, cv2.COLOR_RGB2GRAY)
    _, cell_binary = cv2.threshold(cell_gray, 10, 255, cv2.THRESH_BINARY)
    
    # 触碰边界的区域
    touching_border = cv2.bitwise_and(cell_binary, border_mask)
    
    # 如果有触碰边界的区域，进行腐蚀处理
    if np.any(touching_border > 0):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_size*2+1, border_size*2+1))
        eroded_binary = cv2.erode(cell_binary, kernel, iterations=1)
        
        # 确保腐蚀后仍在 mask 范围内
        eroded_binary = cv2.bitwise_and(eroded_binary, mask_binary)
        
        # 创建新的细胞图像（只保留腐蚀后的区域）
        result = cell_img.copy()
        result[eroded_binary == 0] = 0
    else:
        result = cell_img.copy()
    
    return result

def postprocess_single_image(input_path, output_path, margin=0):
    """
    后处理单张图像

    Parameters:
        input_path: 输入图像路径（组合图像：mask | cell）
        output_path: 输出图像路径
        margin: mask边距调整（正数=扩大，负数=缩小）
    """
    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        print(f"警告：无法读取图像 {input_path}")
        return False

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 提取 mask 和 cell
    mask = extract_mask_from_combined(img)
    cell = extract_cell_from_combined(img)

    # 只调整 mask 到细胞实际范围，不修改细胞图像
    mask_matched = match_mask_to_cell(mask, cell, margin=margin)

    # 重新组合图像
    w = img.shape[1]
    result = np.zeros_like(img)
    result[:, :w//2] = mask_matched
    result[:, w//2:] = cell  # 保持细胞图像不变

    # 保存结果
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)

    return True

def postprocess_separate_images(mask_path, cell_path, output_mask_path, output_cell_path, margin=0):
    """
    后处理分开保存的 mask 和 cell 图像

    Parameters:
        mask_path: mask 图像路径 (real_A)
        cell_path: 生成的细胞图像路径 (fake_B)
        output_mask_path: 输出 mask 图像路径
        output_cell_path: 输出细胞图像路径
        margin: mask边距调整（正数=扩大，负数=缩小）
    """
    # 读取图像
    mask = cv2.imread(mask_path)
    cell = cv2.imread(cell_path)

    if mask is None or cell is None:
        print(f"警告：无法读取图像 {mask_path} 或 {cell_path}")
        return False

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    cell = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)

    # 只调整 mask 到细胞实际范围，不修改细胞图像
    mask_matched = match_mask_to_cell(mask, cell, margin=margin)

    # 保存结果（保存处理后的 mask 和原始细胞图像）
    mask_bgr = cv2.cvtColor(mask_matched, cv2.COLOR_RGB2BGR)
    cell_bgr = cv2.cvtColor(cell, cv2.COLOR_RGB2BGR)  # 保持细胞图像不变
    cv2.imwrite(output_mask_path, mask_bgr)
    cv2.imwrite(output_cell_path, cell_bgr)

    return True

def postprocess_directory(input_dir, output_dir=None, margin=0):
    """
    后处理整个目录的图像

    Parameters:
        input_dir: 输入目录
        output_dir: 输出目录（如果为 None，则覆盖原文件）
        margin: mask边距调整（正数=扩大，负数=缩小）
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    # 获取所有 fake_B 图像文件（生成的细胞图像）
    fake_b_files = [f for f in os.listdir(input_dir)
                    if f.endswith('_fake_B.png')]

    print(f"找到 {len(fake_b_files)} 张生成的细胞图像")
    print(f"Mask 边距调整: {margin} 像素 ({'扩大' if margin > 0 else '缩小' if margin < 0 else '精确匹配'})")
    print(f"注意：只修改 mask，不修改细胞图像")

    success_count = 0
    for fake_b_file in tqdm(fake_b_files, desc="后处理图像"):
        # 构建对应的 real_A 文件名
        base_name = fake_b_file.replace('_fake_B.png', '')
        real_a_file = base_name + '_real_A.png'

        mask_path = os.path.join(input_dir, real_a_file)
        cell_path = os.path.join(input_dir, fake_b_file)
        output_mask_path = os.path.join(output_dir, real_a_file)
        output_cell_path = os.path.join(output_dir, fake_b_file)

        # 检查 mask 文件是否存在
        if not os.path.exists(mask_path):
            print(f"警告：找不到对应的 mask 文件: {real_a_file}")
            continue

        if postprocess_separate_images(mask_path, cell_path, output_mask_path, output_cell_path, margin):
            success_count += 1

    print(f"\n完成！成功处理 {success_count}/{len(fake_b_files)} 张图像")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='后处理生成的细胞图像：调整mask完全匹配细胞实际范围（只修改mask，不修改细胞）')
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（默认覆盖原文件）')
    parser.add_argument('--margin', type=int, default=0, help='Mask边距调整（正数=扩大，负数=缩小，0=精确匹配，默认0）')

    args = parser.parse_args()

    postprocess_directory(
        args.input_dir,
        args.output_dir,
        args.margin
    )

