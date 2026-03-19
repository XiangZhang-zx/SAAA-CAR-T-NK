import os
import glob
import re
from torch_fidelity import calculate_metrics

def calculate_kid(real_dir, gen_dir, num_workers=4, batch_size=50, verbose=True):
    """
    使用torch-fidelity库计算KID (Kernel Inception Distance)
    参数:
    real_dir (str): 包含真实图像的目录路径
    gen_dir (str): 包含生成图像的目录路径
    num_workers (int): 数据加载的工作线程数
    batch_size (int): 批处理大小
    verbose (bool): 是否打印详细信息
    返回:
    float: KID分数
    """
    if verbose:
        print(f"计算KID: {real_dir} vs {gen_dir}")
    # 计算KID和其他指标
    metrics = calculate_metrics(
        input1=real_dir,
        input2=gen_dir,
        cuda=True,
        kid=True,  # 计算KID
        fid=False,  # 不计算FID
        verbose=verbose,
        batch_size=batch_size,
        num_workers=num_workers
    )
    # 返回KID分数
    return metrics['kid'] if 'kid' in metrics else None

def is_kaggle_image(filename):
    """
    判断文件名是否属于Kaggle数据集格式（6位数字前缀）
    例如：000017_fake_B.png
    """
    basename = os.path.basename(filename)
    # 简单直接地检查是否以6位数字开头
    return (len(basename) >= 6 and 
            basename[:6].isdigit() and 
            ('_fake_B.' in basename or '_real_B.' in basename or '_real_A.' in basename))

def is_cart_image(filename):
    """
    判断文件名是否属于Cart数据集格式（文字前缀）
    例如：AF647-Streptavidin-Protein A AF488-Perforin AF568-P-Zeta AF405-F-actin-3_094_fake_B.png
    """
    # 非数字开头且包含_fake_B或_real_B的文件
    basename = os.path.basename(filename)
    return (not basename[0].isdigit() and 
            ('_fake_B.' in basename or '_real_B.' in basename))

def add_kid_to_evaluation_by_dataset(results_dir, metrics_dict=None):
    """
    分别计算Kaggle和Cart数据集的KID
    参数:
    results_dir (str): 结果目录，包含两种不同格式的图像
    metrics_dict (dict, optional): 现有的指标字典，可以将KID添加到其中
    返回:
    dict: 包含两个数据集KID的指标字典
    """
    # 创建临时目录
    temp_kaggle_real_dir = os.path.join(results_dir, "temp_kaggle_real")
    temp_kaggle_fake_dir = os.path.join(results_dir, "temp_kaggle_fake")
    temp_cart_real_dir = os.path.join(results_dir, "temp_cart_real")
    temp_cart_fake_dir = os.path.join(results_dir, "temp_cart_fake")
    
    all_temp_dirs = [
        temp_kaggle_real_dir, temp_kaggle_fake_dir,
        temp_cart_real_dir, temp_cart_fake_dir
    ]
    
    for dir_path in all_temp_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    if metrics_dict is None:
        metrics_dict = {}
    
    try:
        # 获取所有图像
        all_images = glob.glob(os.path.join(results_dir, "*.*"))
        
        # 分类图像
        kaggle_real_images = [img for img in all_images if is_kaggle_image(img) and "_real_B." in img]
        kaggle_fake_images = [img for img in all_images if is_kaggle_image(img) and "_fake_B." in img]
        cart_real_images = [img for img in all_images if is_cart_image(img) and "_real_B." in img]
        cart_fake_images = [img for img in all_images if is_cart_image(img) and "_fake_B." in img]
        
        # 创建符号链接
        for images, temp_dir in [
            (kaggle_real_images, temp_kaggle_real_dir),
            (kaggle_fake_images, temp_kaggle_fake_dir),
            (cart_real_images, temp_cart_real_dir),
            (cart_fake_images, temp_cart_fake_dir)
        ]:
            for img_path in images:
                basename = os.path.basename(img_path)
                os.symlink(os.path.abspath(img_path), os.path.join(temp_dir, basename))
        
        # 检查是否有足够的图像来计算KID
        kaggle_has_enough = len(kaggle_real_images) > 0 and len(kaggle_fake_images) > 0
        cart_has_enough = len(cart_real_images) > 0 and len(cart_fake_images) > 0
        
        # 计算Kaggle数据集的KID
        if kaggle_has_enough:
            print(f"发现 {len(kaggle_real_images)} 个Kaggle真实图像和 {len(kaggle_fake_images)} 个Kaggle生成图像")
            kid_kaggle = calculate_kid(temp_kaggle_real_dir, temp_kaggle_fake_dir)
            metrics_dict['KID_Kaggle'] = kid_kaggle
            print(f"Kaggle数据集KID分数: {kid_kaggle}")
        else:
            print("没有足够的Kaggle图像来计算KID")
        
        # 计算Cart数据集的KID
        if cart_has_enough:
            print(f"发现 {len(cart_real_images)} 个Cart真实图像和 {len(cart_fake_images)} 个Cart生成图像")
            kid_cart = calculate_kid(temp_cart_real_dir, temp_cart_fake_dir)
            metrics_dict['KID_Cart'] = kid_cart
            print(f"Cart数据集KID分数: {kid_cart}")
        else:
            print("没有足够的Cart图像来计算KID")
        
        return metrics_dict
    
    finally:
        # 清理临时目录
        for temp_dir in all_temp_dirs:
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    os.unlink(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)

# 使用示例
if __name__ == "__main__":
    # 示例使用
    RESULTS_DIR = "/research/projects/trans_llm/Xiang_Zhang/data/ControlNet/training/cell_dataset/pytorch-CycleGAN-and-pix2pix/results/cell_pix2pix/test_latest/images"
    
    # 计算两个数据集的KID
    metrics = add_kid_to_evaluation_by_dataset(RESULTS_DIR)
    
    # 将指标保存到文件
    if metrics:
        with open(os.path.join(os.path.dirname(RESULTS_DIR), "metrics_by_dataset.txt"), "w") as f:
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name}: {metric_value}\n")