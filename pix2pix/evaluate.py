import os
import glob
import re
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy import linalg
from tqdm import tqdm
import pytorch_fid.inception as inception
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2

# 辅助函数：检测文件所属的数据集
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
            ('_fake_B.' in basename or '_real_B.' in basename or '_real_A.' in basename))

class ImagePairDataset(Dataset):
    """用于加载真实图像和生成图像对的数据集"""
    def __init__(self, real_filenames, generated_filenames, transform=None):
        assert len(real_filenames) == len(generated_filenames), \
            f"真实图像数量 ({len(real_filenames)}) 与生成图像数量 ({len(generated_filenames)}) 不匹配"
        
        self.real_filenames = real_filenames
        self.generated_filenames = generated_filenames
        self.transform = transform
    
    def __len__(self):
        return len(self.real_filenames)
    
    def __getitem__(self, idx):
        real_img = Image.open(self.real_filenames[idx]).convert('RGB')
        gen_img = Image.open(self.generated_filenames[idx]).convert('RGB')
        
        if self.transform:
            real_img = self.transform(real_img)
            gen_img = self.transform(gen_img)
        
        return real_img, gen_img

class FIDCalculator:
    """计算FID分数的类"""
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 加载预训练的Inception v3模型
        self.inception_model = inception.InceptionV3().to(self.device)
        self.inception_model.eval()
        
        # 用于图像预处理的变换
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_activations(self, image_dir, batch_size=8):
        """计算目录中所有图像的Inception激活"""
        dataloader = DataLoader(
            ImageDataset(image_dir, self.transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )
        
        pred_arr = []
        for batch in tqdm(dataloader, desc=f"处理图像 ({image_dir})"):
            batch = batch.to(self.device)
            with torch.no_grad():
                pred = self.inception_model(batch)[0]
            
            # 如果需要，将特征展平
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
            
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr.append(pred)
        
        return np.concatenate(pred_arr, axis=0)
    
    def calculate_fid(self, real_dir, gen_dir, batch_size=8):
        """计算两个图像目录之间的FID分数"""
        print(f"计算FID: {real_dir} vs {gen_dir}")
        
        # 获取真实图像和生成图像的特征
        act_real = self._get_activations(real_dir, batch_size)
        act_gen = self._get_activations(gen_dir, batch_size)
        
        # 计算均值和协方差
        mu_real = np.mean(act_real, axis=0)
        sigma_real = np.cov(act_real, rowvar=False)
        
        mu_gen = np.mean(act_gen, axis=0)
        sigma_gen = np.cov(act_gen, rowvar=False)
        
        # 计算FID分数
        fid_value = self._calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        return fid_value
    
    def calculate_fid_from_paths(self, real_paths, fake_paths, batch_size=8):
        """从图像路径列表计算FID分数"""
        print(f"从 {len(real_paths)} 对图像计算FID")
        
        # 创建临时目录
        temp_real_dir = os.path.join(os.path.dirname(real_paths[0]), "temp_real_" + str(os.getpid()))
        temp_fake_dir = os.path.join(os.path.dirname(fake_paths[0]), "temp_fake_" + str(os.getpid()))
        
        os.makedirs(temp_real_dir, exist_ok=True)
        os.makedirs(temp_fake_dir, exist_ok=True)
        
        try:
            # 创建符号链接
            for i, (real_path, fake_path) in enumerate(zip(real_paths, fake_paths)):
                real_basename = os.path.basename(real_path)
                fake_basename = os.path.basename(fake_path)
                os.symlink(os.path.abspath(real_path), os.path.join(temp_real_dir, real_basename))
                os.symlink(os.path.abspath(fake_path), os.path.join(temp_fake_dir, fake_basename))
            
            # 计算FID
            return self.calculate_fid(temp_real_dir, temp_fake_dir, batch_size)
        
        finally:
            # 清理临时目录
            if os.path.exists(temp_real_dir):
                for file in os.listdir(temp_real_dir):
                    os.unlink(os.path.join(temp_real_dir, file))
                os.rmdir(temp_real_dir)
            
            if os.path.exists(temp_fake_dir):
                for file in os.listdir(temp_fake_dir):
                    os.unlink(os.path.join(temp_fake_dir, file))
                os.rmdir(temp_fake_dir)
    
    def calculate_fid_from_pairs(self, dataset, batch_size=8):
        """从成对的真实和生成图像计算FID分数"""
        print("从图像对计算FID")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )
        
        real_acts = []
        gen_acts = []
        
        for real_batch, gen_batch in tqdm(dataloader):
            real_batch = real_batch.to(self.device)
            gen_batch = gen_batch.to(self.device)
            
            with torch.no_grad():
                real_act = self.inception_model(real_batch)[0]
                gen_act = self.inception_model(gen_batch)[0]
            
            # 展平特征
            if real_act.size(2) != 1 or real_act.size(3) != 1:
                real_act = torch.nn.functional.adaptive_avg_pool2d(real_act, output_size=(1, 1))
                gen_act = torch.nn.functional.adaptive_avg_pool2d(gen_act, output_size=(1, 1))
            
            real_act = real_act.squeeze(3).squeeze(2).cpu().numpy()
            gen_act = gen_act.squeeze(3).squeeze(2).cpu().numpy()
            
            real_acts.append(real_act)
            gen_acts.append(gen_act)
        
        real_acts = np.concatenate(real_acts, axis=0)
        gen_acts = np.concatenate(gen_acts, axis=0)
        
        # 计算均值和协方差
        mu_real = np.mean(real_acts, axis=0)
        sigma_real = np.cov(real_acts, rowvar=False)
        
        mu_gen = np.mean(gen_acts, axis=0)
        sigma_gen = np.cov(gen_acts, rowvar=False)
        
        # 计算FID分数
        fid_value = self._calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        return fid_value
    
    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """计算Frechet距离"""
        # 计算两个均值向量之间的平方欧几里得距离
        diff = mu1 - mu2
        
        # 计算两个协方差矩阵的乘积的平方根
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 数值问题处理
        if not np.isfinite(covmean).all():
            msg = "fid计算产生了非限定值"
            print(msg)
            covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # 检查并处理虚部
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"虚部的值非常大，可能导致不正确的结果: {m}")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

class ImageDataset(Dataset):
    """用于加载单个目录中的图像的数据集"""
    def __init__(self, image_dir, transform=None):
        self.filenames = sorted(glob.glob(os.path.join(image_dir, "*.*")))
        self.transform = transform
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx]).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img

def calculate_ssim_from_paths(real_paths, fake_paths):
    """计算指定路径对之间的SSIM分数"""
    # 确保文件数量匹配
    assert len(real_paths) == len(fake_paths), \
        f"真实图像数量 ({len(real_paths)}) 与生成图像数量 ({len(fake_paths)}) 不匹配"
    
    ssim_values = []
    
    for real_path, fake_path in tqdm(zip(real_paths, fake_paths), total=len(real_paths), desc="计算SSIM"):
        # 读取图像
        real_img = cv2.imread(real_path)
        fake_img = cv2.imread(fake_path)
        
        # 转换为灰度图
        real_gray = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
        fake_gray = cv2.cvtColor(fake_img, cv2.COLOR_BGR2GRAY)
        
        # 计算SSIM
        score = ssim(real_gray, fake_gray)
        ssim_values.append(score)
    
    # 返回平均SSIM
    return np.mean(ssim_values)

def evaluate_pix2pix_by_dataset(results_dir, use_fid=True, use_ssim=True, batch_size=16):
    """按数据集分别评估pix2pix生成的图像质量"""
    print(f"评估目录: {results_dir}")
    
    # 获取所有图像
    all_images = glob.glob(os.path.join(results_dir, "*.*"))
    
    # 分类图像
    kaggle_real_A = [img for img in all_images if is_kaggle_image(img) and "_real_A." in img]
    kaggle_real_B = [img for img in all_images if is_kaggle_image(img) and "_real_B." in img]
    kaggle_fake_B = [img for img in all_images if is_kaggle_image(img) and "_fake_B." in img]
    
    cart_real_A = [img for img in all_images if is_cart_image(img) and "_real_A." in img]
    cart_real_B = [img for img in all_images if is_cart_image(img) and "_real_B." in img]
    cart_fake_B = [img for img in all_images if is_cart_image(img) and "_fake_B." in img]
    
    # 排序所有图像以确保对应关系
    kaggle_real_A.sort()
    kaggle_real_B.sort()
    kaggle_fake_B.sort()
    cart_real_A.sort()
    cart_real_B.sort()
    cart_fake_B.sort()
    
    print(f"Kaggle数据集: 找到 {len(kaggle_real_B)} 个真实图像B和 {len(kaggle_fake_B)} 个生成图像B")
    print(f"Cart数据集: 找到 {len(cart_real_B)} 个真实图像B和 {len(cart_fake_B)} 个生成图像B")
    
    results = {}
    
    # 处理Kaggle数据集
    if len(kaggle_real_B) > 0 and len(kaggle_fake_B) > 0:
        print("\n===== 评估Kaggle数据集 =====")
        
        # 计算FID
        if use_fid:
            # 创建FID计算器
            fid_calc = FIDCalculator()
            
            # 计算FID
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = ImagePairDataset(kaggle_real_B, kaggle_fake_B, transform)
            fid_score = fid_calc.calculate_fid_from_pairs(dataset, batch_size)
            results['FID_Kaggle'] = fid_score
            print(f"Kaggle数据集FID分数: {fid_score}")
        
        # 计算SSIM
        if use_ssim:
            ssim_score = calculate_ssim_from_paths(kaggle_real_B, kaggle_fake_B)
            results['SSIM_Kaggle'] = ssim_score
            print(f"Kaggle数据集平均SSIM分数: {ssim_score}")
    
    # 处理Cart数据集
    if len(cart_real_B) > 0 and len(cart_fake_B) > 0:
        print("\n===== 评估Cart数据集 =====")
        
        # 计算FID
        if use_fid:
            # 创建FID计算器
            fid_calc = FIDCalculator()
            
            # 计算FID
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = ImagePairDataset(cart_real_B, cart_fake_B, transform)
            fid_score = fid_calc.calculate_fid_from_pairs(dataset, batch_size)
            results['FID_Cart'] = fid_score
            print(f"Cart数据集FID分数: {fid_score}")
        
        # 计算SSIM
        if use_ssim:
            ssim_score = calculate_ssim_from_paths(cart_real_B, cart_fake_B)
            results['SSIM_Cart'] = ssim_score
            print(f"Cart数据集平均SSIM分数: {ssim_score}")
    
    # 如果两个数据集都有图像，也计算全部图像的指标
    if (len(kaggle_real_B) > 0 and len(kaggle_fake_B) > 0) and (len(cart_real_B) > 0 and len(cart_fake_B) > 0):
        print("\n===== 评估全部数据 =====")
        
        all_real_B = kaggle_real_B + cart_real_B
        all_fake_B = kaggle_fake_B + cart_fake_B
        
        # 计算FID
        if use_fid:
            fid_calc = FIDCalculator()
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = ImagePairDataset(all_real_B, all_fake_B, transform)
            fid_score = fid_calc.calculate_fid_from_pairs(dataset, batch_size)
            results['FID_All'] = fid_score
            print(f"全部数据FID分数: {fid_score}")
        
        # 计算SSIM
        if use_ssim:
            ssim_score = calculate_ssim_from_paths(all_real_B, all_fake_B)
            results['SSIM_All'] = ssim_score
            print(f"全部数据平均SSIM分数: {ssim_score}")
    
    return results, {
        'kaggle': {'real_A': kaggle_real_A, 'real_B': kaggle_real_B, 'fake_B': kaggle_fake_B},
        'cart': {'real_A': cart_real_A, 'real_B': cart_real_B, 'fake_B': cart_fake_B}
    }

def visualize_results_by_dataset(results_dir, image_data, num_samples_per_dataset=3):
    """按数据集分别可视化生成结果"""
    print(f"可视化结果: {results_dir}")
    
    kaggle_data = image_data['kaggle']
    cart_data = image_data['cart']
    
    # 计算总行数 - 每个数据集的样本数，如果该数据集有图像
    total_rows = 0
    if kaggle_data['real_B'] and kaggle_data['fake_B']:
        total_rows += min(num_samples_per_dataset, len(kaggle_data['real_B']))
    if cart_data['real_B'] and cart_data['fake_B']:
        total_rows += min(num_samples_per_dataset, len(cart_data['real_B']))
    
    if total_rows == 0:
        print("没有足够的图像用于可视化")
        return
    
    # 创建图像
    fig, axes = plt.subplots(total_rows, 3, figsize=(15, 5 * total_rows))
    
    # 如果只有一行，需要将axes转换为2D数组
    if total_rows == 1:
        axes = np.array([axes])
    
    row_idx = 0
    
    # 可视化Kaggle数据集
    if kaggle_data['real_B'] and kaggle_data['fake_B']:
        num_kaggle_samples = min(num_samples_per_dataset, len(kaggle_data['real_B']))
        
        # 随机选择样本
        if len(kaggle_data['real_B']) > num_kaggle_samples:
            indices = np.random.choice(len(kaggle_data['real_B']), num_kaggle_samples, replace=False)
            kaggle_samples = [(kaggle_data['real_A'][i], kaggle_data['real_B'][i], kaggle_data['fake_B'][i]) 
                             for i in indices]
        else:
            kaggle_samples = list(zip(kaggle_data['real_A'], kaggle_data['real_B'], kaggle_data['fake_B']))
        
        # 添加标题
        if row_idx == 0:
            fig.text(0.5, 1.0, 'Kaggle数据集', ha='center', va='center', fontsize=16, fontweight='bold')
        
        # 可视化样本
        for real_A_path, real_B_path, fake_B_path in kaggle_samples:
            real_A = plt.imread(real_A_path)
            real_B = plt.imread(real_B_path)
            fake_B = plt.imread(fake_B_path)
            
            axes[row_idx, 0].imshow(real_A)
            axes[row_idx, 0].set_title("Input (real_A)")
            axes[row_idx, 0].axis('off')
            
            axes[row_idx, 1].imshow(real_B)
            axes[row_idx, 1].set_title("Ground Truth (real_B)")
            axes[row_idx, 1].axis('off')
            
            axes[row_idx, 2].imshow(fake_B)
            axes[row_idx, 2].set_title("Generated (fake_B)")
            axes[row_idx, 2].axis('off')
            
            # 计算并显示SSIM
            real_B_gray = cv2.cvtColor(cv2.imread(real_B_path), cv2.COLOR_BGR2GRAY)
            fake_B_gray = cv2.cvtColor(cv2.imread(fake_B_path), cv2.COLOR_BGR2GRAY) 
            ssim_val = ssim(real_B_gray, fake_B_gray)
            axes[row_idx, 2].set_xlabel(f"SSIM: {ssim_val:.4f}")
            
            row_idx += 1
    
    # 可视化Cart数据集
    if cart_data['real_B'] and cart_data['fake_B']:
        num_cart_samples = min(num_samples_per_dataset, len(cart_data['real_B']))
        
        # 随机选择样本
        if len(cart_data['real_B']) > num_cart_samples:
            indices = np.random.choice(len(cart_data['real_B']), num_cart_samples, replace=False)
            cart_samples = [(cart_data['real_A'][i], cart_data['real_B'][i], cart_data['fake_B'][i]) 
                           for i in indices]
        else:
            cart_samples = list(zip(cart_data['real_A'], cart_data['real_B'], cart_data['fake_B']))
        
        # 添加标题
        fig.text(0.5, 0.5, 'Cart数据集', ha='center', va='center', fontsize=16, fontweight='bold')
        
        # 可视化样本
        for real_A_path, real_B_path, fake_B_path in cart_samples:
            real_A = plt.imread(real_A_path)
            real_B = plt.imread(real_B_path)
            fake_B = plt.imread(fake_B_path)
            
            axes[row_idx, 0].imshow(real_A)
            axes[row_idx, 0].set_title("Input (real_A)")
            axes[row_idx, 0].axis('off')
            
            axes[row_idx, 1].imshow(real_B)
            axes[row_idx, 1].set_title("Ground Truth (real_B)")
            axes[row_idx, 1].axis('off')
            
            axes[row_idx, 2].imshow(fake_B)
            axes[row_idx, 2].set_title("Generated (fake_B)")
            axes[row_idx, 2].axis('off')
            
            # 计算并显示SSIM
            real_B_gray = cv2.cvtColor(cv2.imread(real_B_path), cv2.COLOR_BGR2GRAY)
            fake_B_gray = cv2.cvtColor(cv2.imread(fake_B_path), cv2.COLOR_BGR2GRAY) 
            ssim_val = ssim(real_B_gray, fake_B_gray)
            axes[row_idx, 2].set_xlabel(f"SSIM: {ssim_val:.4f}")
            
            row_idx += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "visualization_by_dataset.png"))
    plt.show()

# 主函数 - 使用示例
if __name__ == "__main__":
    # 修改这些路径以匹配您的项目结构
    RESULTS_DIR = "/research/projects/trans_llm/Xiang_Zhang/data/ControlNet/training/cell_dataset/pytorch-CycleGAN-and-pix2pix/results/cell_pix2pix/test_latest/images"
    
    # 按数据集评估生成的图像
    metrics, image_data = evaluate_pix2pix_by_dataset(
        results_dir=RESULTS_DIR,
        use_fid=True,
        use_ssim=True,
        batch_size=8
    )
    
    # 可视化结果
    visualize_results_by_dataset(RESULTS_DIR, image_data, num_samples_per_dataset=3)
    
    # 将指标保存到文件
    if metrics:
        with open(os.path.join(os.path.dirname(RESULTS_DIR), "metrics_by_FID_SSIM.txt"), "w") as f:
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name}: {metric_value}\n")