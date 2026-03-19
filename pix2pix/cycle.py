"""
This script uses the official pytorch-CycleGAN-and-pix2pix implementation to train
on a cell dataset with masks. It handles dataset preparation, training, and generation
of cell variants.
"""
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['PYTHONUNBUFFERED'] = '1'  # 强制不缓冲输出
import sys
import shutil
import random
import subprocess
from PIL import Image
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import importlib
import runpy

# 从环境变量或默认值获取训练配置
BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE', '1024'))
GPU_IDS = os.environ.get('TRAIN_GPU_IDS', '0,1,2,3')
TEST_BATCH_SIZE = int(os.environ.get('TEST_BATCH_SIZE', '1024'))
TEST_GPU_IDS = os.environ.get('TEST_GPU_IDS', '0,1,2,3')

# Your dataset paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # CART_NK_Augmentation/
IMAGE_DIR = os.path.join(_PROJECT_ROOT, "datasets", "cart_nk", "source")
MASK_DIR = os.path.join(_PROJECT_ROOT, "datasets", "cart_nk", "mask")

# Output and repository paths
OUTPUT_DIR = "./cell_pix2pix_results"
REPO_DIR = "."  # 使用当前目录
DATASET_DIR = "./datasets/cell_dataset"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{DATASET_DIR}/train", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/test", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/val", exist_ok=True)

# 多进程处理单个图像的函数
def process_single_image(args):
    """处理单个图像文件 - 用于多进程"""
    img_name, split_dir = args
    try:
        # Load mask and image
        mask_path = os.path.join(MASK_DIR, img_name)
        image_path = os.path.join(IMAGE_DIR, img_name)

        if not os.path.exists(mask_path) or not os.path.exists(image_path):
            return f"跳过 {img_name}: 文件不存在"

        mask = Image.open(mask_path).convert("L")
        cell_img = Image.open(image_path).convert("L")

        # Resize to 256x256 (standard size for pix2pix)
        mask = mask.resize((256, 256), Image.LANCZOS)
        cell_img = cell_img.resize((256, 256), Image.LANCZOS)

        # Create the combined A|B image (mask|cell_image)
        combined = Image.new('RGB', (512, 256))

        # Convert grayscale to RGB
        mask_rgb = Image.new('RGB', mask.size)
        mask_rgb.paste(mask)
        cell_rgb = Image.new('RGB', cell_img.size)
        cell_rgb.paste(cell_img)

        # Paste side by side: mask on left (A), cell on right (B)
        combined.paste(mask_rgb, (0, 0))
        combined.paste(cell_rgb, (256, 0))

        # Save the combined image
        output_path = os.path.join(split_dir, f"{os.path.splitext(img_name)[0]}.jpg")
        combined.save(output_path)

        return f"✅ {img_name}"
    except Exception as e:
        return f"❌ {img_name}: {str(e)}"


# Step 1: Prepare the dataset in the format expected by pix2pix
def prepare_dataset():
    # 检查数据集是否已存在
    if os.path.exists(f"{DATASET_DIR}/train") and \
       os.path.exists(f"{DATASET_DIR}/val") and \
       len(os.listdir(f"{DATASET_DIR}/train")) > 0:
        print("✅ 数据集已存在，跳过准备步骤...")
        train_count = len(os.listdir(f"{DATASET_DIR}/train"))
        val_count = len(os.listdir(f"{DATASET_DIR}/val"))
        print(f"📊 现有数据集: 训练集 {train_count} 张, 验证集 {val_count} 张")
        return

    print("🔄 准备数据集...")
    print(f"📁 图像目录: {IMAGE_DIR}")
    print(f"📁 掩码目录: {MASK_DIR}")

    # 检查源目录
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ 错误: 图像目录不存在 {IMAGE_DIR}")
        return
    if not os.path.exists(MASK_DIR):
        print(f"❌ 错误: 掩码目录不存在 {MASK_DIR}")
        return
        
    print("Preparing dataset in pix2pix format...")
    
    # Get list of all image files
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.tif'))]
    
    # Shuffle files for random split
    random.shuffle(image_files)
    
    # Split into train/val (95:5)
    train_size = int(0.95 * len(image_files))
    
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]
    
    # Process each split
    for split_name, file_list in [("train", train_files), ("val", val_files)]:
        split_dir = f"{DATASET_DIR}/{split_name}"
        os.makedirs(split_dir, exist_ok=True)
        print(f"Processing {split_name} set ({len(file_list)} images)...")
        
        # 获取CPU核心数，用于多进程
        num_workers = min(cpu_count(), 32)  # 最多使用32个进程
        print(f"🚀 使用 {num_workers} 个进程加速处理...")

        # 准备多进程参数
        process_args = [(img_name, split_dir) for img_name in file_list]

        # 使用多进程处理
        with Pool(processes=num_workers) as pool:
            # 使用imap_unordered获得进度条
            results = list(tqdm(
                pool.imap_unordered(process_single_image, process_args),
                total=len(process_args),
                desc=f"Processing {split_name}"
            ))

        # 统计结果
        success_count = sum(1 for r in results if r.startswith("✅"))
        error_count = sum(1 for r in results if r.startswith("❌"))
        skip_count = sum(1 for r in results if r.startswith("跳过"))

        print(f"📊 {split_name} 处理完成:")
        print(f"  ✅ 成功: {success_count}")
        print(f"  ❌ 错误: {error_count}")
        print(f"  ⏭️  跳过: {skip_count}")
            
    print("Dataset preparation completed!")
    print(f"Train: {len(train_files)}, Val: {len(val_files)} images")

# Step 2: Train the pix2pix model using the official script
def train_model():
    print("Starting pix2pix training...")
    
    # Change to repository directory
    os.chdir(REPO_DIR)
    
    # Build the training command
    train_cmd = [
        "python", "train.py",
        "--dataroot", f"./datasets/cell_dataset",
        "--name", "cell_pix2pix_v200",
        "--model", "pix2pix",
        "--direction", "AtoB",         # Mask to cell image
        "--input_nc", "3",             # Input channels (RGB)
        "--output_nc", "3",            # Output channels (RGB)
        "--netG", "unet_256",          # Generator architecture
        "--norm", "batch",             # Use batch normalization
        "--n_epochs", "200",           # Number of epochs with initial learning rate
        "--n_epochs_decay", "100",     # Number of epochs with decaying learning rate
        "--batch_size", "32",         # Batch size (optimized for H200)
        "--gpu_ids", "0,1,2,3",        # Use physical GPU 0,1,2,3 directly
        "--save_epoch_freq", "20",     # Save model every 20 epochs
        "--lambda_L1", "100",          # Weight for L1 loss
        "--display_id", "-1"           # Disable visdom display
    ]
    
    # Run the training command
    subprocess.run(train_cmd, check=True)
    print("Training completed!")
    
    # Return to original directory
    os.chdir("..")

# Step 2.1: Train the innovative pix2pix model with new features
def train_model_innovative():
    print("开始使用创新功能的 pix2pix 训练...")

    # 保存当前工作目录
    original_dir = os.getcwd()

    try:
        # 切换到代码库目录
        os.chdir(REPO_DIR)

        # 检查是否使用 Accelerate（通过环境变量控制）
        use_accelerate = os.environ.get('USE_ACCELERATE', 'false').lower() == 'true'

        if use_accelerate:
            print(f"🚀 使用 Accelerate 进行训练")
            # 从环境变量获取进程数
            num_processes = int(os.environ.get('NUM_PROCESSES', '4'))

            # CUDA_VISIBLE_DEVICES 已经在 train_h200_optimized.sh 中设置
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
            print(f"   - CUDA_VISIBLE_DEVICES: {cuda_visible}")
            print(f"   - Num Processes: {num_processes}")

            # BATCH_SIZE 就是每个 GPU 的 batch size（不需要除以 num_processes）
            per_gpu_batch_size = BATCH_SIZE
            total_batch_size = BATCH_SIZE * num_processes
            print(f"   - Per-GPU Batch Size: {per_gpu_batch_size}")
            print(f"   - Total Batch Size: {total_batch_size}")

            # 使用随机端口避免冲突
            import random
            main_process_port = random.randint(30000, 40000)
            print(f"   - Main Process Port: {main_process_port}")

            # 使用 Accelerate 启动（直接在命令行指定参数，不依赖配置文件）
            train_cmd = [
                "accelerate", "launch",
                "--mixed_precision", "fp16",
                "--num_processes", str(num_processes),
                "--main_process_port", str(main_process_port),  # 使用随机端口
                "train_accelerate.py",
                "--dataroot", f"./datasets/cell_dataset",
                "--name", "innov_cell_pixpix",
                "--model", "pix2pix",
                "--direction", "AtoB",
                "--input_nc", "3",
                "--output_nc", "3",
                "--netG", "unet_256",
                "--ngf", "64",
                "--norm", "batch",
                "--n_epochs", "200",
                "--n_epochs_decay", "100",
                "--batch_size", str(per_gpu_batch_size),  # 每个 GPU 的 batch size
                "--gpu_ids", "0",              # 在 Accelerate 中，每个进程只看到一个 GPU (逻辑 ID 0)
                "--save_epoch_freq", "20",
                "--lambda_L1", "100",
                "--use_attention",
                "--lambda_perc", "10.0",
                "--lambda_style", "1.0",
                "--lambda_edge", "5.0",
                "--display_id", "-1"
            ]
            print(f"🔧 Accelerate 训练配置:")
            print(f"   - Per-GPU Batch Size: {per_gpu_batch_size}")
            print(f"   - Total Batch Size: {total_batch_size}")
            print(f"   - Num GPUs: {num_processes}")
        else:
            # 原始训练命令（使用环境变量配置）
            train_cmd = [
                "python", "train.py",
                "--dataroot", f"./datasets/cell_dataset",
                "--name", "innov_cell_pixpix",  # 新的实验名称
                "--model", "pix2pix",
                "--direction", "AtoB",         # 从掩码到细胞图像
                "--input_nc", "3",             # 输入通道数 (RGB)
                "--output_nc", "3",            # 输出通道数 (RGB)
                "--netG", "unet_256",        # 使用UNet++架构替代标准UNet
                "--ngf", "64",                 # 标准 UNet 使用默认 ngf=64
                "--norm", "batch",             # 使用批量归一化
                "--n_epochs", "200",           # 初始学习率的轮数
                "--n_epochs_decay", "100",     # 学习率衰减的轮数
                "--batch_size", str(BATCH_SIZE),  # 从环境变量读取
                "--gpu_ids", GPU_IDS,          # 从环境变量读取
                "--save_epoch_freq", "20",     # 每20轮保存一次模型
                "--lambda_L1", "100",          # L1损失的权重
                "--use_attention",             # 启用注意力机制
                "--lambda_perc", "10.0",       # 启用感知损失
                "--lambda_style", "1.0",       # 启用风格损失
                "--lambda_edge", "5.0",        # 启用边缘保留损失
                "--display_id", "-1"           # 禁用visdom显示
            ]
            print(f"🔧 训练配置:")
            print(f"   - Batch Size: {BATCH_SIZE}")
            print(f"   - GPU IDs: {GPU_IDS}")

        # 直接调用训练脚本 - 实时输出到terminal
        print("🚀 开始训练，所有输出将实时显示...")
        print("=" * 80)
        sys.stdout.flush()

        # 如果使用 Accelerate，直接用 subprocess 运行命令
        if use_accelerate:
            print(f"🚀 使用 subprocess 运行 Accelerate 命令...")
            print(f"   命令: {' '.join(train_cmd)}")
            sys.stdout.flush()

            import subprocess
            result = subprocess.run(train_cmd, cwd=REPO_DIR)
            if result.returncode != 0:
                raise RuntimeError(f"Accelerate 训练失败，返回码: {result.returncode}")
            return  # 训练完成，直接返回

        # 原始方式：设置命令行参数 (去掉 "python" 和 "train.py"，只保留参数)
        sys.argv = ['train.py'] + train_cmd[2:]  # train_cmd[0]='python', train_cmd[1]='train.py'

        print(f"🔍 调试信息: 完整 sys.argv =")
        for i, arg in enumerate(sys.argv):
            print(f"    [{i}] {arg}")
        # 检查 ngf 参数
        if '--ngf' in sys.argv:
            ngf_idx = sys.argv.index('--ngf')
            print(f"🔍 找到 --ngf 参数在位置 {ngf_idx}，值为: {sys.argv[ngf_idx+1]}")
        else:
            print(f"⚠️  警告: 未找到 --ngf 参数！")
        print(f"🔍 当前工作目录: {os.getcwd()}")
        print(f"🔍 train.py 是否存在: {os.path.exists('train.py')}")
        sys.stdout.flush()

        # 清除 Python 模块缓存，避免使用旧的配置
        print("🔧 清除 Python 模块缓存...")
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('options.') or k.startswith('models.')]
        for mod in modules_to_remove:
            del sys.modules[mod]
        print(f"🔧 已清除 {len(modules_to_remove)} 个模块")
        sys.stdout.flush()

        print("🔥 正在执行 train.py...")
        sys.stdout.flush()

        # 禁用输出缓冲以确保实时显示
        old_stdout = sys.stdout
        old_stderr = sys.stderr

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

        # 使用 runpy 运行 train.py (会执行 if __name__ == '__main__' 块)
        try:
            runpy.run_path('train.py', run_name='__main__')
        except Exception as e:
            print(f"❌ 训练出错: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # 恢复原始 stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        print("=" * 80)
        print("✅ 创新版 pix2pix 训练完成!")

    finally:
        # 恢复原始工作目录
        os.chdir(original_dir)

# Step 3: Generate variants using the trained model
def generate_variants(num_variants=5):
    print(f"Generating {num_variants} cell variants...")
    
    # Change to repository directory
    os.chdir(REPO_DIR)
    
    # 首先检查模型文件是否存在
    model_path = "./checkpoints/cell_pix2pix/latest_net_G.pth"
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return
    
    # 创建输出目录
    os.makedirs(f"../{OUTPUT_DIR}", exist_ok=True)
    
    # 添加 --no_dropout 参数到测试命令
    test_cmd = [
        "python", "test.py",
        "--dataroot", f"./datasets/cell_dataset/test",
        "--name", "cell_pix2pix",
        "--model", "test",
        "--netG", "unet_256",
        "--direction", "AtoB",
        "--num_test", str(num_variants),
        "--results_dir", f"../{OUTPUT_DIR}",
        "--no_dropout"  # 添加这个参数来解决模型加载问题
    ]
    
    try:
        print("运行测试命令：", " ".join(test_cmd))
        subprocess.run(test_cmd, check=True)
        print(f"生成的变体已保存到 {OUTPUT_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"生成变体时出错：{e}")
    except Exception as e:
        print(f"发生未知错误：{e}")
    
    # Return to original directory
    os.chdir("..")

# Step 3.1: Generate variants using the innovative trained model
def generate_variants_innovative(num_variants=5):
    print(f"使用创新模型生成 {num_variants} 个细胞变体...")
    
    # 切换到代码库目录
    os.chdir(REPO_DIR)
    
    # 检查模型文件是否存在
    model_path = "./checkpoints/innov_cell_pixpix/latest_net_G.pth"
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return
    
    # 创建输出目录
    output_dir = f"../{OUTPUT_DIR}/innovative"
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建测试命令
    test_cmd = [
        "python", "test.py",
        "--dataroot", f"./datasets/cell_dataset/test",
        "--name", "innov_cell_pixpix",
        "--model", "test",
        "--netG", "unet_256",  # 使用与训练相同的标准 UNet 架构
        "--direction", "AtoB",
        "--num_test", str(num_variants),
        "--results_dir", output_dir,
        "--batch_size", "1024",  # 加速生成 (4 GPU，每个 GPU 32 样本)
        "--no_dropout",
        "--use_attention",      # 启用注意力机制
        "--lambda_perc", "10.0", # 启用感知损失
        "--lambda_edge", "5.0",  # 启用边缘保留损失
        "--eval_innovations"    # 启用创新点评估
    ]
    
    try:
        print("运行测试命令：", " ".join(test_cmd))
        subprocess.run(test_cmd, check=True)
        print(f"生成的变体已保存到 {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"生成变体时出错：{e}")
    except Exception as e:
        print(f"发生未知错误：{e}")
    
    # 返回原目录
    os.chdir("..")

# Step 4: Generate custom variants from new masks
def generate_custom_variants(mask_path, num_variants=3):
    """Generate variants from a specific mask file"""
    print(f"Generating {num_variants} custom variants from {mask_path}...")
    
    # Create a temporary test directory with the custom mask
    temp_test_dir = f"{DATASET_DIR}/custom_test"
    os.makedirs(temp_test_dir, exist_ok=True)
    
    # Load the mask
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((256, 256), Image.LANCZOS)
    
    # For each variant, create a slightly different paired image
    for i in range(num_variants):
        # Create an empty "fake" cell image (black)
        cell_placeholder = Image.new('L', (256, 256), 0)
        
        # Create combined image
        combined = Image.new('RGB', (512, 256))
        mask_rgb = Image.merge('RGB', (mask, mask, mask))
        cell_rgb = Image.merge('RGB', (cell_placeholder, cell_placeholder, cell_placeholder))
        
        combined.paste(mask_rgb, (0, 0))
        combined.paste(cell_rgb, (256, 0))
        
        # Save with different names to get multiple variants
        output_path = os.path.join(temp_test_dir, f"custom_mask_{i}.jpg")
        combined.save(output_path)
    
    # Run the test script on these custom images
    os.chdir(REPO_DIR)
    
    test_cmd = [
        "python", "test.py",
        "--dataroot", f"./datasets/cell_dataset/custom_test",
        "--name", "cell_pix2pix",
        "--model", "test",
        "--netG", "unet_256",
        "--direction", "AtoB",
        "--results_dir", f"../{OUTPUT_DIR}/custom_variants"
    ]
    
    subprocess.run(test_cmd, check=True)
    print(f"Custom variants saved to {OUTPUT_DIR}/custom_variants")
    
    
    # Clean up
    os.chdir("..")
    
# Step 4.1: Generate custom variants using innovative model
def generate_custom_variants_innovative(mask_path, num_variants=3):
    """使用创新模型从特定掩码文件生成变体"""
    print(f"使用创新模型从 {mask_path} 生成 {num_variants} 个自定义变体...")
    
    # 创建临时测试目录，存放自定义掩码
    temp_test_dir = f"{DATASET_DIR}/custom_test_innov"
    os.makedirs(temp_test_dir, exist_ok=True)
    
    # 加载掩码
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((256, 256), Image.LANCZOS)
    
    # 为每个变体创建略微不同的配对图像
    for i in range(num_variants):
        # 创建空的"假"细胞图像（黑色）
        cell_placeholder = Image.new('L', (256, 256), 0)
        
        # 创建组合图像
        combined = Image.new('RGB', (512, 256))
        mask_rgb = Image.merge('RGB', (mask, mask, mask))
        cell_rgb = Image.merge('RGB', (cell_placeholder, cell_placeholder, cell_placeholder))
        
        combined.paste(mask_rgb, (0, 0))
        combined.paste(cell_rgb, (256, 0))
        
        # 使用不同的名称保存，以获得多个变体
        output_path = os.path.join(temp_test_dir, f"custom_mask_innov_{i}.jpg")
        combined.save(output_path)
    
    # 在这些自定义图像上运行测试脚本
    os.chdir(REPO_DIR)
    
    # 使用创新模型进行测试
    test_cmd = [
        "python", "test.py",
        "--dataroot", f"./datasets/cell_dataset/custom_test_innov",
        "--name", "innov_cell_pixpix",
        "--model", "test",
        "--netG", "unet_256",
        "--direction", "AtoB",
        "--batch_size", "1024",  # 加速生成
        "--results_dir", f"../{OUTPUT_DIR}/custom_variants_innovative",
        "--use_attention",      # 启用注意力机制
        "--lambda_perc", "10.0", # 启用感知损失
        "--lambda_edge", "5.0",  # 启用边缘保留损失
        "--eval_innovations"
    ]
    
    subprocess.run(test_cmd, check=True)
    print(f"使用创新模型生成的自定义变体已保存到 {OUTPUT_DIR}/custom_variants_innovative")
    
    # 清理
    os.chdir("..")

# Step 3.2: Generate MASSIVE variants using train dataset (NEW!)
def generate_massive_variants_innovative(num_variants=1000):
    """使用创新模型从训练集生成大量变体 (包含增强mask)"""
    print(f"🚀 使用创新模型从训练集生成 {num_variants} 个细胞变体...")
    print("📊 这将包含所有增强过的mask (elastic, contrast, rotate, flip, scale等)")

    # 保存当前工作目录
    original_dir = os.getcwd()

    try:
        # 切换到代码库目录
        os.chdir(REPO_DIR)

        # 检查模型文件是否存在
        model_path = "./checkpoints/innov_cell_pixpix/latest_net_G.pth"
        if not os.path.exists(model_path):
            print(f"❌ 错误：找不到模型文件 {model_path}")
            print("请先训练创新模型！")
            return

        # 确保结果目录存在 - 保存到result_A
        output_dir = f"../result_A"
        os.makedirs(output_dir, exist_ok=True)

        # 使用训练集而不是测试集
        train_dataroot = f"./datasets/cell_dataset/train"

        # 检查训练集大小
        if os.path.exists(train_dataroot):
            train_files = [f for f in os.listdir(train_dataroot) if f.endswith('.jpg')]
            total_train_files = len(train_files)
            print(f"📁 训练集总文件数: {total_train_files}")
            print(f"🎯 请求生成数量: {num_variants}")

            # 如果请求数量超过训练集大小，使用全部训练集
            actual_num = min(num_variants, total_train_files)
            if actual_num < num_variants:
                print(f"⚠️  训练集只有 {total_train_files} 个文件，将生成 {actual_num} 个变体")
        else:
            print(f"❌ 训练集目录不存在: {train_dataroot}")
            return

        # 构建测试命令 - 使用训练集（使用环境变量配置）
        test_cmd = [
            "python", "test.py",
            "--dataroot", train_dataroot,
            "--name", "innov_cell_pixpix",
            "--model", "test",
            "--netG", "unet_256",  # 使用与训练相同的标准 UNet 架构
            "--direction", "AtoB",
            "--num_test", str(actual_num),
            "--results_dir", output_dir,
            "--batch_size", str(TEST_BATCH_SIZE),  # 从环境变量读取
            "--gpu_ids", TEST_GPU_IDS,  # 从环境变量读取
            "--no_dropout",
            "--use_attention",      # 启用注意力机制
            "--lambda_perc", "10.0", # 启用感知损失
            "--lambda_edge", "5.0",  # 启用边缘保留损失
            "--eval_innovations"    # 启用创新点评估
        ]

        print(f"🔧 测试配置:")
        print(f"   - Batch Size: {TEST_BATCH_SIZE}")
        print(f"   - GPU IDs: {TEST_GPU_IDS}")

        print("🔥 开始大规模生成，所有输出将实时显示...")
        print("=" * 80)
        sys.stdout.flush()

        # 设置命令行参数 (去掉 "python" 和 "test.py"，只保留参数)
        sys.argv = ['test.py'] + test_cmd[2:]  # test_cmd[0]='python', test_cmd[1]='test.py'

        # 禁用输出缓冲以确保实时显示
        old_stdout = sys.stdout
        old_stderr = sys.stderr

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

        # 使用 runpy 运行 test.py
        try:
            runpy.run_path('test.py', run_name='__main__')
        except Exception as e:
            print(f"❌ 生成出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 恢复原始 stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        print("=" * 80)
        print("✅ 大规模创新模型测试完成！")

        # 统计生成结果
        result_images_dir = f"{output_dir}/innov_cell_pixpix/test_latest/images"
        if os.path.exists(result_images_dir):
            real_a_files = [f for f in os.listdir(result_images_dir) if f.endswith('_real_A.png')]
            fake_b_files = [f for f in os.listdir(result_images_dir) if f.endswith('_fake_B.png')]

            print(f"📊 生成统计:")
            print(f"   - 输入mask (*_real_A.png): {len(real_a_files)} 个")
            print(f"   - 生成细胞 (*_fake_B.png): {len(fake_b_files)} 个")

            # 统计增强类型
            augment_types = {}
            for filename in real_a_files:
                if filename.startswith('original_'):
                    # 提取原始文件名
                    original_name = filename.replace('original_', '').replace('_real_A.png', '')
                    if original_name.startswith(('elastic_', 'contrast_', 'rotate', 'flip_', 'scale_', 'morph_')):
                        aug_type = original_name.split('_')[0]
                        augment_types[aug_type] = augment_types.get(aug_type, 0) + 1
                    else:
                        augment_types['original'] = augment_types.get('original', 0) + 1

            print(f"🎨 增强类型分布:")
            for aug_type, count in sorted(augment_types.items()):
                print(f"   - {aug_type}: {count} 个")

    finally:
        # 恢复原始工作目录
        os.chdir(original_dir)

# Run the complete pipeline
if __name__ == "__main__":
    # Ask the user what they want to do
    print("Cell Pix2Pix Pipeline")
    print("1. 准备数据集")
    print("2. 完整流程（准备、训练、生成变体）")
    print("3. 使用训练好的模型生成变体")
    print("4. 从特定掩码文件生成变体")
    print("5. 使用创新点训练模型") # 新选项
    print("6. 使用创新模型生成变体") # 新选项
    print("7. 使用创新模型从特定掩码生成变体") # 新选项
    print("8. 🚀 使用创新模型从训练集生成大量变体 (包含增强mask)") # 新选项

    choice = input("输入您的选择 (1-8): ")
    
    if choice == '1':
        prepare_dataset()
    elif choice == '2':
        prepare_dataset()
        train_model()
        generate_variants()
    elif choice == '3':
        num = input("生成多少个变体? [默认=5]: ")
        num = int(num) if num.strip() else 5
        generate_variants(num)
    elif choice == '4':
        mask_path = input("输入掩码文件路径: ")
        num = input("生成多少个变体? [默认=3]: ")
        num = int(num) if num.strip() else 3
        generate_custom_variants(mask_path, num)
    elif choice == '5':
        prepare_dataset()
        train_model_innovative()
    elif choice == '6':
        num = input("生成多少个变体? [默认=5]: ")
        num = int(num) if num.strip() else 5
        generate_variants_innovative(num)
    elif choice == '7':
        mask_path = input("输入掩码文件路径: ")
        num = input("生成多少个变体? [默认=3]: ")
        num = int(num) if num.strip() else 3
        generate_custom_variants_innovative(mask_path, num)
    elif choice == '8':
        num = input("生成多少个变体? [默认=1000, 最大=35825]: ")
        num = int(num) if num.strip() else 1000
        generate_massive_variants_innovative(num)
    else:
        print("Invalid choice. Exiting.")