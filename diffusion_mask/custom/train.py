import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm

# 路径设置
IMAGE_DIR = "/research/projects/trans_llm/Xiang_Zhang/data/ControlNet/training/cell_dataset/image"
MASK_DIR = "/research/projects/trans_llm/Xiang_Zhang/data/ControlNet/training/cell_dataset/mask"
OUTPUT_DIR = "./output"
SAMPLE_DIR = "./output/samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# 数据集
class CellDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.tif'))]
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, img_name)).convert("L")
        
        image = self.transform(image)
        mask = self.transform(mask)
        binary_mask = (mask > 0.5).float()
        
        # 将图像和掩码拼接为单一输入
        combined = torch.cat([image, binary_mask], dim=0)
        
        return {"combined": combined, "image": image, "mask": binary_mask, "filename": img_name}

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载数据
dataset = CellDataset(IMAGE_DIR, MASK_DIR)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 模型定义 - 使用标准UNet2DModel
model = UNet2DModel(
    sample_size=128,
    in_channels=2,    # 图像 + 掩码拼接在一起
    out_channels=1,   # 只预测图像
    layers_per_block=2,
    block_out_channels=(128, 256, 256, 512),
).to(device)

# 训练设置
scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_epochs = 20

# 生成样本函数
def generate_sample(mask, model, scheduler, device, filename, epoch, step):
    model.eval()
    with torch.no_grad():
        # 从随机噪声开始 (只在细胞区域)
        x = torch.randn((1, 1, 128, 128), device=device) * mask
        
        # 快速采样以节省时间
        for t in tqdm(range(999, -1, -50), desc="生成样本"):
            # 将当前图像和掩码拼接
            model_input = torch.cat([x, mask], dim=1)
            
            # 预测噪声
            noise_pred = model(model_input, torch.tensor([t], device=device)).sample
            
            # 去噪步骤
            x = scheduler.step(noise_pred, t, x).prev_sample
            
            # 只保留细胞区域
            x = x * mask
        
        # 保存生成的样本
        sample = (x.squeeze().cpu().numpy() * 255).clip(0, 255).astype('uint8')
        sample_path = f"{SAMPLE_DIR}/sample_{filename}_e{epoch}_s{step}.png"
        Image.fromarray(sample).save(sample_path)
        print(f"样本已保存至: {sample_path}")
    model.train()
    return sample_path

# 训练循环
for epoch in range(num_epochs):
    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        # 准备数据
        combined = batch["combined"].to(device)  # 图像和掩码的组合
        images = batch["image"].to(device)      # 原始图像
        masks = batch["mask"].to(device)        # 掩码（细胞区域为1，背景为0）
        filenames = batch["filename"]           # 文件名，用于样本生成
        
        # 添加噪声到原始图像
        noise = torch.randn_like(images).to(device)
        timesteps = torch.randint(0, 1000, (images.shape[0],), device=device)
        
        # 只对细胞区域添加噪声
        masked_noise = noise * masks
        noisy_images = scheduler.add_noise(images * masks, masked_noise, timesteps)
        
        # 创建模型输入
        model_input = torch.cat([noisy_images, masks], dim=1)
        
        # 模型预测
        noise_pred = model(model_input, timesteps).sample
        
        # 只在细胞区域计算损失
        loss = F.mse_loss(noise_pred * masks, masked_noise)
        
        # 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 定期保存模型和生成样本
        if step % 100 == 0:  # 减少到每100步
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.6f}")
            
            # 保存模型检查点
            checkpoint_path = f"{OUTPUT_DIR}/model_e{epoch+1}_s{step}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"模型已保存至: {checkpoint_path}")
            
            # 生成样本 (使用批次中的第一个样本)
            if step > 0:  # 避免在最初步骤就生成
                sample_path = generate_sample(
                    masks[0:1], 
                    model, 
                    scheduler, 
                    device, 
                    filenames[0].split('.')[0], 
                    epoch+1, 
                    step
                )

# 保存最终模型
final_model_path = f"{OUTPUT_DIR}/final_model.pt"
torch.save(model.state_dict(), final_model_path)
print(f"最终模型已保存至: {final_model_path}")
print("训练完成!")

# 生成变体函数
def generate_variants(image_path, mask_path, num_samples=3):
    # 加载模型
    model_path = f"{OUTPUT_DIR}/final_model.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 准备输入
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    image = transform(Image.open(image_path).convert("L")).unsqueeze(0).to(device)
    mask = transform(Image.open(mask_path).convert("L")).unsqueeze(0).to(device)
    binary_mask = (mask > 0.5).float()
    
    # 生成变体
    for i in range(num_samples):
        # 初始噪声
        x = torch.randn((1, 1, 128, 128), device=device) * binary_mask
        
        # 去噪过程
        for t in tqdm(range(999, -1, -10), desc=f"生成变体 {i+1}/{num_samples}"):
            with torch.no_grad():
                # 模型输入
                model_input = torch.cat([x, binary_mask], dim=1)
                
                # 预测噪声
                noise_pred = model(model_input, torch.tensor([t], device=device)).sample
                
                # 去噪步骤
                x = scheduler.step(noise_pred, t, x).prev_sample
                
                # 只保留细胞区域
                x = x * binary_mask
        
        # 保存结果
        result = (x.squeeze().cpu().numpy() * 255).clip(0, 255).astype('uint8')
        variant_path = f"{OUTPUT_DIR}/variant_{i+1}.png"
        Image.fromarray(result).save(variant_path)
        print(f"变体 {i+1} 已保存至: {variant_path}")
    
    print(f"已生成 {num_samples} 个变体!")

# 训练完成后取消下面注释来生成变体
"""
test_image = os.path.join(IMAGE_DIR, os.listdir(IMAGE_DIR)[0])
test_mask = os.path.join(MASK_DIR, os.listdir(MASK_DIR)[0])
generate_variants(test_image, test_mask, num_samples=3)
"""