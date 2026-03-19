#!/usr/bin/env python3
"""
训练无条件扩散模型生成 mask 图像
基于 Hugging Face Diffusers 的 DDPM 训练脚本
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="训练无条件扩散模型")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="训练数据目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="uncond-ddpm-mask-256",
        help="输出目录",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="图像分辨率",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="训练批次大小",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="训练轮数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="学习率",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="学习率预热步数",
    )
    parser.add_argument(
        "--save_images_epochs",
        type=int,
        default=5,
        help="每隔多少轮保存生成的图像",
    )
    parser.add_argument(
        "--save_model_epochs",
        type=int,
        default=10,
        help="每隔多少轮保存模型",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="混合精度训练",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="使用 EMA",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="日志目录",
    )
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化 Accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, args.logging_dir),
    )
    
    # 加载数据集
    dataset = load_dataset(
        "imagefolder",
        data_dir=args.train_data_dir,
        split="train",
    )
    
    # 数据预处理
    preprocess = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}
    
    dataset.set_transform(transform)
    
    # 创建数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4
    )
    
    # 创建模型
    model = UNet2DModel(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    
    # 创建噪声调度器
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 创建学习率调度器
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )
    
    # 使用 Accelerator 准备
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # 准备 EMA（在 accelerator.prepare 之后，确保在正确的设备上）
    if args.use_ema:
        ema_model = EMAModel(model.parameters())
        ema_model.to(accelerator.device)
    
    # 训练循环
    global_step = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            
            # 采样噪声
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            
            # 采样随机时间步
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()
            
            # 添加噪声到图像
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # 预测噪声
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                
                # 计算损失
                loss = F.mse_loss(noise_pred, noise)
                
                # 反向传播
                accelerator.backward(loss)
                
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 更新 EMA
            if args.use_ema:
                ema_model.step(model.parameters())
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        
        progress_bar.close()
        
        # 保存生成的图像
        if accelerator.is_main_process:
            if (epoch + 1) % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                # 生成图像
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(model),
                    scheduler=noise_scheduler,
                )
                
                if args.use_ema:
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                
                generator = torch.Generator(device=pipeline.device).manual_seed(0)
                images = pipeline(
                    generator=generator,
                    batch_size=4,
                    num_inference_steps=50,
                    output_type="pil",
                ).images
                
                # 保存图像（二值化为纯黑白）
                image_dir = os.path.join(args.output_dir, "samples", f"epoch_{epoch}")
                os.makedirs(image_dir, exist_ok=True)
                for i, image in enumerate(images):
                    # 二值化处理：转为纯黑白
                    gray = image.convert('L')
                    img_array = np.array(gray)
                    binary_array = np.where(img_array > 128, 255, 0).astype(np.uint8)
                    binary_image = Image.fromarray(binary_array, mode='L')

                    # 保存原始图像和二值化图像
                    binary_image.save(os.path.join(image_dir, f"sample_{i}.png"))
                    image.save(os.path.join(image_dir, f"sample_{i}_raw.png"))  # 保留原始图像用于对比
                
                if args.use_ema:
                    ema_model.restore(model.parameters())
            
            # 保存模型
            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(model),
                    scheduler=noise_scheduler,
                )
                
                if args.use_ema:
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                
                pipeline.save_pretrained(os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}"))
                
                if args.use_ema:
                    ema_model.restore(model.parameters())
    
    accelerator.end_training()

if __name__ == "__main__":
    main()
