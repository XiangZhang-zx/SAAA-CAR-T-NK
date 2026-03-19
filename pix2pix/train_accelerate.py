"""Accelerate-based training script for image-to-image translation.

This is a modified version of train.py that uses Hugging Face Accelerate for:
- Automatic multi-GPU training with balanced memory usage
- Mixed precision training (FP16)
- Gradient accumulation for larger effective batch sizes

Usage:
    accelerate launch --config_file accelerate_config.yaml train_accelerate.py [options]
    
Example:
    accelerate launch --config_file accelerate_config.yaml train_accelerate.py \
        --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
"""
import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from accelerate import Accelerator
import os


def _save_model_safe(model, epoch, accelerator):
    """Safely save model in Accelerate DDP environment.

    This function unwraps the DDP model and saves only on the main process,
    avoiding NCCL communication issues during checkpoint saving.

    Args:
        model: The model to save (may be wrapped in DDP)
        epoch: Epoch identifier for the checkpoint filename
        accelerator: Accelerator instance
    """
    # Only save on main process
    if not accelerator.is_main_process:
        return

    # Save each network separately
    for name in model.model_names:
        if isinstance(name, str):
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(model.save_dir, save_filename)
            net = getattr(model, 'net' + name)

            # Unwrap the model from DDP/Accelerate wrapper
            unwrapped_net = accelerator.unwrap_model(net)

            # Save to CPU to avoid GPU memory issues
            torch.save(unwrapped_net.cpu().state_dict(), save_path)

            # Move back to device
            unwrapped_net.to(accelerator.device)


if __name__ == '__main__':
    # Initialize Accelerator with DDP kwargs
    from accelerate.utils import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,  # Required for GAN training where D and G are updated separately
        broadcast_buffers=False,  # Avoid broadcasting buffers to reduce overhead
    )

    accelerator = Accelerator(
        mixed_precision='bf16',  # Use BF16 mixed precision (better for H200)
        gradient_accumulation_steps=1,  # Can increase for larger effective batch size
        kwargs_handlers=[ddp_kwargs],  # Pass DDP configuration
    )

    opt = TrainOptions().parse()   # get training options

    # Enable gradient checkpointing if requested
    use_gradient_checkpointing = hasattr(opt, 'use_gradient_checkpointing') and opt.use_gradient_checkpointing

    # 修改 gpu_ids 为单个 GPU（Accelerate 会自动分配）
    # 每个进程只看到一个 GPU（通过 CUDA_VISIBLE_DEVICES）
    opt.gpu_ids = [0]  # 强制使用单 GPU 模式，避免 DataParallel

    # Only print on main process
    if accelerator.is_main_process:
        print(f'🚀 Accelerate Training Started')
        print(f'   - Device: {accelerator.device}')
        print(f'   - Num processes: {accelerator.num_processes}')
        print(f'   - Mixed precision: {accelerator.mixed_precision}')
        print(f'   - GPU IDs (per process): {opt.gpu_ids}')

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    if accelerator.is_main_process:
        print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if accelerator.is_main_process:
        print(f"\n🔧 准备模型和优化器...")
        print(f"   - Generator: {type(model.netG).__name__}")
        print(f"   - Discriminator: {type(model.netD).__name__}")

    # Convert BatchNorm to SyncBatchNorm to avoid inplace operation errors
    import torch.nn as nn
    model.netG = nn.SyncBatchNorm.convert_sync_batchnorm(model.netG)
    model.netD = nn.SyncBatchNorm.convert_sync_batchnorm(model.netD)
    if accelerator.is_main_process:
        print("✅ 已将 BatchNorm 转换为 SyncBatchNorm")

    # Enable gradient checkpointing if requested
    if use_gradient_checkpointing:
        if hasattr(model.netG, 'gradient_checkpointing_enable'):
            model.netG.gradient_checkpointing_enable()
            if accelerator.is_main_process:
                print("✅ Generator: 已启用 Gradient Checkpointing")
        else:
            # For UNet, manually enable gradient checkpointing
            from torch.utils.checkpoint import checkpoint
            if accelerator.is_main_process:
                print("⚠️  UNet 不支持自动 gradient checkpointing，将在训练中手动应用")

    # Prepare model, optimizers, and dataloader with Accelerator
    # Important: Prepare networks and optimizers together
    if hasattr(model, 'netG') and hasattr(model, 'netD'):
        # Prepare main networks and optimizers
        model.netG, model.netD, model.optimizer_G, model.optimizer_D, dataset = accelerator.prepare(
            model.netG, model.netD, model.optimizer_G, model.optimizer_D, dataset
        )

        # Move loss functions and other modules to the correct device
        # These don't need gradients but need to be on the right device
        if hasattr(model, 'vgg'):
            model.vgg = model.vgg.to(accelerator.device)
        if hasattr(model, 'criterionGAN'):
            model.criterionGAN = model.criterionGAN.to(accelerator.device)
        if hasattr(model, 'criterionL1'):
            model.criterionL1 = model.criterionL1.to(accelerator.device)
        if hasattr(model, 'criterionPerceptual'):
            model.criterionPerceptual = model.criterionPerceptual.to(accelerator.device)
        if hasattr(model, 'criterionStyle'):
            model.criterionStyle = model.criterionStyle.to(accelerator.device)
        if hasattr(model, 'criterionEdge'):
            model.criterionEdge = model.criterionEdge.to(accelerator.device)
        if hasattr(model, 'edge_filter'):
            model.edge_filter = model.edge_filter.to(accelerator.device)

        if accelerator.is_main_process:
            print(f"✅ 模型和优化器已准备完成")
    else:
        raise RuntimeError("Model must have netG, netD, optimizer_G, and optimizer_D attributes")
    
    visualizer = Visualizer(opt) if accelerator.is_main_process else None  # create visualizer only on main process
    total_iters = 0                # the total number of training iterations

    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        print(f"🚀 开始训练循环...")
        print(f"   - 总轮数: {opt.n_epochs + opt.n_epochs_decay}")
        print(f"   - 起始轮数: {opt.epoch_count}")
        print(f"{'='*80}\n")

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        if accelerator.is_main_process:
            print(f"\n{'='*80}")
            print(f"📊 Epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay}")
            print(f"{'='*80}")

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        if visualizer:
            visualizer.reset()

        model.update_learning_rate()

        if accelerator.is_main_process:
            print(f"🔄 开始迭代数据...")

        for i, data in enumerate(dataset):
            if accelerator.is_main_process and i == 0:
                print(f"✅ 收到第一个 batch，开始训练...")
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # Set input - move data to accelerator device
            AtoB = opt.direction == 'AtoB'
            model.real_A = data['A' if AtoB else 'B'].to(accelerator.device)
            model.real_B = data['B' if AtoB else 'A'].to(accelerator.device)
            model.image_paths = data['A_paths' if AtoB else 'B_paths']

            # Use model's optimize_parameters with Accelerate support
            model.optimize_parameters(accelerator=accelerator)

            # Display and logging (only on main process)
            if accelerator.is_main_process:
                if total_iters % opt.display_freq == 0:
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_iters % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                if total_iters % opt.save_latest_freq == 0:
                    if accelerator.is_main_process:
                        print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                        save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                        # Only main process saves - unwrap model to avoid DDP issues
                        _save_model_safe(model, save_suffix, accelerator)

            iter_data_time = time.time()
        
        # Save model at end of epoch (only on main process)
        if accelerator.is_main_process:
            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                # Only main process saves - use safe save to avoid DDP issues
                _save_model_safe(model, 'latest', accelerator)
                _save_model_safe(model, epoch, accelerator)

            print('End of epoch %d / %d \t Time Taken: %d sec' % 
                  (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

