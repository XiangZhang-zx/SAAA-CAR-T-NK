"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot ./datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def save_innovation_results(model, web_dir, name, visualize_features=True, data=None):
    """保存创新点测试结果到网页和图像文件
    
    Args:
        model: 要测试的模型
        web_dir: 保存结果的目录
        name: 实验名称
        visualize_features: 是否可视化特征图
        data: 用于测试的输入数据，如果为None，则尝试使用模型中已有的数据
    """
    # 如果提供了数据，设置输入
    if data is not None:
        model.set_input(data)
    
    # 检查模型是否有test_with_innovations方法
    if not hasattr(model, 'test_with_innovations'):
        print("警告：当前模型没有test_with_innovations方法，尝试使用基本测试...")
        # 调用基本的测试方法
        model.test()
        # 获取基本的可视化结果
        visuals = model.get_current_visuals()
        
        # 创建基本结果目录
        if not os.path.exists(os.path.join(web_dir, 'innovations')):
            os.makedirs(os.path.join(web_dir, 'innovations'))
        
        # 创建简单的HTML报告
        innovations_html = os.path.join(web_dir, 'innovations', f'{name}_innovations.html')
        with open(innovations_html, 'w') as f:
            f.write('<html><body>\n')
            f.write(f'<h1>基本测试结果 - {name}</h1>\n')
            f.write('<p>注意：当前模型不支持创新点测试。显示基本测试结果。</p>\n')
            
            # 保存生成的图像
            for label, img_tensor in visuals.items():
                if isinstance(img_tensor, torch.Tensor):
                    # 转换tensor为numpy图像
                    img_np = (img_tensor[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2.0
                    img_path = os.path.join(web_dir, 'innovations', f'{name}_{label}.png')
                    plt.imsave(img_path, np.clip(img_np, 0, 1))
                    
                    # 添加到HTML
                    f.write(f'<div><h3>{label}</h3><img src="./innovations/{name}_{label}.png" width="256"></div>\n')
            
            f.write('</body></html>\n')
        
        print(f"基本测试结果已保存到: {innovations_html}")
        return
    
    # 如果模型支持创新点测试，继续原有逻辑
    results = model.test_with_innovations()
    
    if not os.path.exists(os.path.join(web_dir, 'innovations')):
        os.makedirs(os.path.join(web_dir, 'innovations'))
    
    # 创建可视化网页
    innovations_html = os.path.join(web_dir, 'innovations', f'{name}_innovations.html')
    with open(innovations_html, 'w') as f:
        f.write('<html><body>\n')
        f.write(f'<h1>创新点效果展示 - {name}</h1>\n')
        f.write('<table><tr><th>类型</th><th>效果图</th><th>相关指标</th></tr>\n')
        
        # 保存基本生成结果
        if 'standard' in results:
            img = results['standard']['image']
            # 归一化到[0,1]
            img_np = (img[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2.0
            img_path = os.path.join(web_dir, 'innovations', f'{name}_standard.png')
            plt.imsave(img_path, np.clip(img_np, 0, 1))
            
            # 添加到HTML
            f.write(f'<tr><td>标准模型</td><td><img src="./innovations/{name}_standard.png" width="256"></td><td>基准模型</td></tr>\n')
        
        # 保存注意力机制结果
        if 'attention' in results:
            img = results['attention']['image']
            img_np = (img[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2.0
            img_path = os.path.join(web_dir, 'innovations', f'{name}_attention.png')
            plt.imsave(img_path, np.clip(img_np, 0, 1))
            
            f.write(f'<tr><td>注意力机制</td><td><img src="./innovations/{name}_attention.png" width="256"></td><td>增强特征表示能力</td></tr>\n')
        
        # 保存感知损失相关结果
        if 'perceptual' in results and visualize_features:
            metrics_html = '<ul>'
            for layer_name, loss in results['perceptual']['losses'].items():
                metrics_html += f'<li>{layer_name}: {loss:.4f}</li>'
            metrics_html += '</ul>'
            
            # 可视化一些特征图
            if 'feature_maps' in results['perceptual']:
                for layer_name, feat_map in results['perceptual']['feature_maps'].items():
                    feat_np = feat_map[0, 0].detach().cpu().numpy()
                    feat_path = os.path.join(web_dir, 'innovations', f'{name}_percept_{layer_name}.png')
                    plt.imsave(feat_path, feat_np, cmap='viridis')
                    metrics_html += f'<div><img src="./innovations/{name}_percept_{layer_name}.png" width="128" title="{layer_name}"></div>'
            
            f.write(f'<tr><td>感知损失</td><td><img src="./innovations/{name}_standard.png" width="256"></td><td>{metrics_html}</td></tr>\n')
        
        # 保存风格损失相关结果
        if 'style' in results:
            metrics_html = '<ul>'
            for layer_name, loss in results['style']['losses'].items():
                metrics_html += f'<li>{layer_name}: {loss:.4f}</li>'
            metrics_html += '</ul>'
            
            f.write(f'<tr><td>风格损失</td><td><img src="./innovations/{name}_standard.png" width="256"></td><td>{metrics_html}</td></tr>\n')
        
        # 保存边缘保留损失相关结果
        if 'edge' in results:
            # 保存边缘特征图
            fake_edge = results['edge']['fake_edges']
            real_edge = results['edge']['real_edges']
            
            fake_edge_np = fake_edge[0, 0].detach().cpu().numpy()
            real_edge_np = real_edge[0, 0].detach().cpu().numpy()
            
            fake_edge_path = os.path.join(web_dir, 'innovations', f'{name}_fake_edge.png')
            real_edge_path = os.path.join(web_dir, 'innovations', f'{name}_real_edge.png')
            
            plt.imsave(fake_edge_path, fake_edge_np, cmap='gray')
            plt.imsave(real_edge_path, real_edge_np, cmap='gray')
            
            edge_loss = results['edge']['loss']
            metrics_html = f'<p>边缘损失: {edge_loss:.4f}</p>'
            metrics_html += f'<div>生成图边缘:<img src="./innovations/{name}_fake_edge.png" width="128"></div>'
            metrics_html += f'<div>真实图边缘:<img src="./innovations/{name}_real_edge.png" width="128"></div>'
            
            f.write(f'<tr><td>边缘保留损失</td><td><img src="./innovations/{name}_standard.png" width="256"></td><td>{metrics_html}</td></tr>\n')
            
        # 模型类型信息
        if 'model_type' in results:
            f.write(f'<tr><td>模型架构</td><td>{results["model_type"]}</td><td>使用的生成器架构类型</td></tr>\n')
            
        f.write('</table>\n')
        f.write('</body></html>\n')
    
    print(f"创新点测试结果已保存到: {innovations_html}")
    

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    # 注释掉 batch_size 的硬编码，允许从命令行参数设置
    # opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options

    # Convert BatchNorm to SyncBatchNorm BEFORE loading weights
    # This is needed if the model was trained with Accelerate (which uses SyncBatchNorm)
    import torch.nn as nn
    if hasattr(model, 'netG'):
        model.netG = nn.SyncBatchNorm.convert_sync_batchnorm(model.netG)
        print("✅ 已将 Generator 的 BatchNorm 转换为 SyncBatchNorm")
    if hasattr(model, 'netD'):
        model.netD = nn.SyncBatchNorm.convert_sync_batchnorm(model.netD)
        print("✅ 已将 Discriminator 的 BatchNorm 转换为 SyncBatchNorm")

    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    # 创建网页展示结果
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    
    # 测试创新点功能并生成报告
    if opt.eval_innovations:
        print("正在测试创新点功能...")
        # 获取第一个样本数据用于创新点测试
        test_data = None
        for i, data in enumerate(dataset):
            test_data = data
            break
        if test_data is not None:
            save_innovation_results(model, web_dir, opt.name, data=test_data)
        else:
            print("警告：未能获取测试数据，尝试使用模型中已有数据进行创新点测试")
            save_innovation_results(model, web_dir, opt.name)
        
    # 常规测试流程
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    # 计数已处理的图像数量（而不是 batch 数量）
    num_images_processed = 0
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths

        # 获取当前 batch 的实际图像数量
        current_batch_size = len(img_path)

        # 检查是否超过 num_test 限制
        if num_images_processed + current_batch_size > opt.num_test:
            # 只处理剩余的图像
            remaining = opt.num_test - num_images_processed
            if remaining > 0:
                # 截取 visuals 和 img_path
                for key in visuals.keys():
                    visuals[key] = visuals[key][:remaining]
                img_path = img_path[:remaining]
                save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
                num_images_processed += remaining
            break

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (num_images_processed, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        num_images_processed += current_batch_size

        if num_images_processed >= opt.num_test:
            break

    print(f'✅ 完成！共处理 {num_images_processed} 张图像')
    webpage.save()  # save the HTML
