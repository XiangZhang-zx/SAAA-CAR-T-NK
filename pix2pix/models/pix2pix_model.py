import torch
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import sys


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            # 添加新的创新点选项
            parser.add_argument('--use_attention', action='store_true', help='use attention mechanism in generator')
            parser.add_argument('--lambda_perc', type=float, default=10.0, help='weight for perceptual loss')
            parser.add_argument('--lambda_style', type=float, default=1.0, help='weight for style loss')
            parser.add_argument('--lambda_edge', type=float, default=5.0, help='weight for edge-preserving loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        
        # 添加新的损失函数到损失列表中，如果启用了这些新功能
        if self.isTrain and hasattr(opt, 'lambda_perc') and opt.lambda_perc > 0:
            self.loss_names.append('G_perc')
        if self.isTrain and hasattr(opt, 'lambda_style') and opt.lambda_style > 0:
            self.loss_names.append('G_style')
        if self.isTrain and hasattr(opt, 'lambda_edge') and opt.lambda_edge > 0:
            self.loss_names.append('G_edge')
            
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
            
        # 是否使用注意力机制
        self.use_attention = hasattr(opt, 'use_attention') and opt.use_attention

        print(f"🔧 开始创建生成器 (netG={opt.netG}, attention={self.use_attention})...")
        sys.stdout.flush()

        # define networks (both generator and discriminator)
        print(f"  🔍 调用 networks.define_G...")
        print(f"  🔍 参数: input_nc={opt.input_nc}, output_nc={opt.output_nc}, ngf={opt.ngf}")
        print(f"  🔍 参数: netG={opt.netG}, norm={opt.norm}, dropout={not opt.no_dropout}")
        print(f"  🔍 参数: init_type={opt.init_type}, init_gain={opt.init_gain}, gpu_ids={self.gpu_ids}")
        sys.stdout.flush()

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      use_attention=self.use_attention)

        print(f"  ✅ networks.define_G 返回成功")
        sys.stdout.flush()

        print(f"✅ 生成器创建完成")
        sys.stdout.flush()

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            print(f"🔧 开始创建判别器 (netD={opt.netD})...")
            sys.stdout.flush()

            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            print(f"✅ 判别器创建完成")
            sys.stdout.flush()

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            
            # 添加感知损失、样式损失和边缘保留损失（如果启用）
            if hasattr(opt, 'lambda_perc') and opt.lambda_perc > 0:
                # 初始化感知损失计算
                print(f"🔧 开始加载 VGG19 模型用于感知损失...")
                sys.stdout.flush()

                from torchvision.models import vgg19
                from torch.nn import L1Loss, MSELoss

                self.vgg = vgg19(pretrained=True).features[:35].eval().to(self.device)
                for param in self.vgg.parameters():
                    param.requires_grad = False

                self.criterionPerceptual = L1Loss()
                self.lambda_perc = opt.lambda_perc

                print(f"✅ VGG19 模型加载完成")
                sys.stdout.flush()
            
            if hasattr(opt, 'lambda_style') and opt.lambda_style > 0:
                # 初始化样式损失计算
                self.criterionStyle = MSELoss()
                self.lambda_style = opt.lambda_style
            
            if hasattr(opt, 'lambda_edge') and opt.lambda_edge > 0:
                # 初始化边缘保留损失
                self.edge_filter = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).unsqueeze(0).unsqueeze(0).float().to(self.device)
                self.criterionEdge = L1Loss()
                self.lambda_edge = opt.lambda_edge
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        
    def extract_features(self, x):
        """使用VGG19提取特征，用于感知和风格损失计算"""
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in {1, 6, 11, 20, 29}:  # 选择VGG中的特定层
                features.append(x)
        return features

    def compute_gram_matrix(self, x):
        """计算格拉姆矩阵，用于风格损失计算"""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
        
    def extract_edges(self, img):
        """使用Laplacian滤波器提取边缘信息"""
        # 确保图像是灰度图
        if img.shape[1] == 3:
            # 将RGB转为灰度
            img_gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            img_gray = img
            
        # 使用卷积操作提取边缘
        padding = nn.ReflectionPad2d(1)
        img_gray = padding(img_gray)
        edge = nn.functional.conv2d(img_gray, self.edge_filter)
        return edge

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        
        # 计算其他损失函数（如果启用）
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        
        # 感知损失和风格损失（共享VGG特征提取）
        if (hasattr(self, 'lambda_perc') and self.lambda_perc > 0) or (hasattr(self, 'lambda_style') and self.lambda_style > 0):
            # 规范化到VGG输入范围
            fake_B_norm = (self.fake_B + 1) / 2  # 从[-1,1]转换到[0,1]
            real_B_norm = (self.real_B + 1) / 2

            # 提取特征（只提取一次，供感知损失和风格损失共享）
            fake_features = self.extract_features(fake_B_norm)
            real_features = self.extract_features(real_B_norm)

            # 计算感知损失
            if hasattr(self, 'lambda_perc') and self.lambda_perc > 0:
                self.loss_G_perc = 0
                for fake_feat, real_feat in zip(fake_features, real_features):
                    self.loss_G_perc += self.criterionPerceptual(fake_feat, real_feat)
                self.loss_G_perc *= self.lambda_perc
                self.loss_G += self.loss_G_perc

            # 计算风格损失（使用相同的特征）
            if hasattr(self, 'lambda_style') and self.lambda_style > 0:
                self.loss_G_style = 0
                for fake_feat, real_feat in zip(fake_features, real_features):
                    fake_gram = self.compute_gram_matrix(fake_feat)
                    real_gram = self.compute_gram_matrix(real_feat)
                    self.loss_G_style += self.criterionStyle(fake_gram, real_gram)
                self.loss_G_style *= self.lambda_style
                self.loss_G += self.loss_G_style
            
        # 边缘保留损失
        if hasattr(self, 'lambda_edge') and self.lambda_edge > 0:
            fake_edges = self.extract_edges(self.fake_B)
            real_edges = self.extract_edges(self.real_B)
            self.loss_G_edge = self.criterionEdge(fake_edges, real_edges) * self.lambda_edge
            self.loss_G += self.loss_G_edge
            
        # 计算梯度
        self.loss_G.backward()

    def optimize_parameters(self, accelerator=None):
        """Standard optimization with optional Accelerate support

        Args:
            accelerator: Optional Accelerate accelerator for distributed training
        """
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero

        if accelerator is not None:
            # Use Accelerate's backward for distributed training
            self.backward_D_no_backward()  # calculate loss without calling .backward()
            accelerator.backward(self.loss_D)
        else:
            self.backward_D()  # calculate gradients for D (includes .backward())

        self.optimizer_D.step()          # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero

        if accelerator is not None:
            # Use Accelerate's backward for distributed training
            self.backward_G_no_backward()  # calculate loss without calling .backward()
            accelerator.backward(self.loss_G)
        else:
            self.backward_G()  # calculate graidents for G (includes .backward())

        self.optimizer_G.step()             # update G's weights

    def backward_D_no_backward(self):
        """Calculate GAN loss for discriminator WITHOUT calling .backward()"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss (no .backward())
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

    def backward_G_no_backward(self):
        """Calculate GAN and L1 loss for generator WITHOUT calling .backward()"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # 计算其他损失函数（如果启用）
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        # 感知损失和风格损失（共享VGG特征提取）
        if (hasattr(self, 'lambda_perc') and self.lambda_perc > 0) or (hasattr(self, 'lambda_style') and self.lambda_style > 0):
            # 规范化到VGG输入范围
            fake_B_norm = (self.fake_B + 1) / 2  # 从[-1,1]转换到[0,1]
            real_B_norm = (self.real_B + 1) / 2

            # 提取特征（只提取一次，供感知损失和风格损失共享）
            fake_features = self.extract_features(fake_B_norm)
            real_features = self.extract_features(real_B_norm)

            # 计算感知损失
            if hasattr(self, 'lambda_perc') and self.lambda_perc > 0:
                self.loss_G_perc = 0
                for fake_feat, real_feat in zip(fake_features, real_features):
                    self.loss_G_perc += self.criterionPerceptual(fake_feat, real_feat)
                self.loss_G_perc *= self.lambda_perc
                self.loss_G += self.loss_G_perc

            # 计算风格损失（使用相同的特征）
            if hasattr(self, 'lambda_style') and self.lambda_style > 0:
                self.loss_G_style = 0
                for fake_feat, real_feat in zip(fake_features, real_features):
                    fake_gram = self.compute_gram_matrix(fake_feat)
                    real_gram = self.compute_gram_matrix(real_feat)
                    self.loss_G_style += self.criterionStyle(fake_gram, real_gram)
                self.loss_G_style *= self.lambda_style
                self.loss_G += self.loss_G_style

        # 边缘保留损失
        if hasattr(self, 'lambda_edge') and self.lambda_edge > 0:
            fake_edges = self.extract_edges(self.fake_B)
            real_edges = self.extract_edges(self.real_B)
            self.loss_G_edge = self.criterionEdge(fake_edges, real_edges) * self.lambda_edge
            self.loss_G += self.loss_G_edge

    def test_with_innovations(self):
        """测试所有添加的创新点，并返回每个创新对应的结果和指标
        
        Returns:
            results (dict): 包含各种创新功能的效果对比
        """
        results = {}
        
        # 保存原始前向传播结果
        self.forward()
        original_fake_B = self.fake_B.clone()
        results['standard'] = {'image': original_fake_B}
        
        # 如果有注意力机制
        if hasattr(self, 'use_attention') and self.use_attention:
            # 这里我们只能展示最终结果，因为注意力机制已集成到模型中
            results['attention'] = {'image': self.fake_B.clone()}
            
            # 如果需要可视化注意力权重，可以修改网络结构返回注意力图
        
        # 计算感知损失
        if hasattr(self, 'lambda_perc') and self.lambda_perc > 0:
            # 规范化到VGG输入范围
            fake_B_norm = (self.fake_B + 1) / 2
            real_B_norm = (self.real_B + 1) / 2
            
            # 提取特征
            fake_features = self.extract_features(fake_B_norm)
            real_features = self.extract_features(real_B_norm)
            
            # 计算各层特征的感知损失
            perc_losses = {}
            layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
            for i, (fake_feat, real_feat) in enumerate(zip(fake_features, real_features)):
                loss = self.criterionPerceptual(fake_feat, real_feat).item()
                perc_losses[layer_names[i]] = loss
                
            # 可视化中间特征（取第一个特征图）
            feature_vis = {}
            for i, feat in enumerate(fake_features):
                # 取第一个通道的特征图
                feature_map = feat[0, 0].detach().cpu().unsqueeze(0).unsqueeze(0)
                # 归一化特征图以便可视化
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                feature_vis[layer_names[i]] = feature_map
                
            results['perceptual'] = {
                'losses': perc_losses,
                'feature_maps': feature_vis
            }
        
        # 计算风格损失
        if hasattr(self, 'lambda_style') and self.lambda_style > 0:
            if not hasattr(self, 'lambda_perc'):  # 如果还没计算特征
                fake_B_norm = (self.fake_B + 1) / 2
                real_B_norm = (self.real_B + 1) / 2
                fake_features = self.extract_features(fake_B_norm)
                real_features = self.extract_features(real_B_norm)
                
            # 计算各层特征的Gram矩阵和风格损失
            style_losses = {}
            layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
            for i, (fake_feat, real_feat) in enumerate(zip(fake_features, real_features)):
                fake_gram = self.compute_gram_matrix(fake_feat)
                real_gram = self.compute_gram_matrix(real_feat)
                loss = self.criterionStyle(fake_gram, real_gram).item()
                style_losses[layer_names[i]] = loss
                
            results['style'] = {
                'losses': style_losses
            }
        
        # 计算边缘保留损失
        if hasattr(self, 'lambda_edge') and self.lambda_edge > 0:
            fake_edges = self.extract_edges(self.fake_B)
            real_edges = self.extract_edges(self.real_B)
            edge_loss = self.criterionEdge(fake_edges, real_edges).item()
            
            # 归一化边缘特征图以便可视化
            fake_edge_vis = (fake_edges - fake_edges.min()) / (fake_edges.max() - fake_edges.min() + 1e-8)
            real_edge_vis = (real_edges - real_edges.min()) / (real_edges.max() - real_edges.min() + 1e-8)
            
            results['edge'] = {
                'loss': edge_loss,
                'fake_edges': fake_edge_vis,
                'real_edges': real_edge_vis
            }
        
        # 如果使用了UNet++，可以添加相应的可视化
        model_type = self.opt.netG if hasattr(self.opt, 'netG') else ''
        if 'unetpp' in model_type:
            results['model_type'] = 'UNet++'
        else:
            results['model_type'] = 'UNet' if 'unet' in model_type else 'ResNet'
        
        return results
