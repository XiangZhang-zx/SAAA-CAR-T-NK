import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


# 添加自注意力模块
class SelfAttention(nn.Module):
    """ 自注意力层，用于增强特征图的表示能力 """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        # 初始化基本参数
        self.in_dim = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
        # 初始化权重参数，用于函数式API
        self.channel_factor = 8  # 通道缩减因子
        self.query_weight = nn.Parameter(torch.Tensor(in_dim // self.channel_factor, in_dim, 1, 1))
        self.key_weight = nn.Parameter(torch.Tensor(in_dim // self.channel_factor, in_dim, 1, 1))
        self.value_weight = nn.Parameter(torch.Tensor(in_dim, in_dim, 1, 1))
        
        # 初始化权重
        nn.init.kaiming_uniform_(self.query_weight)
        nn.init.kaiming_uniform_(self.key_weight)
        nn.init.kaiming_uniform_(self.value_weight)
        
    def get_attention_weights(self, C):
        """根据输入通道数动态调整权重"""
        # 如果通道数与初始化时一致，直接返回已有权重
        if C == self.in_dim:
            return self.query_weight, self.key_weight, self.value_weight
        
        # 否则，重新创建适合当前通道数的权重
        reduced_dim = max(C // self.channel_factor, 1)
        if not hasattr(self, 'dynamic_query_weight') or self.dynamic_query_weight.size(1) != C:
            self.dynamic_query_weight = nn.Parameter(torch.Tensor(reduced_dim, C, 1, 1)).to(self.gamma.device)
            self.dynamic_key_weight = nn.Parameter(torch.Tensor(reduced_dim, C, 1, 1)).to(self.gamma.device)
            self.dynamic_value_weight = nn.Parameter(torch.Tensor(C, C, 1, 1)).to(self.gamma.device)
            
            # 初始化动态权重
            nn.init.kaiming_uniform_(self.dynamic_query_weight)
            nn.init.kaiming_uniform_(self.dynamic_key_weight)
            nn.init.kaiming_uniform_(self.dynamic_value_weight)
            
        return self.dynamic_query_weight, self.dynamic_key_weight, self.dynamic_value_weight
    
    def forward(self, x):
        """
        inputs :
            x : 输入特征图 (B X C X W X H)
        returns :
            out : 自注意力值 + 输入特征
        """
        batch_size, C, width, height = x.size()
        
        # 获取适合当前输入通道数的权重
        q_weight, k_weight, v_weight = self.get_attention_weights(C)
        
        # 使用F.conv2d进行卷积运算
        proj_query = F.conv2d(x, q_weight).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = F.conv2d(x, k_weight).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        proj_value = F.conv2d(x, v_weight).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        # 残差连接
        out = self.gamma * out + x
        return out

# 添加Squeeze-and-Excitation模块
class SEModule(nn.Module):
    """ 通道注意力机制，Squeeze-and-Excitation模块 """
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.init_channel = channel
        self.reduction = reduction
        
        # 初始化基本的全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        
        # 如果输入通道数与初始化时不同，动态调整fc层
        if c != self.init_channel:
            # 创建适合当前通道数的全连接层
            device = x.device
            reduced_c = max(c // self.reduction, 1)
            
            fc = nn.Sequential(
                nn.Linear(c, reduced_c, bias=False).to(device),
                nn.ReLU(inplace=True),
                nn.Linear(reduced_c, c, bias=False).to(device),
                nn.Sigmoid()
            )
            y = fc(y).view(b, c, 1, 1)
        else:
            y = self.fc(y).view(b, c, 1, 1)
            
        return x * y.expand_as(x)


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    import sys

    # 计数器用于显示进度
    counter = {'count': 0, 'last_print': 0}

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

            # 显示进度（每100层打印一次）
            counter['count'] += 1
            if counter['count'] - counter['last_print'] >= 100:
                print(f"    ⏳ 已初始化 {counter['count']} 层...")
                sys.stdout.flush()
                counter['last_print'] = counter['count']

        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    print(f"    ✅ 总共初始化了 {counter['count']} 层")
    sys.stdout.flush()


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    import sys
    import time

    # 先在 CPU 上初始化权重（避免 GPU 兼容性问题导致卡住）
    print(f"  🔧 在 CPU 上初始化权重 (init_type={init_type})...")
    sys.stdout.flush()
    start = time.time()
    init_weights(net, init_type, init_gain=init_gain)
    print(f"  ✅ 权重初始化完成 (用时: {time.time()-start:.2f}秒)")
    sys.stdout.flush()

    # 然后将网络移动到 GPU
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        print(f"  🔧 将网络移动到 GPU {gpu_ids[0]}...")
        sys.stdout.flush()
        start = time.time()

        # 使用 non_blocking=True 加速传输
        net.to(gpu_ids[0])

        print(f"  ✅ 网络已移动到 GPU (用时: {time.time()-start:.2f}秒)")
        sys.stdout.flush()

        # 检查是否在 Accelerate 环境中（通过环境变量）
        import os
        use_accelerate = os.environ.get('ACCELERATE_USE_FSDP', None) is not None or \
                        os.environ.get('ACCELERATE_TORCH_DEVICE', None) is not None or \
                        len(gpu_ids) == 1  # 如果只有一个 GPU，也不需要 DataParallel

        if use_accelerate:
            print(f"  ⚠️  检测到 Accelerate 环境或单 GPU，跳过 DataParallel")
            print(f"  🔍 ACCELERATE_TORCH_DEVICE: {os.environ.get('ACCELERATE_TORCH_DEVICE', 'Not set')}")
            print(f"  🔍 GPU IDs: {gpu_ids}")
            sys.stdout.flush()
        else:
            print(f"  🔧 创建 DataParallel (GPUs: {gpu_ids})...")
            print(f"  🔍 CUDA device count: {torch.cuda.device_count()}")
            print(f"  🔍 Requested GPU IDs: {gpu_ids}")
            print(f"  🔍 Max GPU ID: {max(gpu_ids) if gpu_ids else 'N/A'}")
            sys.stdout.flush()
            start = time.time()
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
            print(f"  ✅ DataParallel 创建完成 (用时: {time.time()-start:.2f}秒)")
            sys.stdout.flush()

    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], use_attention=True):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        use_attention (bool) -- if use attention mechanisms in generator

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    print(f"  📦 创建 {netG} 网络实例...")
    import sys
    sys.stdout.flush()

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_attention=use_attention)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_attention=use_attention)
    elif netG == 'unetpp_128':
        net = UnetPlusPlusGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_attention=use_attention)
    elif netG == 'unetpp_256':
        net = UnetPlusPlusGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_attention=use_attention)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    print(f"  ✅ 网络实例创建完成")
    print(f"  🔧 开始初始化网络权重 (init_type={init_type})...")
    sys.stdout.flush()

    result = init_net(net, init_type, init_gain, gpu_ids)

    print(f"  ✅ 网络初始化完成")
    sys.stdout.flush()

    return result


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator with self-attention mechanism"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_attention=True):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            use_dropout (bool) -- if use dropout layers
            use_attention (bool) -- if use self-attention mechanism
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        
        # 在较深的层次添加自注意力模块
        if use_attention:
            # 在最深处添加自注意力机制
            attention_block = SelfAttention(ngf * 8)
            unet_block = nn.Sequential(unet_block, attention_block)
        
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        
        # 在中间层添加通道注意力模块
        if use_attention and num_downs > 5:
            se_block = SEModule(ngf * 8)
            unet_block = nn.Sequential(unet_block, se_block)
        
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


# 添加UNet++生成器
class UnetPlusPlusGenerator(nn.Module):
    """创建基于UNet++的生成器
    
    UNet++通过在原始UNet的基础上添加更多的跳跃连接，加强特征融合
    论文: UNet++: A Nested U-Net Architecture for Medical Image Segmentation
    """
    
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_attention=True):
        """构建UNet++生成器
        
        Parameters:
            input_nc (int)  -- 输入图像的通道数
            output_nc (int) -- 输出图像的通道数
            num_downs (int) -- UNet中的下采样次数，如果num_downs = 7，那么图像大小会从128x128降到1x1
            ngf (int)       -- 最后一个卷积层的过滤器数量
            norm_layer      -- 标准化层
            use_dropout (bool) -- 是否使用dropout
            use_attention (bool) -- 是否使用注意力机制
        """
        super(UnetPlusPlusGenerator, self).__init__()

        print(f"    🏗️  初始化 UNet++ (num_downs={num_downs}, ngf={ngf}, attention={use_attention})...")
        import sys
        sys.stdout.flush()

        # 编码器部分
        print(f"      📝 创建编码器列表...")
        sys.stdout.flush()
        self.encoders = nn.ModuleList()

        # 添加1层编码器，处理输入
        print(f"      📝 创建第0层编码器...")
        sys.stdout.flush()
        encoder0_0 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(inplace=True)
        )
        self.encoders.append(nn.ModuleList([encoder0_0]))
        print(f"      ✅ 第0层编码器创建完成")
        sys.stdout.flush()

        # 其他深度的编码器
        print(f"      📝 创建第1-{num_downs-1}层编码器...")
        sys.stdout.flush()
        for i in range(1, num_downs):
            if i % 2 == 0:
                print(f"        ⏳ 创建第{i}层编码器...")
                sys.stdout.flush()
            layer_encoders = nn.ModuleList()
            # 主编码器路径
            encoder = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.Conv2d(ngf * (2**(i-1)), ngf * (2**i), kernel_size=3, padding=1, bias=False),
                norm_layer(ngf * (2**i)),
                nn.ReLU(inplace=True),
                nn.Conv2d(ngf * (2**i), ngf * (2**i), kernel_size=3, padding=1, bias=False),
                norm_layer(ngf * (2**i)),
                nn.ReLU(inplace=True)
            )
            layer_encoders.append(encoder)
            self.encoders.append(layer_encoders)
        print(f"      ✅ 所有编码器创建完成")
        sys.stdout.flush()

        # 解码器部分
        print(f"      📝 创建解码器...")
        sys.stdout.flush()
        self.decoders = nn.ModuleList()
        for i in range(1, num_downs):
            if i % 2 == 0:
                print(f"        ⏳ 创建第{i}层解码器...")
                sys.stdout.flush()
            layer_decoders = nn.ModuleList()
            for j in range(i+1):
                if i >= 6 and j == 0:
                    print(f"          🔧 创建第{i}层第{j}个解码器...")
                    sys.stdout.flush()

                in_channels = 0
                # 计算输入通道数（取决于连接方式）
                if j == 0:  # 直接连接上层编码器
                    in_channels = ngf * (2**i)
                else:  # 连接编码器特征与上一个解码器特征
                    # 编码器特征: ngf * (2**(i-j))
                    # 上一个解码器输出: ngf * (2**(i-1))
                    in_channels = ngf * (2**(i-j)) + ngf * (2**(i-1))

                # 添加注意力机制
                attention_module = None
                if use_attention and i >= num_downs // 2:
                    if j == 0:  # 主解码路径上添加自注意力
                        if i >= 6:
                            print(f"            🔧 创建 SelfAttention(in_dim={ngf * (2**(i-1))})...")
                            sys.stdout.flush()
                        attention_module = SelfAttention(ngf * (2**(i-1)))
                        if i >= 6:
                            print(f"            ✅ SelfAttention 创建完成")
                            sys.stdout.flush()
                    elif i == num_downs - 1:  # 最深层添加SE模块
                        if i >= 6:
                            print(f"            🔧 创建 SEModule(in_dim={ngf * (2**(i-1))})...")
                            sys.stdout.flush()
                        attention_module = SEModule(ngf * (2**(i-1)))
                        if i >= 6:
                            print(f"            ✅ SEModule 创建完成")
                            sys.stdout.flush()
                
                # 具体的解码器
                # 最后一层的最后一个节点不需要上采样（已经达到输入尺寸）
                if i == num_downs - 1 and j == i:
                    # 最后一个节点：不上采样，只做卷积
                    decoder = nn.Sequential(
                        nn.Conv2d(in_channels, ngf * (2**(i-1)), kernel_size=3, padding=1, bias=False),
                        norm_layer(ngf * (2**(i-1))),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(ngf * (2**(i-1)), ngf * (2**(i-1)), kernel_size=3, padding=1, bias=False),
                        norm_layer(ngf * (2**(i-1))),
                        nn.ReLU(inplace=True)
                    )
                else:
                    # 其他节点：正常上采样
                    decoder = nn.Sequential(
                        nn.ConvTranspose2d(in_channels, ngf * (2**(i-1)), kernel_size=4, stride=2, padding=1, bias=False),
                        norm_layer(ngf * (2**(i-1))),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(ngf * (2**(i-1)), ngf * (2**(i-1)), kernel_size=3, padding=1, bias=False),
                        norm_layer(ngf * (2**(i-1))),
                        nn.ReLU(inplace=True)
                    )
                
                if attention_module:
                    decoder = nn.Sequential(decoder, attention_module)
                    
                if use_dropout and i >= num_downs // 2:
                    decoder = nn.Sequential(decoder, nn.Dropout(0.5))
                    
                layer_decoders.append(decoder)
            self.decoders.append(layer_decoders)

        print(f"      ✅ 所有解码器创建完成")
        sys.stdout.flush()

        # 最后的输出层
        print(f"      📝 创建最终输出层...")
        sys.stdout.flush()
        # 最后一层解码器的输出通道数是 ngf * (2**(num_downs-2))
        final_channels = ngf * (2**(num_downs-2))
        self.final = nn.Sequential(
            nn.Conv2d(final_channels, output_nc, kernel_size=1),
            nn.Tanh()
        )

        print(f"    ✅ UNet++ 初始化完成")
        sys.stdout.flush()

    def forward(self, x):
        # 保存所有节点的输出
        encoder_outputs = []
        decoder_outputs = []

        # 编码器前向传播
        for depth, encoders in enumerate(self.encoders):
            depth_encoder_outputs = []
            if depth == 0:
                # 第一层编码器
                out = encoders[0](x)
                depth_encoder_outputs.append(out)
                # print(f"🔍 Encoder depth={depth}, output shape: {out.shape}")
            else:
                # 更深层的编码器
                prev_out = encoder_outputs[-1][0]  # 获取上一层的输出
                out = encoders[0](prev_out)
                depth_encoder_outputs.append(out)
                # print(f"🔍 Encoder depth={depth}, output shape: {out.shape}")

            encoder_outputs.append(depth_encoder_outputs)
        
        # 解码器前向传播
        for depth, decoders in enumerate(self.decoders):
            current_depth = depth + 1  # 解码器从深度1开始
            depth_decoder_outputs = []

            for node_idx, decoder in enumerate(decoders):
                if node_idx == 0:
                    # 直连路径
                    encoder_feature = encoder_outputs[current_depth][0]
                    out = decoder(encoder_feature)
                    # print(f"🔍 Decoder depth={current_depth}, node={node_idx}, output shape: {out.shape}")
                else:
                    # 连接当前编码器特征与上层解码器特征
                    encoder_feature = encoder_outputs[current_depth-node_idx][0]
                    prev_decoder_feature = depth_decoder_outputs[-1]

                    # 确保特征图大小一致
                    if encoder_feature.shape[2:] != prev_decoder_feature.shape[2:]:
                        encoder_feature = nn.functional.interpolate(
                            encoder_feature,
                            size=prev_decoder_feature.shape[2:],
                            mode='bilinear',
                            align_corners=True
                        )

                    # 连接特征并通过解码器
                    combined = torch.cat([encoder_feature, prev_decoder_feature], dim=1)
                    out = decoder(combined)
                    # print(f"🔍 Decoder depth={current_depth}, node={node_idx}, output shape: {out.shape}")

                depth_decoder_outputs.append(out)

            decoder_outputs.append(depth_decoder_outputs)
        
        # 使用最后一层解码器的所有输出取平均
        # 需要先将所有输出 resize 到相同尺寸
        if len(decoder_outputs) > 0 and len(decoder_outputs[-1]) > 0:
            # 获取目标尺寸（使用最后一个输出的尺寸）
            target_size = decoder_outputs[-1][-1].shape[2:]

            # Resize 所有输出到相同尺寸
            resized_outputs = []
            for output in decoder_outputs[-1]:
                if output.shape[2:] != target_size:
                    output = nn.functional.interpolate(
                        output,
                        size=target_size,
                        mode='bilinear',
                        align_corners=True
                    )
                resized_outputs.append(output)

            # Stack 并取平均
            final_features = torch.stack(resized_outputs, dim=0).mean(dim=0)
        else:
            # 如果没有解码器输出，使用最后一个编码器输出
            final_features = encoder_outputs[-1][0]

        return self.final(final_features)
