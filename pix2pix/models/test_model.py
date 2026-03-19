from .base_model import BaseModel
from . import networks
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B']  # 修改为与pix2pix模型一致的命名
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        
        # 保存是否使用注意力机制的标志
        self.use_attention = hasattr(opt, 'use_attention') and opt.use_attention
        
        # 加载带有注意力机制的生成器（如果需要）
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      use_attention=self.use_attention)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.
        
        # 保存与创新点相关的选项
        self.has_perceptual = hasattr(opt, 'lambda_perc') and opt.lambda_perc > 0
        self.has_style = hasattr(opt, 'lambda_style') and opt.lambda_style > 0
        self.has_edge = hasattr(opt, 'lambda_edge') and opt.lambda_edge > 0
        
        # 如果启用了感知损失，加载VGG模型
        if self.has_perceptual or self.has_style:
            try:
                from torchvision.models import vgg19
                self.vgg = vgg19(pretrained=True).features[:35].eval().to(self.device)
                for param in self.vgg.parameters():
                    param.requires_grad = False
            except:
                self.vgg = None
                print("警告: 无法加载VGG模型用于感知损失计算")
        
        # 如果启用了边缘损失，创建边缘检测滤波器
        if self.has_edge:
            self.edge_filter = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).unsqueeze(0).unsqueeze(0).float().to(self.device)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        For single dataset mode, only domain A is available, so we use it as both real_A and real_B.
        """
        # 在测试阶段，数据集通常是single模式，只有A或B
        if 'B' in input:  # 如果有B域数据
            AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.image_paths = input['A_paths' if AtoB else 'B_paths']
        else:  # 如果是单域数据集
            self.real_A = input['A'].to(self.device)
            # 在测试模式下，我们不需要真实的B，但为了保持一致性，我们可以设置一个虚拟的real_B
            self.real_B = torch.zeros_like(self.real_A)  # 创建一个全零tensor作为占位符
            self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass."""
        self.fake_B = self.netG(self.real_A)  # G(A)，修改为与pix2pix模型一致的命名

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
        
    def extract_features(self, x):
        """使用VGG19提取特征，用于感知和风格损失计算"""
        if not hasattr(self, 'vgg') or self.vgg is None:
            return []
            
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
        if not hasattr(self, 'edge_filter'):
            return None
            
        # 确保图像是灰度图
        if img.shape[1] == 3:
            # 将RGB转为灰度
            img_gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            img_gray = img
            
        # 使用卷积操作提取边缘
        padding = nn.ReflectionPad2d(1)
        img_gray = padding(img_gray)
        edge = F.conv2d(img_gray, self.edge_filter)
        return edge
        
    def test_with_innovations(self):
        """测试所有添加的创新点，并返回每个创新对应的结果和指标
        
        Returns:
            results (dict): 包含各种创新功能的效果对比
        """
        results = {}
        
        # 检查是否已经设置了输入数据
        if not hasattr(self, 'real_A') or self.real_A is None:
            print("警告: real_A未设置，请确保在调用test_with_innovations前先调用set_input")
            # 如果需要继续执行，可以创建一个空的tensor作为real_A
            if hasattr(self, 'real_B'):
                self.real_A = torch.zeros_like(self.real_B)
            else:
                # 如果连real_B都没有，可能需要从其他地方获取尺寸信息
                # 这里简单返回一个空字典，表示无法执行测试
                return {"error": "输入数据未设置"}
        
        # 确保forward已经运行
        if not hasattr(self, 'fake_B'):
            self.forward()
        
        # 基本生成结果
        original_fake = self.fake_B.clone()
        results['standard'] = {'image': original_fake}
        
        # 注意力机制（如果启用）
        if self.use_attention:
            # 在测试模式下，我们只能展示最终结果
            results['attention'] = {'image': self.fake_B.clone()}
        
        # 计算感知损失（如果启用）
        if self.has_perceptual and hasattr(self, 'vgg') and self.vgg is not None:
            # 从[-1,1]规范化到[0,1]，用于VGG
            fake_norm = (self.fake_B + 1) / 2
            
            # 提取特征
            fake_features = self.extract_features(fake_norm)
            
            if fake_features:
                # 创建特征可视化
                layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
                feature_vis = {}
                perc_losses = {}  # 空的损失字典，因为测试时没有真实目标
                
                for i, feat in enumerate(fake_features):
                    if i < len(layer_names):
                        # 取第一个通道的特征图
                        feature_map = feat[0, 0].detach().cpu().unsqueeze(0).unsqueeze(0)
                        # 归一化特征图以便可视化
                        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                        feature_vis[layer_names[i]] = feature_map
                        # 添加一个占位符损失值
                        perc_losses[layer_names[i]] = 0.0
                
                results['perceptual'] = {
                    'feature_maps': feature_vis,
                    'losses': perc_losses
                }
        
        # 边缘提取（如果启用）
        if self.has_edge:
            fake_edges = self.extract_edges(self.fake_B)
            
            if fake_edges is not None:
                # 归一化边缘特征图以便可视化
                fake_edge_vis = (fake_edges - fake_edges.min()) / (fake_edges.max() - fake_edges.min() + 1e-8)
                
                # 测试模式下可能没有real_B，或者real_B是零tensor，所以使用fake_B的边缘作为real_edges的替代
                real_edges = fake_edges.clone()  # 在测试时，我们没有真实的边缘可以比较
                real_edge_vis = fake_edge_vis.clone()
                
                results['edge'] = {
                    'fake_edges': fake_edge_vis,
                    'real_edges': real_edge_vis,
                    'loss': 0.0  # 占位符损失值
                }
        
        # 模型类型信息
        model_type = self.opt.netG if hasattr(self.opt, 'netG') else ''
        if 'unetpp' in model_type:
            results['model_type'] = 'UNet++'
        else:
            results['model_type'] = 'UNet' if 'unet' in model_type else 'ResNet'
        
        return results
