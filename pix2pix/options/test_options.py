import contextlib
import io
import os

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        
        # 添加创新点相关的测试选项
        parser.add_argument('--eval_innovations', action='store_true', help='测试所有添加的创新点并生成对比报告')
        parser.add_argument('--visualize_features', action='store_true', help='可视化中间特征图')
        
        # 添加与训练阶段相同的创新点参数
        parser.add_argument('--use_attention', action='store_true', help='use attention mechanism in generator')
        parser.add_argument('--lambda_perc', type=float, default=10.0, help='weight for perceptual loss')
        parser.add_argument('--lambda_edge', type=float, default=5.0, help='weight for edge-preserving loss')
        
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser

    def parse(self):
        """Parse options and ensure critical generator settings match the training config."""
        opt = super().parse()
        if not opt.isTrain:
            synced = self._sync_with_training_options(opt)
            if synced:
                # Re-save options file with the updated values without printing twice
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    self.print_options(opt)
        self.opt = opt
        return self.opt

    def _sync_with_training_options(self, opt):
        """Load architecture-critical flags from the saved training options."""
        train_opt_path = os.path.join(opt.checkpoints_dir, opt.name, 'train_opt.txt')
        if not os.path.exists(train_opt_path):
            return False

        parsed = self._parse_option_file(train_opt_path)
        keys_to_sync = [
            'netG', 'ngf', 'ndf', 'input_nc', 'output_nc', 'norm',
            'use_attention', 'use_gradient_checkpointing',
            'lambda_edge', 'lambda_perc', 'lambda_style'
        ]

        updated_fields = []
        for key in keys_to_sync:
            if key not in parsed or not hasattr(opt, key):
                continue
            current_value = getattr(opt, key)
            new_value = self._cast_option_value(parsed[key], current_value)
            if new_value != current_value:
                setattr(opt, key, new_value)
                updated_fields.append(f'{key}={new_value}')

        if updated_fields:
            print(f"[TestOptions] Synced settings from training options: {', '.join(updated_fields)}")
            return True
        return False

    @staticmethod
    def _parse_option_file(path):
        """Parse the saved option file (train_opt.txt) into a dictionary."""
        options = {}
        with open(path, 'r') as opt_file:
            for line in opt_file:
                if ':' not in line:
                    continue
                key, raw_value = line.split(':', 1)
                key = key.strip()
                if not key or key.startswith('-'):
                    continue
                value = raw_value.split('\t', 1)[0].strip()
                if value:
                    options[key] = value
        return options

    @staticmethod
    def _cast_option_value(value, reference):
        """Convert option values to match the current argparse namespace types."""
        if isinstance(reference, bool):
            return value.lower() in ('true', '1', 'yes', 'y')
        if isinstance(reference, int) and not isinstance(reference, bool):
            try:
                return int(value)
            except ValueError:
                return reference
        if isinstance(reference, float):
            try:
                return float(value)
            except ValueError:
                return reference
        return value
