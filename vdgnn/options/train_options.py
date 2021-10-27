from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """Training Options"""

    def initialize(self, parser):
        paser = BaseOptions.initialize(self, parser)
        # Optimization settings
        parser.add_argument_group('Optimization specific arguments')
        parser.add_argument('--num_epochs', default=45, type=int, help='maximum epoch for training')
        parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
        parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
        parser.add_argument('--lr_decay_rate', default=0.9997592083, type=float, help='decay for lr')
        parser.add_argument('--min_lr', default=5e-5, type=float, help='minimum learning rate')
        parser.add_argument('--grad_clip', default=10.0, type=float, help='gradient clipping')
        # Logging settings
        parser.add_argument_group('Logging specific arguments')
        parser.add_argument('--log_step', default=20, type=int, help='save checkpoint after every save_step epochs')
        self.isTrain = True
        return parser
