import os
import re
import time
from typing import Any, Dict, Union

from pytorch_lightning.loggers import WandbLogger
from swanlab.integration.pytorch_lightning import SwanLabLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from common.evaluation import AverageMeter
from common import utils

from pprint import PrettyPrinter

class CustomProgressBar(TQDMProgressBar):
    """
    Custom progress bar for seperated training and validation processes
    """
    def __init__(self):
        super(CustomProgressBar, self).__init__()

    def on_train_epoch_start(self, trainer, *_: Any):
        super(CustomProgressBar, self).on_train_epoch_start(trainer)
        self.train_progress_bar.set_description(f"[trn] ep: {trainer.current_epoch:>3}")

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

class MeterCallback(Callback):
    """
    A class that initiates classificaiton and segmentation metrics
    """
    def __init__(self, args):
        super(MeterCallback, self).__init__()
        self.args = args

    def global_init(self, pl_module: pl.LightningModule):
        if self.args.data_set == 'pascal':
            pl_module.novel_label_num = 5       
            pl_module.base_label_num = pl_module.novel_label_num*3+1       # 5*3+1, include background

        elif self.args.data_set == 'coco':
            pl_module.novel_label_num = 20       
            pl_module.base_label_num = pl_module.novel_label_num*3+1       # 5*3+1, include background

        elif self.args.data_set == 'FSS1000':
            pl_module.novel_label_num = 240
            pl_module.base_label_num = 520

        elif self.args.data_set == 'pascal_part':
            total = 57      # total part classes, not include background
            if self.args.fold == 0:
                pl_module.novel_label_num = 31
            elif self.args.fold == 1:
                pl_module.novel_label_num = 4
            elif self.args.fold == 2:
                pl_module.novel_label_num = 5
            elif self.args.fold == 3:
                pl_module.novel_label_num = 16
            pl_module.base_label_num = total - pl_module.novel_label_num + 1        # 1 for background

        elif self.args.data_set == 'paco_part':
            if self.args.shot == 5:
                total = 199      # total part classes, not include background
                if self.args.fold == 0:
                    pl_module.novel_label_num = 39
                elif self.args.fold == 1:
                    pl_module.novel_label_num = 44
                elif self.args.fold == 2:
                    pl_module.novel_label_num = 32
                elif self.args.fold == 3:
                    pl_module.novel_label_num = 33
                pl_module.base_label_num = total - pl_module.novel_label_num + 1        # 1 for background
            elif self.args.shot == 1:
                total = 303      # total part classes, not include background
                if self.args.fold == 0:
                    pl_module.novel_label_num = 79
                elif self.args.fold == 1:
                    pl_module.novel_label_num = 82
                elif self.args.fold == 2:
                    pl_module.novel_label_num = 66
                elif self.args.fold == 3:
                    pl_module.novel_label_num = 76
                pl_module.base_label_num = total - pl_module.novel_label_num + 1        # 1 for background
            else:
                raise ValueError('Warning: For shots other than 1 and 5, the number of novel classes needs to be recalculated')
        elif self.args.data_set == 'lvis':
            if self.args.shot == 1:
                pl_module.novel_label_num = 92
                pl_module.base_label_num = pl_module.novel_label_num*9+1       # 5*3+1, include background
            elif self.args.shot == 5:
                pl_module.novel_label_num = 68
                pl_module.base_label_num = pl_module.novel_label_num*9+1       # 5*3+1, include background

        else:
            raise ValueError('Unknown dataset: {}'.format(self.args.data_set))

        pl_module.val_num = 0
        pl_module.best_miou = 0.
        pl_module.best_FBiou = 0.
        pl_module.best_piou = 0.
        pl_module.best_epoch = 0
        pl_module.keep_epoch = 0
        pl_module.best_miou_m = 0.
        pl_module.best_miou_b = 0.
        pl_module.best_FBiou_m = 0.
        pl_module.start_time = time.time()

    def on_fit_start(self, trainer, pl_module):
        PrettyPrinter().pprint(vars(self.args))
        utils.print_param_count(pl_module)

        if not self.args.nowandb and not self.args.eval and isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.watch(pl_module)

        self.global_init(pl_module)

    def on_test_start(self, trainer, pl_module):
        PrettyPrinter().pprint(vars(self.args))
        self.global_init(pl_module)
        utils.print_param_count(pl_module)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        print(f'\n----- ep: {pl_module.current_epoch+1:>3}-----')
        utils.fix_randseed(None)
        pl_module.train_average_meter = {
            'batch_time': AverageMeter(),
            'data_time': AverageMeter(),
            'main_loss_meter': AverageMeter(),
            'aux_loss_meter1': AverageMeter(),
            'aux_loss_meter2': AverageMeter(),
            'loss_meter': AverageMeter(),
            'intersection_meter': AverageMeter(),
            'union_meter': AverageMeter(),
            'target_meter': AverageMeter(),
            'proto_loss_meter': AverageMeter(),
        }

        for cross_num in range(pl_module.cross_num):
            pl_module.train_average_meter[f'mask_ratio_c{cross_num}_meter'] = AverageMeter()

        pl_module.trn_end = time.time()
        pl_module.val_time = 0.

        pl_module.train_mode()

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.test_num = len(trainer.val_dataloaders) * self.args.batch_size
        self._shared_eval_epoch_start(pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.test_num = len(trainer.test_dataloaders) * self.args.batch_size
        self._shared_eval_epoch_start(pl_module)

    def _shared_eval_epoch_start(self, pl_module):
        utils.fix_randseed(0)
        pl_module.val_average_meter = {
            'batch_time': AverageMeter(),
            'data_time': AverageMeter(),
            'model_time': AverageMeter(),
            'loss_meter': AverageMeter(),
            'intersection_meter': AverageMeter(),
            'union_meter': AverageMeter(),
            'target_meter': AverageMeter(),
            'intersection_meter_m': AverageMeter(),     # meta
            'union_meter_m': AverageMeter(),
            'target_meter_m': AverageMeter(),
            'proto_loss_meter': AverageMeter(),
            'sim_meter': AverageMeter()
        }

        pl_module.val_num += 1

        pl_module.class_intersection_meter = [0]*pl_module.novel_label_num
        pl_module.class_union_meter = [0]*pl_module.novel_label_num  
        pl_module.class_intersection_meter_m = [0]*pl_module.novel_label_num
        pl_module.class_union_meter_m = [0]*pl_module.novel_label_num
        pl_module.class_intersection_meter_b = [0]*(pl_module.base_label_num-1)
        pl_module.class_union_meter_b = [0]*(pl_module.base_label_num-1)
        pl_module.class_target_meter_b = [0]*(pl_module.base_label_num-1)

        pl_module.val_start = time.time()
        pl_module.val_end = time.time()
        pl_module.eval()

class CustomCheckpoint(ModelCheckpoint):
    """
    Checkpoint load & save
    """
    def __init__(self, args):
        dirpath = os.path.join(args.logpath, args.vis_name, 'model')
        if args.eval or args.resume:
            assert os.path.exists(dirpath), f'{dirpath} not exists'
        self.filename = 'best_model'

        super(CustomCheckpoint, self).__init__(dirpath=dirpath,
                                               monitor='val/mIoU',
                                               filename=self.filename,
                                               mode='max',
                                               verbose=True,
                                               save_last=True)
        self.best_modelpath, self.last_modelpath = self.return_model_path(self.dirpath)


        # For evaluation, load best_model-v(k).cpkt where k is the max index
        if args.eval:
            print('evaluating', self.best_modelpath)
        # For training, set the filename as best_model.ckpt
        # For resuming training, pytorch_lightning will automatically set the filename as best_model-v(k).ckpt
        elif args.resume:
            print('resuming', self.best_modelpath)


    def return_model_path(self, dirpath):
        ckpt_files = os.listdir(dirpath)  # list of strings
        vers = [ckpt_file for ckpt_file in ckpt_files if 'best_model' in ckpt_file]
        # 使用正则表达式提取版本号
        version_pattern = re.compile(r'best_model-v(\d+)\.ckpt')
        versions = []

        for checkpoint in vers:
            match = version_pattern.match(checkpoint)
            if match:
                versions.append(int(match.group(1)))

        # 找到最大的版本号
        if versions:
            max_version = max(versions)
            best_model = f'best_model-v{max_version}.ckpt'
        else:
            best_model = 'best_model.ckpt'  # 如果没有版本号，返回默认的文件名

        ckpt_files = os.listdir(dirpath)  # list of strings
        vers = [ckpt_file for ckpt_file in ckpt_files if 'last' in ckpt_file]
        # 使用正则表达式提取版本号
        version_pattern = re.compile(r'last-v(\d+)\.ckpt')
        versions = []

        for checkpoint in vers:
            match = version_pattern.match(checkpoint)
            if match:
                versions.append(int(match.group(1)))

        # 找到最大的版本号
        if versions:
            max_version = max(versions)
            last_model = f'last-v{max_version}.ckpt'
        else:
            last_model = 'last.ckpt'  # 如果没有版本号，返回默认的文件名

        return os.path.join(self.dirpath, best_model), os.path.join(self.dirpath, last_model)


# class OnlineLogger(WandbLogger):
class OnlineLogger(SwanLabLogger):
    """
    A wandb logger class that is customed with the experiment log path
    """
    def __init__(self, args):
        super(OnlineLogger, self).__init__(
            name=args.description,
            project=f'{args.arch}-{args.data_set}-{args.backbone}',
            group=f'{args.description}',
            log_model=False,
            save_dir=args.logpath,
        )
        self.experiment.config.update(args)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        # 确保仅主进程记录指标
        super().log_metrics(metrics, step)