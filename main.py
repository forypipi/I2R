import logging
from pathlib import Path
import os, sys
import numpy as np
import torch
import argparse
import time
import pytorch_lightning as pl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)


from model import BAM
from data.dataset import FSSDatasetModule
from common.callbacks import MeterCallback, CustomProgressBar, CustomCheckpoint, OnlineLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateMonitor, LearningRateFinder
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner
from torch.utils.data import DataLoader

from util import transform, transform_tri, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, get_logger, get_save_path, \
                                    is_same_model, fix_bn, sum_list, check_makedirs
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def get_parser(args):
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def main(args):

    job_id = os.getenv('SLURM_JOB_ID')
    if job_id:
        print(f"SLURM Job ID: {job_id}")
    else:
        print("Not running in a SLURM job.")

    gpu_count = torch.cuda.device_count()

    # 打印每块GPU的型号
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        current_device = os.environ['CUDA_VISIBLE_DEVICES']
        print(f"current GPU {i}: {props.name} {current_device}")
        
    args = get_parser(args)
    check(args)


    args.backbone = 'vgg16' if args.vgg else f'resnet{args.layers}'
    args.split = args.fold
    args.logpath = os.path.join(args.logpath, f'{args.arch}_{args.backbone}_{args.data_set}_{args.shot}s_{args.split}split')
    mkdirs_path = os.path.join(args.logpath, args.vis_name, 'model')
    if not os.path.exists(mkdirs_path):
        os.makedirs(mkdirs_path, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val/mIoU", mode="max", min_delta=0.0, patience=args.stop_interval),   # check after all val turns, check val /(5 training), early stop after 75 epochs 
        MeterCallback(args),
        CustomCheckpoint(args),
        # StochasticWeightAveraging(swa_lrs=args.base_lr/100, swa_epoch_start=0.8, annealing_epochs=20),       # line 258  trainer.fit_loop._skip_backward = True    # official is false, manually change to true
        # LearningRateFinder(),
        # CustomProgressBar(),
        # ModelSummary(max_depth=-1)
    ]

    if not (args.nowandb or args.eval):
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    # Pytorch-lightning main trainer
    checkpoint_callback = CustomCheckpoint(args)

    if not args.eval:
        if args.resume:
            # Loading the best model checkpoint from args.logpath
            model = BAM.OneModel.load_from_checkpoint(checkpoint_callback.best_modelpath, args=args, strict=False)
        else:
            model = BAM.OneModel(args, cls_type='Base')


    print(args)
    
    if not args.eval:
        # Train
        dm = FSSDatasetModule(args)
        trainer = pl.Trainer(
            accelerator='gpu',
            strategy='ddp_find_unused_parameters_true',
            callbacks=callbacks,
            logger=False if args.nowandb or args.eval else OnlineLogger(args),
            num_nodes=1,        # gpu
            default_root_dir=args.logpath,
            min_epochs=1,
            max_epochs=args.epochs,
            num_sanity_val_steps=0,
            fast_dev_run=1 if args.debug else False,
            precision='bf16',
            enable_progress_bar=False,
            inference_mode=False,
            val_check_interval=args.val_check_interval if args.val_check_interval is not None else 1.0,  # check val twice in 1 epoch
            # profiler='advanced',  # see each module running time, only need when first run to check bottleneck; simple for each stage, advanced for each function
            # limit_train_batches=10,
            # limit_val_batches=10,
            # limit_test_batches=2,
            )
    
        trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    else:

        # args.batch_size = 1
        setup_seed(args.val_num, False)
        seed_array = np.random.randint(0, 1000, args.val_num)
        test_list = []
        model = BAM.OneModel.load_from_checkpoint(checkpoint_callback.best_modelpath, args=args, strict=False)

        print(model)
        for i, val_id in enumerate(range(args.val_num)):
            val_seed = int(seed_array[val_id])

            print(f'\n\n\ntime {i+1}, testing seed: {val_seed}')
            setup_seed(val_seed, args.seed_deterministic)
            dm = FSSDatasetModule(args)

            trainer = pl.Trainer(
                accelerator='gpu',
                strategy='ddp_find_unused_parameters_true',
                callbacks=callbacks,
                logger=False,
                num_nodes=1,        # gpu
                default_root_dir=args.logpath,
                min_epochs=1,
                num_sanity_val_steps=0,
                precision='bf16',
                fast_dev_run=1 if args.debug else False,
                enable_progress_bar=False,
                inference_mode=False,
                # limit_test_batches=2,
                )

            test_log = trainer.test(model, dataloaders=dm.test_dataloader())
            test_log[0]['seed'] = val_seed
            test_list.extend(test_log)
        # test_list = [
        #     {'val/Loss': 0.12448923289775848, 'val/FBIoU': 83.00249481201172, 'val/FBIoU_m': 79.21916198730469, 'val/mIoU': 70.01322937011719, 'val/mIoU_m': 66.45452880859375, 'val/sim': 0.24637603759765625, 'val/best_mIoU': 70.01322937011719, 'seed': 2},
        #     # 其他 9 个序列...
        # ]

        # 初始化最大值和对应的 seed
        max_fbiou = -np.inf
        max_miou_m = -np.inf
        max_miou = -np.inf
        max_fbiou_seed = None
        max_miou_m_seed = None
        max_miou_seed = None

        # 用于计算平均值和标准差的列表
        fbiou_values = []
        miou_m_values = []
        miou_values = []
        PFsim_values = []

        # 遍历数据
        for entry in test_list:
            fbiou = entry['val/FBIoU']
            miou_m = entry['val/mIoU_m']
            miou = entry['val/mIoU']
            seed = entry['seed']
            sim = entry['val/sim']

            # 更新最大值和对应的 seed
            if fbiou > max_fbiou:
                max_fbiou = fbiou
                max_fbiou_seed = seed
            if miou_m > max_miou_m:
                max_miou_m = miou_m
                max_miou_m_seed = seed
            if miou > max_miou:
                max_miou = miou
                max_miou_seed = seed

            # 保存值用于后续计算
            fbiou_values.append(fbiou)
            miou_m_values.append(miou_m)
            miou_values.append(miou)
            PFsim_values.append(sim)

        # 计算平均值和标准差
        avg_fbiou = np.mean(fbiou_values)
        std_fbiou = np.std(fbiou_values)

        avg_miou_m = np.mean(miou_m_values)
        std_miou_m = np.std(miou_m_values)

        avg_miou = np.mean(miou_values)
        std_miou = np.std(miou_values)

        avg_PFsim = np.mean(PFsim_values)
        std_PFsim = np.std(PFsim_values)

        # 输出结果
        print('\n\n\n<<<<<<<<<<<<<<<<< Summary <<<<<<<<<<<<<<<<<')
        print(f'Max FBIoU: {max_fbiou:.4f} (seed: {max_fbiou_seed})')
        print(f'Max mIoU_m: {max_miou_m:.4f} (seed: {max_miou_m_seed})')
        print(f'Max mIoU: {max_miou:.4f} (seed: {max_miou_seed})')
        print(f'Average FBIoU: {avg_fbiou:.4f}, Std: {std_fbiou:.4f}')
        print(f'Average mIoU_m: {avg_miou_m:.4f}, Std: {std_miou_m:.4f}')
        print(f'Average mIoU: {avg_miou:.4f}, Std: {std_miou:.4f}')
        print(f'Average PF_sim: {avg_PFsim:.6f}, Std: {std_PFsim:.6f}')

        # Best_Seed_m: 988 	 Best_Seed_F: 988 	 Best_Seed_p: 988
        # Best_mIoU: 0.4188 	 Best_mIoU_m: 0.3954 	 Best_FBIoU: 0.6648 	 Best_FBIoU_m: 0.6265 	 Best_pIoU: 0.4474
        # Mean_mIoU: 0.3914 	 Mean_mIoU_m: 0.3727 	 Mean_FBIoU: 0.6575 	 Mean_FBIoU_m: 0.6189 	 Mean_pIoU: 0.4377

def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    # assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0

if __name__ == '__main__':
    # Arguments parsing
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    parser = argparse.ArgumentParser(description='Template for FSS using Pytorch Lightning')
    parser.add_argument('--description', type=str, required=True, help='description for your experiment')
    parser.add_argument('--fold', type=int, required=True, help='fold x')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--method', type=str, required=True, help='method for CAM')
    parser.add_argument('--vis_name', type=str, required=True, help='log name, use to save model and vis')
    parser.add_argument('--threshold', default='auto', help='threshold for CAM to Mask')
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--eval', action='store_true', help='Flag to evaluate a model checkpoint')
    parser.add_argument('--debug', action='store_true', help='Flag to debug')
    parser.add_argument('--nowandb', action='store_true', help='Flag not to log at wandb')

    args = parser.parse_args()
    main(args)