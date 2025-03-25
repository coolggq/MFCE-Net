
import torch
import time
import os
import datetime
import random
import numpy as np
import argparse
from models import build_model_from_name
from models.loss import build_criterion
from data.fastmri import build_dataset
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from engine import train_one_epoch, evaluate, distributed_evaluate, do_vis
from util.misc import init_distributed_mode, get_rank, save_on_master
from config import build_config
import os
from util.misc import WarmUpPolyLR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
torch.cuda.set_device(1)

def main(args, work):

    init_distributed_mode(args)

    # build criterion and model first
    model = build_model_from_name(args, work)


    criterion = build_criterion(args) # 根据提供的参数构建损失函数

    start_epoch = 0

    seed = args.SEED + get_rank()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 强制使用 CPU 设备
    # model.to(device)

    # device = torch.device(args.SOLVER.DEVICE)
    device = torch.device("cuda:1")

    model.to(device)
    criterion.to(device)

    if args.distributed:
        # 如果启用了分布式训练，通过 torch.nn.parallel.DistributedDataParallel 封装模型，
        # 以支持在多个GPU上并行训练，device_ids=[args.gpu] 指定使用的GPU。
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: %.2f M' % (n_parameters / 1024 / 1024))

    # build optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.SOLVER.LR, momentum=0.9, weight_decay=args.SOLVER.WEIGHT_DECAY)




    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.SOLVER.LR_DROP, gamma=0.1)

    # build dataset
    dataset_train = build_dataset(args, mode='train')
    dataset_val = build_dataset(args, mode='val')

    dataset_val_len = len(dataset_val)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.SOLVER.BATCH_SIZE, drop_last=True)

    dataloader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                  num_workers=args.SOLVER.NUM_WORKERS, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.SOLVER.BATCH_SIZE,
                                sampler=sampler_val, num_workers=args.SOLVER.NUM_WORKERS,
                                pin_memory=True)

    # #lr_scheduler = WarmUpPolyLR(optimizer,
    #                          start_lr=args.SOLVER.LR,  # 初始学习率。在训练开始时使用的学习
    #                          lr_power=0.9,  # 多项式衰减的幂次。控制学习率衰减的速度
    #                          total_iters=args.TRAIN.EPOCHS,  # 总的迭代次数。训练过程中总的迭代次数
    #                          warmup_steps=len(dataloader_train) * 2)

    if args.RESUME != '':
        checkpoint = torch.load(args.RESUME)
        checkpoint = checkpoint['model']
        checkpoint = {key.replace("module.", ""): val for key, val in checkpoint.items()}
        print('resume from %s' % args.RESUME)
        model.load_state_dict(checkpoint, strict=False)


    start_time = time.time()

    best_status = {'NMSE': 10000000, 'PSNR': 0, 'SSIM': 0}

    best_checkpoint = None

    for epoch in range(start_epoch, args.TRAIN.EPOCHS):
        train_status = train_one_epoch(args,
            model, criterion, dataloader_train, optimizer, epoch, args.SOLVER.PRINT_FREQ, device)
        lr_scheduler.step()    # args.SOLVER.PRINT_FREQ 是一个通常用于控制在训练过程中打印进度信息的参数。
                                # 具体来说，它决定了每隔多少个训练步（或轮次）打印一次训练状态或进度信息。

        if args.distributed:
            eval_status = distributed_evaluate(args, model, criterion, dataloader_val, device, dataset_val_len)
        else:
            eval_status = evaluate(args, model, criterion, dataloader_val, device, args.OUTPUTDIR)

        if eval_status['PSNR']>best_status['PSNR']:
            best_status = eval_status
            best_checkpoint = {
                'model': model.state_dict(), # model.module.state_dict(), model.module 只有在使用 torch.nn.DataParallel 或 torch.nn.DistributedDataParallel 进行并行化训练时才会出现。
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }

        # save model
        if args.OUTPUTDIR:
            Path(args.OUTPUTDIR).mkdir(parents=True, exist_ok=True)
            checkpoint_path = os.path.join(args.OUTPUTDIR, f'checkpoint{epoch:04}.pth') # 每个epoch的参数都保存下来

            if args.distributed:
                save_on_master({
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            else:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

    print('The bset epoch is ', best_checkpoint['epoch'])
    print("Results ----------")
    print("NMSE: {:.4}".format(best_status['NMSE']))
    print("PSNR: {:.4}".format(best_status['PSNR']))
    print("SSIM: {:.4}".format(best_status['SSIM']))
    print("------------------")
    if args.OUTPUTDIR:
        checkpoint_path = os.path.join(args.OUTPUTDIR, 'best.pth')

        if args.distributed:
            save_on_master(best_checkpoint, checkpoint_path)
        else:
            torch.save(best_checkpoint, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="a unit Cross Multi modity transformer")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument(
        "--experiment", default="sr_multi_CN", help="choose a experiment to do") # sr_multi_cross
    args = parser.parse_args()

    print('doing ', args.experiment)

    cfg = build_config(args.experiment)

    print(cfg)

    main(cfg, args.experiment)





