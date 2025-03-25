import time
from PIL import Image
from typing import Iterable
import util.misc as utils
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from util.metric import nmse, psnr, ssim, AverageMeter

import torch
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from util.vis import vis_img, save_reconstructions

writer = SummaryWriter('./log/tensorboard')

def train_one_epoch(args, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, print_freq: int, device):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ") # delimiter="  " 表示在打印输出时使用两个空格作为分隔符
    # add_meter 方法用于添加一个新的指标监控器，这里添加的是学习率（'lr'）
    # window_size=1 表示每次更新学习率时只考虑最近的一个值
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data in metric_logger.log_every(data_loader, print_freq, header):
        # pdfs:LR_image_m, target, mean, std, fname, slice_num, LR_image   低分辨率掩码图像，不剪裁不掩码图像，低分辨率图像都有
        # pd  :LR_image_m, target, mean, std, fname, slice_num, LR_image   低分辨率掩码图像，不剪裁不掩码图像，低分辨率图像都有
        pd, pdfs, _ = data


        target_pdfs = pdfs[1].unsqueeze(1)
        target_pdfs = target_pdfs.to(device) #



        pd_lr = pd[6].unsqueeze(1) #

        # # ######################## saved_images
        # image_data = pd_lr.squeeze().cpu().numpy()  # 转换为 numpy 数组并去除维度
        #
        # # 创建保存路径
        # save_dir = './saved_image'
        # os.makedirs(save_dir, exist_ok=True)
        #
        # # 使用 matplotlib 保存图像
        # plt.imshow(image_data, cmap='gray')  # 使用灰度图
        # plt.axis('off')  # 关闭坐标轴
        # # 使用 bbox_inches='tight' 去除边框和空白区域
        # plt.savefig(os.path.join(save_dir, 'pd_lr.png'), bbox_inches='tight', pad_inches=0)
        # plt.close()



        pd_lr = pd_lr.to(device) # (4,1,160,160)
        pdfs_lr =pdfs[6].unsqueeze(1) #  两个低分变率图像
        pdfs_lr = pdfs_lr.to(device)

        pdfs_lr_mask = pdfs[0].unsqueeze(1)

        # # ######################## saved_images
        # image_data = pdfs_lr_mask.squeeze().cpu().numpy()  # 转换为 numpy 数组并去除维度
        #
        # # 创建保存路径
        # save_dir = './saved_image'
        # os.makedirs(save_dir, exist_ok=True)
        #
        # # 使用 matplotlib 保存图像
        # plt.imshow(image_data, cmap='gray')  # 使用灰度图
        # plt.axis('off')  # 关闭坐标轴
        # # 使用 bbox_inches='tight' 去除边框和空白区域
        # plt.savefig(os.path.join(save_dir, 'pdfs_lr_mask.png'), bbox_inches='tight', pad_inches=0)
        # plt.close()

        pdfs_lr_mask = pdfs_lr_mask.to(device) # (4,1,160,160)
        #
        # pd_img = pd[1].unsqueeze(1) # (2, 1, 320, 320)
        # pd_img1 = pd[0].unsqueeze(1) # (8,1,160,160)
        # pdfs_img = pdfs[0].unsqueeze(1)
        # target = target.unsqueeze(1) # (8,1,320,320)
        # target = target.unsqueeze(1)
        #
        # pd_img = pd_img.to(device) # (8,1,320,320)
        # pd_img1 = pd_img1.to(device) # (8,1,160,160)
        # pdfs_img = pdfs_img.to(device) # (8,1,160,160)
        # target = target.to(device) # (2,1,320,320)

        if args.USE_MULTI_MODEL and args.USE_CL1_LOSS:
            # outputs, complement = model(pdfs_img, pd_img1) # (2,1,80,80),(2,1,160,160) pdfs经过模型大小不变，是重建；pd经过模型大小变大，是超分
            # 两倍时，输入都是160×160；四倍时，输入80×80
            outputs, complement = model(pd_lr, pdfs_lr_mask) # outputs(160,160);complement(320,320)  # 取pd的低分辨率图像作为辅助图像；取pdfs的低分辨率欠采样图像作为目标图像，做超分辨率重建(超分扩大了四倍)
            loss = criterion(outputs, pdfs_lr, complement, target_pdfs)
        # elif args.USE_MULTI_MODEL:
        #     outputs = model(pdfs_img, pd_img)
        #     loss = criterion(outputs, target)
        # else:
        #     outputs = model(pdfs_img)
        #     loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step() #

        metric_logger.update(loss=loss['loss'])
        metric_logger.update(l1_loss=loss['l1_loss'])
        if args.USE_CL1_LOSS:
            metric_logger.update(cl1_loss = loss['cl1_loss'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    global_step = int(epoch * len(data_loader) + len(data_loader))
    for key, meter in metric_logger.meters.items():
        writer.add_scalar("train/%s" % key, meter.global_avg)

    return {"loss": metric_logger.meters['loss'].global_avg, "global_step": global_step}

@torch.no_grad()
def evaluate(args, model, criterion, data_loader, device, output_dir):

    model.eval()
    criterion.eval()
    criterion.to(device)

    nmse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    output_dic = defaultdict(dict)
    target_dic = defaultdict(dict)
    input_dic = defaultdict(dict)
    start_time = time.time()

    for data in data_loader:

        # LR_image, target, mean, std, fname, slice_num
        # LR_image_m, target, mean, std, fname, slice_num, LR_image
        pd, pdfs, _ = data
        target = pdfs[1]

        mean = pdfs[2]
        std = pdfs[3]

        fname = pdfs[4]
        slice_num = pdfs[5]

        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)

        mean = mean.to(device)
        std = std.to(device)

        pd_lr = pd[6].unsqueeze(1)  #
        pd_lr = pd_lr.to(device)
        pdfs_lr = pdfs[6].unsqueeze(1)  # 两个低分变率图像
        pdfs_lr = pdfs_lr.to(device)

        pdfs_lr_mask = pdfs[0].unsqueeze(1)
        pdfs_lr_mask = pdfs_lr_mask.to(device)
#######################################
        # pd_img = pd[1].unsqueeze(1)
        # pdfs_img = pdfs[0].unsqueeze(1)
        #
        # pd_img = pd_img.to(device)
        # pdfs_img = pdfs_img.to(device)
        target = target.to(device)


        if args.USE_MULTI_MODEL and args.USE_CL1_LOSS:
            #outputs, _ = model(pdfs_img, pd_img)  #  以超分变率的结果计算相关指标
            outputs, complement = model(pd_lr, pdfs_lr_mask)  #  以超分变率的结果计算相关指标


        elif args.USE_MULTI_MODEL:
            outputs, complement = model(pd_lr, pdfs_lr_mask)
        else:
            outputs, complement = model(pdfs_lr_mask)
        # for  qualitative  analysis
        # if k == 0:
        #     outputs = outputs.squeeze().cpu().numpy()  # 使用 .cpu() 将数据移到CPU
        #     complement = complement.squeeze().cpu().numpy()
        #
        #     # 如果是RGB图像，调整通道顺序
        #     if  outputs.ndim == 3:
        #         outputs =  outputs.transpose(1, 2, 0)
        #         complement = complement.transpose(1, 2, 0)
        #     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        #
        #     outputs =  outputs.transpose(2, 0, 1) # (80, 80, 8) -> (8, 80, 80)
        #     complement = complement.transpose( 2, 0, 1)
        #     outputs =  outputs[0]  # 选择第一个通道
        #     complement = complement[0]  # 选择第一个通道
        #
        #     axs[0].imshow(outputs)
        #     axs[0].set_title('Input Image')
        #     axs[0].axis('off')
        #
        #     axs[1].imshow(complement)
        #     axs[1].set_title('Output Image')
        #     axs[1].axis('off')
        #
        #     # 保存图像到指定目录
        #     output_dir = '/home/gegq/Change_CMF/output_image'
        #     os.makedirs(output_dir, exist_ok=True)
        #     plt.savefig(os.path.join(output_dir, f'result_{1}.png'))
        #     plt.close()
        # k+=1



        # outputs = outputs.squeeze(1)
        complement = complement.squeeze(1)

        # outputs = outputs * std + mean
        complement = complement * std + mean
        target = target * std + mean
        # inputs = pdfs_img.squeeze(1) * std + mean
        inputs = pdfs_lr_mask.squeeze(1) * std + mean

        for i, f in enumerate(fname):
            # output_dic[f][slice_num[i]] = outputs[i]
            output_dic[f][slice_num[i]] = complement[i]
            target_dic[f][slice_num[i]] = target[i]
            input_dic[f][slice_num[i]] = inputs[i]


    for name in output_dic.keys():
        f_output = torch.stack([v for _, v in output_dic[name].items()])
        f_target = torch.stack([v for _, v in target_dic[name].items()])
        our_nmse = nmse(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_psnr = psnr(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_ssim = ssim(f_target.cpu().numpy(), f_output.cpu().numpy())

        nmse_meter.update(our_nmse, 1)
        psnr_meter.update(our_psnr, 1)
        ssim_meter.update(our_ssim, 1)

    save_reconstructions(output_dic, output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print("==> Evaluate Metric")
    print("Results ----------")
    print('Evaluate time {}'.format(total_time_str))
    print("NMSE: {:.4}".format(nmse_meter.avg))
    print("PSNR: {:.4}".format(psnr_meter.avg))
    print("SSIM: {:.4}".format(ssim_meter.avg))
    print("------------------")

    return {'NMSE': nmse_meter.avg, 'PSNR': psnr_meter.avg, 'SSIM':ssim_meter.avg}


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]


@torch.no_grad()
def distributed_evaluate(args, model, criterion, data_loader, device, dataset_len):
    model.eval()
    criterion.eval()
    criterion.to(device)

    nmse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    start_time = time.time()

    output_list = []
    target_list = []
    id_list = []
    slice_list = []

    for data in data_loader:
        pd, pdfs, id = data
        target = pdfs[1]

        mean = pdfs[2]
        std = pdfs[3]

        fname = pdfs[4]
        slice_num = pdfs[5]

        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)

        id = id.to(device)

        slice_num = slice_num.to(device)
        mean = mean.to(device)
        std = std.to(device)

        pd_img = pd[1].unsqueeze(1)
        pdfs_img = pdfs[0].unsqueeze(1)

        pd_img = pd_img.to(device)
        pdfs_img = pdfs_img.to(device)
        target = target.to(device)

        if args.USE_MULTI_MODEL and args.USE_CL1_LOSS:
            outputs, _ = model(pdfs_img, pd_img)
        elif args.USE_MULTI_MODEL:
            outputs = model(pdfs_img, pd_img)
        else:
            outputs = model(pdfs_img)
        outputs = outputs.squeeze(1)
        outputs = outputs * std + mean
        target = target * std + mean

        output_list.append(outputs)
        target_list.append(target)
        id_list.append(id)
        slice_list.append(slice_num)

    final_id = distributed_concat(torch.cat((id_list), dim=0), dataset_len)
    final_output = distributed_concat(torch.cat((output_list), dim=0), dataset_len)
    final_target = distributed_concat(torch.cat((target_list), dim=0), dataset_len)
    final_slice = distributed_concat(torch.cat((slice_list), dim=0), dataset_len)

    output_dic = defaultdict(dict)
    target_dic = defaultdict(dict)

    final_id = final_id.cpu().numpy()

    for i, f in enumerate(final_id):
        output_dic[f][final_slice[i]] = final_output[i]
        target_dic[f][final_slice[i]] = final_target[i]

    for name in output_dic.keys():
        f_output = torch.stack([v for _, v in output_dic[name].items()])
        f_target = torch.stack([v for _, v in target_dic[name].items()])
        our_nmse = nmse(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_psnr = psnr(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_ssim = ssim(f_target.cpu().numpy(), f_output.cpu().numpy())

        nmse_meter.update(our_nmse, 1)
        psnr_meter.update(our_psnr, 1)
        ssim_meter.update(our_ssim, 1)



    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print("==> Evaluate Metric")
    print("Results ----------")
    print('Evaluate time {}'.format(total_time_str))
    print("NMSE: {:.4}".format(nmse_meter.avg))
    print("PSNR: {:.4}".format(psnr_meter.avg))
    print("SSIM: {:.4}".format(ssim_meter.avg))
    print("------------------")

    return {'NMSE': nmse_meter.avg, 'PSNR': psnr_meter.avg, 'SSIM':ssim_meter.avg}

def do_vis(dataloader):
    # 从给定的数据加载器 (dataloader) 中获取一批数据，并对其中的一些图像进行预处理和可视化

    for idx, data in enumerate(dataloader):
        pd, pdfs, _ = data

        pd_img, pd_target, pd_mean, pd_std, pd_fname, pd_slice = pd
        pdfs_img, pdfs_target, pdfs_mean, pdfs_std, pdfs_fname, pdfs_slice = pdfs

        pd_mean = pd_mean.unsqueeze(1).unsqueeze(2)
        pd_std = pd_std.unsqueeze(1).unsqueeze(2)

        pdfs_mean = pdfs_mean.unsqueeze(1).unsqueeze(2)
        pdfs_std = pdfs_std.unsqueeze(1).unsqueeze(2)

        pdfs_img = pdfs_img * pdfs_std + pdfs_mean
        pdfs_target = pdfs_target * pdfs_std + pdfs_mean
        pd_target = pd_target * pd_std + pd_mean

        vis_img(pdfs_img.squeeze(0), str(idx), 'pdfs_lr', 'show_rc')
        vis_img(pdfs_target.squeeze(0), str(idx), 'pdfs_target', 'show_rc')
        vis_img(pd_target.squeeze(0), str(idx), 'pd_target', 'show_rc')




