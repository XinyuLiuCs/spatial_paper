import os
import json
import argparse
import threading
import subprocess
from tqdm import tqdm
from datetime import datetime
from pytorch_msssim import ssim
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter
from dataset.loader import get_dataloader
from model import TotalLoss, SlimEnhancer


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='paper_1')
parser.add_argument('--cnum', default=5, type=int, help='curve group number')
parser.add_argument('--cnode', default=64, type=int, help='curve node')
parser.add_argument('--eval_freq', default=1, type=int, help='frequency of valid')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--epochs', default=200, type=int, help='number of train epochs')
parser.add_argument('--batch_size', default=3, type=int, help='9, 12, 15')
# dir
parser.add_argument('--is_save_model', default=True, action='store_true', help='save the enhanced results')
parser.add_argument('--data_dir', default='./data', type=str, help='path to data')
parser.add_argument('--save_result', default='./result/{exp_name}/', type=str, help='path to training log saving')
parser.add_argument('--save_model', default='./model/{exp_name}/', type=str, help='path to models saving')
# loss
parser.add_argument('--use_l1', action='store_true', default=False, help='Use L1 loss')
parser.add_argument('--l1_weight', type=float, default=1.0, help='Weight for L1 loss')
parser.add_argument('--use_mse', action='store_true', default=True, help='Use MSE loss')
parser.add_argument('--mse_weight', type=float, default=1.0, help='Weight for MSE loss')
parser.add_argument('--use_color', action='store_true', default=False, help='Use color loss')
parser.add_argument('--color_weight', type=float, default=0.05, help='Weight for color loss')
parser.add_argument('--use_tv', action='store_true', default=False, help='Use total variation loss')
parser.add_argument('--tv_weight', type=float, default=0.1, help='Weight for total variation loss')
parser.add_argument('--use_ssim', action='store_true', default=True, help='Use SSIM loss')
parser.add_argument('--ssim_weight', type=float, default=0.2, help='Weight for SSIM loss')
parser.add_argument('--use_deltaE', action='store_true', default=False, help='Use DeltaE loss')
parser.add_argument('--deltaE_weight', type=float, default=0.2, help='Weight for DeltaE loss')
# clip
parser.add_argument('--clip_model', default='ViT-L-14-336px', type=str)
parser.add_argument('--clip_model_path', default='./clip_model', type=str)
parser.add_argument('--clip_learning_rate', default=1e-5, type=float, help='Learning rate for CLIP fine-tuning')
parser.add_argument('--use_text_features', action='store_true', default=True, help='Use text features from CLIP')
parser.add_argument('--use_image_features', action='store_true', default=True, help='Use image features from CLIP')
parser.add_argument('--fusion_method', type=str, default='multiply', choices=['concat', 'add', 'multiply', 'attention','cross_attention'], help='Method to fuse image and text features')
# style
parser.add_argument('--csv_path', type=str, default='./style data', help='path to csv file')
parser.add_argument('--input_styles', type=list, default=['06-Input-ExpertC1.5'], help='input styles')
parser.add_argument('--gt_styles', type=list, default=['01-Experts-A', '02-Experts-B', '03-Experts-C', '04-Experts-D', '05-Experts-E'], help='gt styles')
parser.add_argument('--valid_input_style', type=str, default=['06-Input-ExpertC1.5'], help='validation input style')
parser.add_argument('--valid_gt_style', type=str, default=['01-Experts-A', '02-Experts-B', '03-Experts-C', '04-Experts-D', '05-Experts-E'], help='validation gt style')
# TensorBoard
parser.add_argument('--use_tensorboard', action='store_true', default=False, help='Enable TensorBoard logging')
args = parser.parse_args()


def run_tensorboard(logdir):
    subprocess.run(["tensorboard", "--logdir", logdir, "--port", "6010", "--host", "0.0.0.0"])

def train(train_loaders, network, criterion, optimizer, pbar, writer, global_step):
    losses = AverageMeter()
    loss_breakdown_avg = defaultdict(AverageMeter)

    network.train()

    for loader in train_loaders:
        for (input_img, target_img, text, img_name) in loader:
            input_img = input_img.cuda(non_blocking=True)
            target_img = target_img.cuda(non_blocking=True)
            output = network(input_img, text)

            loss, loss_breakdown = criterion(output, target_img)
            losses.update(loss.item(), input_img.size(0))
            for loss_name, loss_value in loss_breakdown.items():
                loss_breakdown_avg[loss_name].update(loss_value, input_img.size(0))
            
            if args.use_tensorboard and writer:
                writer.add_scalar('Train/TotalLoss', loss.item(), global_step)
                for loss_name, loss_value in loss_breakdown.items():
                    writer.add_scalar(f'Train/{loss_name}', loss_value, global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            pbar.update(1)

            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})

    final_loss_breakdown = {k: v.avg for k, v in loss_breakdown_avg.items()}
    return losses.avg, final_loss_breakdown, global_step

def valid(val_loader, network, writer, epoch):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for (input_img, target_img, text, img_name) in val_loader:
        input_img = input_img.cuda(non_blocking=True)
        target_img = target_img.cuda(non_blocking=True)

        with torch.no_grad():
            output = network(input_img, text).clamp_(0, 1)

        mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
        psnr_val = 10 * torch.log10(1 / mse_loss).mean().item()
        ssim_val = ssim(output, target_img, data_range=1, size_average=False).item()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        if args.use_tensorboard and writer:
            writer.add_scalar('Val/PSNR', PSNR.avg, epoch)
            writer.add_scalar('Val/SSIM', SSIM.avg, epoch)
    return PSNR.avg, SSIM.avg

def valid_all_styles(network, writer, epoch, args):
    results = []
    avg_psnr_sum, avg_ssim_sum = 0, 0
    input_styles = args.valid_input_style if isinstance(args.valid_input_style, list) else [args.valid_input_style]
    gt_styles = args.valid_gt_style if isinstance(args.valid_gt_style, list) else [args.valid_gt_style]
    cur_06to03_psnr, cur_06to03_ssim = 0, 0
    psnr_dict, ssim_dict = {}, {}
    log_lines = []
    for input_style in input_styles:
        for gt_style in gt_styles:
            args.input_styles = [input_style]
            args.gt_styles = [gt_style]
            val_loader = get_dataloader('test', args)
            psnr, ssim_val = valid(val_loader, network, writer, epoch)
            results.append((input_style, gt_style, psnr, ssim_val))
            avg_psnr_sum += psnr
            avg_ssim_sum += ssim_val
            psnr_dict[(input_style, gt_style)] = psnr
            ssim_dict[(input_style, gt_style)] = ssim_val
            line = f"Val {input_style} -> {gt_style}: PSNR={psnr:.4f}, SSIM={ssim_val:.4f}"
            print(line)
            log_lines.append(line)
            if input_style == '06-Input-ExpertC1.5' and gt_style == '03-Experts-C':
                cur_06to03_psnr = psnr
                cur_06to03_ssim = ssim_val
    avg_psnr = avg_psnr_sum / (len(input_styles) * len(gt_styles))
    avg_ssim = avg_ssim_sum / (len(input_styles) * len(gt_styles))
    avg_line = f"Val AVG: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}"
    print(avg_line)
    log_lines.append(avg_line)
    return results, avg_psnr, avg_ssim, cur_06to03_psnr, cur_06to03_ssim, log_lines


if __name__ == '__main__':

    args.save_model = args.save_model.format(exp_name=args.exp_name)
    args.save_result = args.save_result.format(exp_name=args.exp_name)
    os.makedirs(args.save_model, exist_ok=True)
    print(f"model save: {args.save_model}")
    os.makedirs(args.save_result, exist_ok=True)
    print(f"result save: {args.save_result}")

    args_dict = vars(args)
    args_str = json.dumps(args_dict, indent=4)
    print(args_str)
    log_file_path = os.path.join(args.save_result, 'training_log.txt')
    with open(log_file_path, "a") as log_file:
        log_file.write(args_str + "\n")

    writer = None
    tensorboard_thread = None
    if args.use_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(args.save_result, 'tensorboard_logs'))
        tensorboard_thread = threading.Thread(target=run_tensorboard, args=(os.path.join(args.save_result, 'tensorboard_logs'),))
        tensorboard_thread.start()
        print("TensorBoard started. Open http://localhost:6010/ in your browser to view it.")

    network = SlimEnhancer(args)
    network.cuda()

    criterion = TotalLoss(args)

    train_loaders = get_dataloader('train', args)
    args.input_styles = [args.valid_input_style]

    clip_params = []
    trainable_parts = []

    if args.use_image_features and hasattr(network, 'clip_visual'):
        clip_params.extend(network.clip_visual.parameters())
        trainable_parts.append('CLIP Visual')

    if args.use_text_features and hasattr(network, 'clip_model'):
        text_parts = []
        if hasattr(network.clip_model, 'transformer'):
            clip_params.extend(network.clip_model.transformer.parameters())
            text_parts.append('transformer')
        if hasattr(network.clip_model, 'token_embedding'):
            clip_params.extend(network.clip_model.token_embedding.parameters())
            text_parts.append('token_embedding')
        if hasattr(network.clip_model, 'positional_embedding'):
            clip_params.append(network.clip_model.positional_embedding)
            text_parts.append('positional_embedding')
        if hasattr(network.clip_model, 'ln_final'):
            clip_params.extend(network.clip_model.ln_final.parameters())
            text_parts.append('ln_final')
        if hasattr(network.clip_model, 'text_projection'):
            clip_params.append(network.clip_model.text_projection)
            text_parts.append('text_projection')
        if text_parts:
            trainable_parts.append(f"CLIP Text ({', '.join(text_parts)})")

    other_params = list(network.spatial_net.parameters())
    trainable_parts.append('Restormer')

    other_params.extend(network.curve_mapper.parameters())
    trainable_parts.append('Curve Mapper')

    if hasattr(network, 'enhanced_cross_attention'):
        other_params.extend(network.enhanced_cross_attention.parameters())
        trainable_parts.append('Enhanced Cross Attention')

    if hasattr(network, 'cross_attention'):
        other_params.extend(network.cross_attention.parameters())
        trainable_parts.append('Cross Attention')

    optimizer = torch.optim.Adam([
        {'params': clip_params, 'lr': args.clip_learning_rate},
        {'params': other_params, 'lr': args.lr}
    ])

    print(f"Trainable parts: {', '.join(trainable_parts)}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    all_param_ids = set(id(p) for p in network.parameters() if p.requires_grad)
    grouped_param_ids = set(id(p) for p in clip_params + other_params)
    unmanaged_count = len(all_param_ids - grouped_param_ids)
    print('unmanaged_count:', unmanaged_count)
    if unmanaged_count > 0:
        for name, param in network.named_parameters():
            if param.requires_grad and id(param) not in grouped_param_ids:
                print(f"unmanaged_params: {name}, shape: {param.shape}")

    global_step = 0
    best_psnr_avg = 0
    best_ssim_avg = 0
    best_psnr_avg_epoch = 0
    best_ssim_avg_epoch = 0
    best_06to03_psnr = 0
    best_06to03_ssim = 0
    best_06to03_psnr_epoch = 0
    best_06to03_ssim_epoch = 0

    total_batches = sum([len(loader) for loader in train_loaders])

    for epoch in range(args.epochs + 1):
        with tqdm(total=total_batches, desc=f"Epoch {epoch}/{args.epochs}", unit='batch') as pbar:
            loss, loss_breakdown, global_step = train(train_loaders, network, criterion, optimizer, pbar, writer, global_step)

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        current_time = datetime.now().strftime('%m-%d %H:%M')
        log_message = f"{args.exp_name}, {current_time}, {epoch}/{args.epochs}, LR:{current_lr:.3e}, Loss:{loss:.4f}\n"

        for loss_name, loss_value in loss_breakdown.items():
            log_message += f"{loss_name}: {loss_value:.4f}, "
        log_message = log_message.rstrip(', ') + '\n'

        print(log_message.strip())
        with open(log_file_path, "a") as log_file:
            log_file.write(log_message)

        if epoch % args.eval_freq == 0:
            results, avg_psnr, avg_ssim, cur_06to03_psnr, cur_06to03_ssim, log_lines = valid_all_styles(network, writer, epoch, args)
            with open(log_file_path, "a") as log_file:
                for line in log_lines:
                    log_file.write(line + "\n")
            if avg_psnr > best_psnr_avg:
                best_psnr_avg = avg_psnr
                best_psnr_avg_epoch = epoch
                network_save_path = os.path.join(args.save_model, f"enhancer_psnr_avg.pth")
                if args.is_save_model:
                    torch.save({'state_dict': network.state_dict()}, network_save_path)
            if avg_ssim > best_ssim_avg:
                best_ssim_avg = avg_ssim
                best_ssim_avg_epoch = epoch
                network_save_path = os.path.join(args.save_model, f"enhancer_ssim_avg.pth")
                if args.is_save_model:
                    torch.save({'state_dict': network.state_dict()}, network_save_path)
            if cur_06to03_psnr > best_06to03_psnr:
                best_06to03_psnr = cur_06to03_psnr
                best_06to03_psnr_epoch = epoch
                network_save_path = os.path.join(args.save_model, f"enhancer_psnr_06to03.pth")
                if args.is_save_model:
                    torch.save({'state_dict': network.state_dict()}, network_save_path)
            if cur_06to03_ssim > best_06to03_ssim:
                best_06to03_ssim = cur_06to03_ssim
                best_06to03_ssim_epoch = epoch
                network_save_path = os.path.join(args.save_model, f"enhancer_ssim_06to03.pth")
                if args.is_save_model:
                    torch.save({'state_dict': network.state_dict()}, network_save_path)
            summary = f"Val:{epoch}/{args.epochs}, BestP-AVG:{best_psnr_avg:.2f}({best_psnr_avg_epoch}), NowP-AVG:{avg_psnr:.2f}, BestS-AVG:{best_ssim_avg:.4f}({best_ssim_avg_epoch}), NowS-AVG:{avg_ssim:.4f}\n"
            summary += f"BestP-06to03:{best_06to03_psnr:.2f}({best_06to03_psnr_epoch}), NowP-06to03:{cur_06to03_psnr:.2f}, BestS-06to03:{best_06to03_ssim:.4f}({best_06to03_ssim_epoch}), NowS-06to03:{cur_06to03_ssim:.4f}"
            print(summary)
            with open(log_file_path, "a") as log_file:
                log_file.write(summary + "\n")

        if args.use_tensorboard and writer:
            writer.add_scalar('Train/LearningRate', scheduler.get_last_lr()[0], epoch)

    if args.use_tensorboard and writer:
        writer.close()
    
    if args.use_tensorboard:
        print("Training finished. Press Ctrl+C to stop TensorBoard and exit.")
        tensorboard_thread.join()
    else:
        print("Training finished.")