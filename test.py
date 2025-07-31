import os
import time
import torch
import argparse
from datetime import datetime

import lpips
import pyiqa
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from model import SlimEnhancer
from dataset import get_dataloader
from utils import AverageMeter, write_img, chw_to_hwc, deltaEab, deltaE00


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='multi_fin2_style_1')
parser.add_argument('--cnum', default=5, type=int, help='curve number')
parser.add_argument('--cnode', default=64, type=int, help='curve node')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--model_dir', default='./model/{exp_name}/', type=str)
parser.add_argument('--data_dir', default='./data/', type=str, help='path to data')
parser.add_argument('--image_dir', default='./result/{exp_name}/', type=str, help='path to save results')
parser.add_argument('--is_save_result', default=False, help='save the enhanced results')
parser.add_argument('--is_test_deltaE', default=False, help='test deltaE')
# clip
parser.add_argument('--clip_model', default='ViT-L-14-336px', type=str, help='CLIP model to use')
parser.add_argument('--clip_model_path', default='/share/datasets/clip_model/', type=str)
parser.add_argument('--use_text_features', action='store_true', default=True, help='Use text features from CLIP')
parser.add_argument('--use_image_features', action='store_true', default=True, help='Use image features from CLIP')
parser.add_argument('--fusion_method', type=str, default='multiply', choices=['concat', 'add', 'multiply', 'attention'], help='Method to fuse image and text features')
# style
parser.add_argument('--csv_path', type=str, default='./style data', help='path to csv file')
parser.add_argument('--input_styles', type=list, default=['06-Input-ExpertC1.5'], help='input styles')
parser.add_argument('--gt_styles', type=list, default=['05-Experts-E'], help='gt styles')
args = parser.parse_args()

args.model_dir = args.model_dir.format(exp_name=args.exp_name)
print(f"model read: {args.model_dir}")
args.image_dir = args.image_dir.format(exp_name=args.exp_name)
os.makedirs(args.image_dir, exist_ok=True)
print(f"result save: {args.image_dir}")


def test(test_loader, network, metric):
	time_start = time.time()

	deltaEab_metric = deltaEab()
	deltaE00_metric = deltaE00()
	loss_fn_alex = lpips.LPIPS(net='alex').cuda()
	loss_fn_vgg = lpips.LPIPS(net='vgg', version='0.1').cuda()
	iqa_metric = pyiqa.create_metric('ssim').cuda()

	PSNR = AverageMeter()
	SSIM = AverageMeter()
	LPIPS = AverageMeter()
	SSIM_PYIQA = AverageMeter()
	DeltaEab = AverageMeter()
	DeltaE00 = AverageMeter()
	InferenceTime = AverageMeter()

	torch.cuda.empty_cache()
	network.eval()

	for ind, (input_img, target_img, text, img_name) in enumerate(test_loader):
		input_img = input_img.cuda(non_blocking=True)
		target_img = target_img.cuda(non_blocking=True)

		inference_start = time.time()
		with torch.no_grad():
			output = network(input_img, text)
			output = output.clamp_(0, 1)
		inference_end = time.time()
		inference_time = inference_end - inference_start
		InferenceTime.update(inference_time)

		output_cpu = output.cpu().numpy().squeeze(0).transpose(1, 2, 0)
		target_img_cpu = target_img.cpu().numpy().squeeze(0).transpose(1, 2, 0)
		psnr_val = psnr(target_img_cpu, output_cpu, data_range=1.0)
		ssim_val = ssim(target_img_cpu, output_cpu, win_size = 11, channel_axis=-1, data_range=1.0)
		ssim_pyiqa_val = iqa_metric(output, target_img).item()
		# lpips_val = loss_fn_vgg(output, target_img).item()
		lpips_val = loss_fn_alex(output * 2 - 1, target_img * 2 - 1).item()

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)
		LPIPS.update(lpips_val)
		SSIM_PYIQA.update(ssim_pyiqa_val)

		print('Testing: [{0}]\t'
			  'PSNR: {psnr.val:.2f} ({psnr.avg:.2f})\t'
			  'SSIM: {ssim.val:.4f} ({ssim.avg:.4f})\t'
			  'LPIPS: {lpips.val:.04f} ({lpips.avg:.04f})\t'
			  'SSIM_PYIQA: {ssim_pyiqa.val:.04f} ({ssim_pyiqa.avg:.04f})\t'
			  .format(img_name[0], psnr=PSNR, ssim=SSIM, lpips=LPIPS, ssim_pyiqa=SSIM_PYIQA))

		if args.is_test_deltaE:
			deltaEab_val = deltaEab_metric(output, target_img)
			deltaE00_val = deltaE00_metric(output, target_img)
			DeltaEab.update(deltaEab_val)
			DeltaE00.update(deltaE00_val)
			print('Testing: [{0}]\t'
				'DeltaEab: {deltaEab.val:.04f} ({deltaEab.avg:.04f})\t'
				'DeltaE00: {deltaE00.val:.04f} ({deltaE00.avg:.04f})\t'
				.format(img_name[0], deltaEab=DeltaEab, deltaE00=DeltaE00))

		if args.is_save_result:
			os.makedirs(args.image_dir, exist_ok=True)
			out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
			out_filename = '%04d-%.02f-%.4f.png' % (ind+4501, psnr_val, ssim_val)
			write_img(os.path.join(args.image_dir, out_filename), out_img)

	time_end = time.time()
	total_time = time_end - time_start
	log_file_path = os.path.join(args.image_dir, 'testing_log.txt')
	current_time = datetime.now().strftime('%m-%d %H:%M')

	log_message_base = (
		f"{args.exp_name}, {current_time}, total_time: {total_time:.3}, "
		f"from {args.input_styles} to {args.gt_styles}, "
		f"Avg PSNR: {PSNR.avg:.2f}, Avg SSIM: {SSIM.avg:.4f}, Avg LPIPS: {LPIPS.avg:.4f}, Avg SSIM_PYIQA: {SSIM_PYIQA.avg:.4f}, "
		f"Avg Inference Time: {InferenceTime.avg:.4f}s"
	)
	print(log_message_base.strip())

	with open(log_file_path, "a") as log_file:
		if args.is_test_deltaE:
			log_message_extended = (
				f", Avg DeltaEab: {DeltaEab.avg:.4f}, Avg DeltaE00: {DeltaE00.avg:.4f}, Metric: {metric}\n"
			)
			log_file.write(log_message_base + log_message_extended)
		else:
			log_file.write(log_message_base + '\n')


if __name__ == '__main__':

	test_loader = get_dataloader('test', args)

	network = SlimEnhancer(args)
	network.cuda()

	network_save_path = os.path.join(args.model_dir, f"enhancer_psnr_avg.pth")
	network.load_state_dict(torch.load(os.path.join(network_save_path))['state_dict'])
	test(test_loader, network, metric='PSNR')

	network_save_path = os.path.join(args.model_dir, f"enhancer_ssim_avg.pth")
	network.load_state_dict(torch.load(os.path.join(network_save_path))['state_dict'])
	test(test_loader, network, metric='SSIM')