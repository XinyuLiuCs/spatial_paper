import os
import cv2
import json
import torch
import random
import numpy as np
import pandas as pd
from utils import hwc_to_chw, read_img
from torch.utils.data import Dataset, DataLoader

# ====== Global Resolution Settings ======
DEFAULT_TARGET_SIZES = [(448, 448), (448, 448)]  # Training set resolution (H, W)
TEST_TARGET_SIZES = [(344, 512), (512, 344)]     # Test set resolution (H, W)
TARGET_SIZE_MULTIPLE = 8  # Target multiple

def create_style_description(target_id, style_vector):

    # Ensure target_id is within valid range
    target_id = max(0, min(4, target_id))
    
    style_names = {
        0: "enhance photo like expert A",
        1: "enhance photo like expert B",
        2: "enhance photo like expert C",
        3: "enhance photo like expert D",
        4: "enhance photo like expert E"
    }
    
    def get_brightness_level(val):
        """Brightness average: Set the brightness to {level}"""
        if val >= 1.5: return "very high"
        elif 0.5 <= val < 1.5: return "high"
        elif -0.5 <= val < 0.5: return "medium"
        elif -1.5 <= val < -0.5: return "low"
        else: return "very low"
    
    def get_saturation_level(val):
        """Saturation average: Make the colors {level}"""
        if val >= 1.5: return "intensely vibrant"
        elif 0.5 <= val < 1.5: return "vibrant"
        elif -0.5 <= val < 0.5: return "natural"
        elif -1.5 <= val < -0.5: return "muted"
        else: return "desaturated"
    
    def get_sat_variance_level(val):
        """Saturation standard deviation: Adjust color variation to be {level}"""
        if val >= 1.5: return "extreme"
        elif 0.5 <= val < 1.5: return "high"
        elif -0.5 <= val < 0.5: return "moderate"
        elif -1.5 <= val < -0.5: return "low"
        else: return "minimal"
    
    def get_brightness_var_level(val):
        """Brightness standard deviation: Set the lighting to be {level}"""
        if val >= 1.5: return "dramatic"
        elif 0.5 <= val < 1.5: return "dynamic"
        elif -0.5 <= val < 0.5: return "balanced"
        elif -1.5 <= val < -0.5: return "soft"
        else: return "flat"
    
    def get_color_range_level(val):
        """Color richness: Use a {level} color palette"""
        if val >= 1.5: return "full-spectrum"
        elif 0.5 <= val < 1.5: return "rich"
        elif -0.5 <= val < 0.5: return "standard"
        elif -1.5 <= val < -0.5: return "limited"
        else: return "monochromatic"
    
    def get_contrast_level(val):
        """Contrast: Make the contrast {level}"""
        if val >= 1.5: return "very high"
        elif 0.5 <= val < 1.5: return "high"
        elif -0.5 <= val < 0.5: return "medium"
        elif -1.5 <= val < -0.5: return "low"
        else: return "very low"

    # Build new description text format
    text = f"{style_names[target_id]}, set the brightness to {get_brightness_level(style_vector[0])}, make the colors {get_saturation_level(style_vector[1])}, adjust color variation to be {get_sat_variance_level(style_vector[2])}, set the lighting to be {get_brightness_var_level(style_vector[3])}, use a {get_color_range_level(style_vector[4])} color palette, make the contrast {get_contrast_level(style_vector[5])}."
    
    return text


def read_style_vector(style_id, img_idx, csv_path):
    csv_path1 = csv_path
    csv_path2 = csv_path
    
    # Read two CSV files, skip the first row
    csv_file1 = os.path.join(csv_path1, f"{style_id+1}.csv")
    csv_file2 = os.path.join(csv_path2, f"6.csv")

    style_data1 = pd.read_csv(csv_file1, header=None, skiprows=1)
    style_data2 = pd.read_csv(csv_file2, header=None, skiprows=1)
    
    style_vector1 = style_data1.iloc[:, img_idx].values
    style_vector2 = style_data2.iloc[:, img_idx].values
    style_vector = style_vector1 - style_vector2
    
    return style_vector


def resize_to_target(img, target_size):
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    # Ensure target size is a multiple of TARGET_SIZE_MULTIPLE
    target_h = (target_h // TARGET_SIZE_MULTIPLE) * TARGET_SIZE_MULTIPLE
    target_w = (target_w // TARGET_SIZE_MULTIPLE) * TARGET_SIZE_MULTIPLE
    
    if (h, w) == (target_h, target_w):
        return img
    
    # cv2.resize expects format (width, height)
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def augment_batch(imgs, down_ratio=2):
    B, C, H, W = imgs[0].shape
    
    # Ensure size is a multiple of TARGET_SIZE_MULTIPLE
    H = (H // TARGET_SIZE_MULTIPLE) * TARGET_SIZE_MULTIPLE
    W = (W // TARGET_SIZE_MULTIPLE) * TARGET_SIZE_MULTIPLE
    
    if random.random() < 0.1:
        Hc, Wc = H, W
    elif random.randint(0, 1) == 1:
        randr = random.uniform(0.75, 1.0)
        Hc = int(H * randr) // TARGET_SIZE_MULTIPLE * TARGET_SIZE_MULTIPLE
        Wc = int(W * randr) // TARGET_SIZE_MULTIPLE * TARGET_SIZE_MULTIPLE
    else:
        Hc = int(H * random.uniform(0.75, 1.0)) // TARGET_SIZE_MULTIPLE * TARGET_SIZE_MULTIPLE
        Wc = int(W * random.uniform(0.75, 1.0)) // TARGET_SIZE_MULTIPLE * TARGET_SIZE_MULTIPLE

    Hs = random.randint(0, H-Hc)
    Ws = random.randint(0, W-Wc)
    
    augmented_imgs = []
    for img in imgs:
        img = img[:, :, Hs:(Hs+Hc), Ws:(Ws+Wc)]
        augmented_imgs.append(img)

    if random.randint(0, 1) == 1:
        augmented_imgs = [torch.flip(img, [3]) for img in augmented_imgs]    # Horizontal flip
    if random.randint(0, 1) == 1: 
        augmented_imgs = [torch.flip(img, [2]) for img in augmented_imgs]    # Vertical flip
    if random.randint(0, 1) == 1:
        augmented_imgs = [img.permute(0, 1, 3, 2) for img in augmented_imgs] # Transpose (width-height swap)
    
    return augmented_imgs


class Mydata(Dataset):
    def __init__(self, dataset_type, args):
        self.dataset_type = dataset_type
        self.is_train = True if dataset_type == 'train' else False
        # Support multiple style inputs
        self.csv_path = args.csv_path
        self.input_styles = getattr(args, 'input_styles', ['06-Input-ExpertC1.5'])
        self.gt_styles = getattr(args, 'gt_styles', ['03-Experts-C'])
        self.data_dir = args.data_dir

        # Read all image names under input/gt styles, establish style-image name mapping
        self.input_imgs = self._collect_imgs(self.input_styles, is_input=True)
        self.gt_imgs = self._collect_imgs(self.gt_styles, is_input=False)
        self.img_names = list(self.input_imgs.keys())

        self.target_sizes = DEFAULT_TARGET_SIZES
        if dataset_type == 'train':
            self.size_groups = self._group_images()

    def _collect_imgs(self, style_list, is_input=True):
        # Return dict: {img_name: [style1_path, style2_path, ...]}
        img_dict = {}
        for style in style_list:
            style_dir = os.path.join(self.data_dir, self.dataset_type, style)
            if not os.path.isdir(style_dir):
                print(f"Warning: {style_dir} does not exist")
                continue
            for img_name in os.listdir(style_dir):
                if img_name not in img_dict:
                    img_dict[img_name] = []
                img_dict[img_name].append(os.path.join(style_dir, img_name))
        return img_dict

    def _group_images(self):
        # Group by input style and size
        size_groups = {}
        for style in self.input_styles:
            for img_name in self.input_imgs:
                # Only group images that exist under this style
                style_img_paths = [p for p in self.input_imgs[img_name] if style in p]
                if not style_img_paths:
                    continue
                img_path = style_img_paths[0]
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                aspect_ratio = h / w
                # Use global variables
                target_size = DEFAULT_TARGET_SIZES[1] if aspect_ratio > 1 else DEFAULT_TARGET_SIZES[0]
                key = (style, target_size)
                if key not in size_groups:
                    size_groups[key] = []
                size_groups[key].append(img_name)
        # Print group count, size, style
        for key, value in size_groups.items():
            print(f"Size group: {key}, Number of images: {len(value)}")
        return size_groups

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_name = self.img_names[idx]

        # Randomly select style
        input_style_path = random.choice(self.input_imgs[img_name])
        gt_style_path = random.choice(self.gt_imgs[img_name])

        # Parse style name
        input_style = os.path.basename(os.path.dirname(input_style_path))
        gt_style = os.path.basename(os.path.dirname(gt_style_path))

        target_id = int(gt_style.split('-')[0]) - 1
        source_id = int(input_style.split('-')[0]) - 1

        img_idx = int(img_name.split('.')[0][1:]) - 1
        style_vector = read_style_vector(target_id, img_idx, self.csv_path)
        text = create_style_description(target_id, style_vector)

        input_img = read_img(input_style_path)
        target_img = read_img(gt_style_path)

        h, w = input_img.shape[:2]
        aspect_ratio = h / w
        if self.is_train:
            target_size = DEFAULT_TARGET_SIZES[1] if aspect_ratio > 1 else DEFAULT_TARGET_SIZES[0]
        else:
            target_size = TEST_TARGET_SIZES[1] if aspect_ratio > 1 else TEST_TARGET_SIZES[0]
        input_img = resize_to_target(input_img, target_size)
        target_img = resize_to_target(target_img, target_size)

        input_img = torch.from_numpy(hwc_to_chw(input_img))
        target_img = torch.from_numpy(hwc_to_chw(target_img))

        return input_img, target_img, text, img_name

def collate_fn(batch):
    inputs, targets, texts, img_names = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    inputs, targets = augment_batch([inputs, targets])

    return inputs, targets, texts, img_names

def get_dataloader(dataset_type, args):
    dataset = Mydata(dataset_type, args)
    
    if dataset_type == 'train':
        dataloaders = []
        for target_size, img_names in dataset.size_groups.items():
            group_dataset = torch.utils.data.Subset(dataset, [dataset.img_names.index(name) for name in img_names])
            
            group_loader = DataLoader(
                group_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_fn
            )
            dataloaders.append(group_loader)
        return dataloaders
    else:
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )