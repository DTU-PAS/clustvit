import os
import random
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset
from typing import Optional

@DATASETS.register_module()
class SuimDataset(BaseSegDataset):
    METAINFO = {
        'classes': [
            'background', 'water', 'sky', 'sand', 'coral', 'plant', 'fish', 'wreck', 'human'
        ],
        'palette': [
            [0, 0, 0],        # background
            [0, 0, 255],      # water
            [0, 255, 255],    # sky
            [194, 178, 128],  # sand
            [255, 0, 0],      # coral
            [0, 255, 0],      # plant
            [255, 255, 0],    # fish
            [255, 0, 255],    # wreck
            [128, 0, 128],    # human
        ]
    }

    def __init__(self, data_root, img_dir, mask_dir, split_file=None, **kwargs):
        self.img_dir = os.path.join(data_root, img_dir)
        self.mask_dir = os.path.join(data_root, mask_dir)
        self.split_file = os.path.join(data_root, split_file) if split_file else None
        super().__init__(data_root=data_root, **kwargs)

    def load_data_list(self):
        if self.split_file and os.path.exists(self.split_file):
            with open(self.split_file, 'r') as f:
                basenames = [line.strip() for line in f if line.strip()]
        else:
            # Use all images in img_dir
            basenames = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        data_list = []
        for name in basenames:
            img_path = os.path.join(self.img_dir, name + '.jpg')
            mask_path = os.path.join(self.mask_dir, name + '.png')
            data_list.append({
                'img_path': img_path,
                'seg_map_path': mask_path,
                'reduce_zero_label': False,
                'seg_fields': []
            })
        return data_list

def suim_train_val_split(suim_root: str, split_ratio=0.85, seed=42):
    """
    Split images in suim_root/train_val/images into train and val sets and write to txt files in suim_root/train_val/.
    """
    img_dir = os.path.join(suim_root, 'train_val/images')
    train_txt = os.path.join(suim_root, 'train_val/train.txt')
    val_txt = os.path.join(suim_root, 'train_val/val.txt')
    all_imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    all_basenames = [os.path.splitext(f)[0] for f in all_imgs]
    random.seed(seed)
    random.shuffle(all_basenames)
    split_idx = int(len(all_basenames) * split_ratio)
    train_basenames = all_basenames[:split_idx]
    val_basenames = all_basenames[split_idx:]
    with open(train_txt, 'w') as f:
        for name in train_basenames:
            f.write(name + '\n')
    with open(val_txt, 'w') as f:
        for name in val_basenames:
            f.write(name + '\n')

def suim_convert_masks(suim_root: str):
    """
    Convert RGB masks to single-channel label masks for both train_val and test folders in SUIM root.
    """
    import numpy as np
    from PIL import Image
    PALETTE = [
        (0, 0, 0),        # background
        (0, 0, 255),      # water
        (0, 255, 255),    # sky
        (194, 178, 128),  # sand
        (255, 0, 0),      # coral
        (0, 255, 0),      # plant
        (255, 255, 0),    # fish
        (255, 0, 255),    # wreck
        (128, 0, 128),    # human
    ]
    def rgb_to_label(mask):
        mask = np.array(mask)
        label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for idx, color in enumerate(PALETTE):
            matches = np.all(mask == color, axis=-1)
            label_mask[matches] = idx
        return label_mask
    def convert_masks(mask_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for fname in os.listdir(mask_dir):
            if fname.endswith('.bmp'):
                mask = Image.open(os.path.join(mask_dir, fname)).convert('RGB')
                label_mask = rgb_to_label(mask)
                out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + '.png')
                Image.fromarray(label_mask).save(out_path)
    # Convert train_val masks
    convert_masks(os.path.join(suim_root, 'train_val/masks'), os.path.join(suim_root, 'train_val/masks_label'))
    # Convert test masks
    convert_masks(os.path.join(suim_root, 'test/masks'), os.path.join(suim_root, 'test/masks_label'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SUIM utility functions')
    parser.add_argument('suim_root', type=str, help='Path to SUIM root directory')
    parser.add_argument('--split', action='store_true', help='Run train/val split')
    parser.add_argument('--convert', action='store_true', help='Convert masks to label format')
    args = parser.parse_args()
    if args.split:
        suim_train_val_split(args.suim_root)
    if args.convert:
        suim_convert_masks(args.suim_root) 