import os
import numpy as np
from PIL import Image

# SUIM color palette (RGB) and class indices
PALETTE = [
    (0, 0, 0),        # background (waterbody)
    (0, 0, 255),      # human divers
    (0, 255, 0),      # aquatic plants and sea-grass
    (135, 206, 235),  # wrecks and ruins (sky blue)
    (255, 0, 0),      # robots (AUVs/ROVs/instruments)
    (255, 192, 203),  # reefs and invertebrates (pink)
    (255, 255, 0),    # fish and vertebrates (yellow)
    (255, 255, 255),  # sea-floor and rocks (white)
]

# Handle PIL version compatibility for NEAREST
RESAMPLE_NEAREST = None
if hasattr(Image, 'Resampling'):
    RESAMPLE_NEAREST = getattr(Image.Resampling, 'NEAREST', None)
if RESAMPLE_NEAREST is None:
    RESAMPLE_NEAREST = getattr(Image, 'NEAREST', None)
if RESAMPLE_NEAREST is None:
    raise ImportError('Cannot find NEAREST resampling in PIL.Image')

def rgb_to_label(mask):
    mask = np.array(mask)
    label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        matches = np.all(mask == color, axis=-1)
        label_mask[matches] = idx
    return label_mask

def convert_masks(image_dir, mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(image_dir):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + '.bmp')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + '.png')
            if not os.path.exists(mask_path):
                print(f"Warning: No mask found for {fname}")
                continue
        img = Image.open(img_path)
        mask = Image.open(mask_path).convert('RGB')
        if img.size != mask.size:
            print(f"Warning: Image and mask size mismatch for {fname}: image {img.size}, mask {mask.size}. Resizing mask to match image.")
            mask = mask.resize(img.size, RESAMPLE_NEAREST)
        label_mask = rgb_to_label(mask)
        out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + '.png')
        Image.fromarray(label_mask).save(out_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert SUIM RGB masks to single-channel label masks (no resizing unless mismatch).')
    parser.add_argument('suim_root', type=str, help='Path to SUIM root directory')
    args = parser.parse_args()
    # Convert train_val masks
    convert_masks(
        os.path.join(args.suim_root, 'train_val/images'),
        os.path.join(args.suim_root, 'train_val/masks'),
        os.path.join(args.suim_root, 'train_val/masks_label')
    )
    # Convert test masks
    convert_masks(
        os.path.join(args.suim_root, 'test/images'),
        os.path.join(args.suim_root, 'test/masks'),
        os.path.join(args.suim_root, 'test/masks_label')
    )