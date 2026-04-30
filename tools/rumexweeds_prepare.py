import glob
import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

# Define the output structure
SPLITS = ["train", "val", "test"]
SPLIT_SUFFIXES = {
    "train": "random_train.txt",
    "val": "random_val.txt",
    "test": "random_test.txt",
}

# RumexWeeds mask format: 0=background, 1=rumex_obtusifolius, 2=rumex_crispus
CLASSES_TO_CONSIDER = ["rumex_obtusifolius", "rumex_crispus"]
CLASS_MAP = {"background": 0, "rumex_obtusifolius": 1, "rumex_crispus": 2}

# Helper to find the annotation XML for a given image path
# Given e.g. 20210806_hegnstrup/seq0/imgs/20210806_hegnstrup_rgb_0_1628251180326227114.png
# annotation is at 20210806_hegnstrup/seq0/annotations_seg.xml


def get_xml_for_image(rumex_root, rel_img_path):
    seq_dir = os.path.dirname(
        os.path.dirname(rel_img_path)
    )  # e.g. 20210806_hegnstrup/seq0
    xml_path = os.path.join(rumex_root, seq_dir, "annotations_seg.xml")
    return xml_path


# Parse annotation_seg.xml and create a mask for the given image
# (Assumes the same logic as visualize_img_data.py)
def create_mask_from_xml(xml_path, img_name, img_size):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mask = np.zeros(img_size[::-1], dtype=np.uint8)  # (H, W)
    # Find the <image> element with the correct name
    for image_elem in root.findall("image"):
        if image_elem.attrib.get("name") != img_name:
            continue
        for poly in image_elem.findall("polygon"):
            label = poly.attrib.get("label", "").strip().lower()
            if label in CLASSES_TO_CONSIDER:
                points_str = poly.attrib["points"]
                pts = []
                for pt_str in points_str.split(";"):
                    x_str, y_str = pt_str.strip().split(",")
                    x = int(float(x_str))
                    y = int(float(y_str))
                    pts.append((x, y))
                if pts:
                    from PIL import ImageDraw

                    mask_img = Image.fromarray(mask)
                    draw = ImageDraw.Draw(mask_img)
                    draw.polygon(pts, outline=CLASS_MAP[label], fill=CLASS_MAP[label])
                    mask = np.array(mask_img)
    return mask


def process_split(rumex_root, split_name, split_txt, out_root):
    with open(split_txt, "r") as f:
        rel_img_paths = [line.strip() for line in f if line.strip()]
    img_out_dir = os.path.join(out_root, split_name, "images")
    mask_out_dir = os.path.join(out_root, split_name, "masks")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    for rel_img_path in rel_img_paths:
        img_file = os.path.join(rumex_root, rel_img_path)
        xml_file = get_xml_for_image(rumex_root, rel_img_path)
        if not os.path.exists(img_file) or not os.path.exists(xml_file):
            print(f"Warning: Could not find image or annotation for {rel_img_path}")
            continue
        # Create mask
        img = Image.open(img_file)
        mask = create_mask_from_xml(xml_file, os.path.basename(img_file), img.size)
        # Always save image and mask
        shutil.copy(img_file, os.path.join(img_out_dir, os.path.basename(rel_img_path)))
        mask_save_path = os.path.join(
            mask_out_dir, os.path.splitext(os.path.basename(rel_img_path))[0] + ".png"
        )
        Image.fromarray(mask).save(mask_save_path)
        print(f"Processed {rel_img_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare RumexWeeds dataset for segmentation (SUIM format)."
    )
    parser.add_argument(
        "rumex_root", type=str, help="Path to RumexWeeds root directory"
    )
    args = parser.parse_args()
    split_dir = os.path.join(args.rumex_root, "dataset_splits")
    out_root = os.path.join(args.rumex_root, "processed")
    for split in SPLITS:
        for split_file in os.listdir(split_dir):
            if split_file.endswith(SPLIT_SUFFIXES[split]):
                process_split(
                    args.rumex_root,
                    split,
                    os.path.join(split_dir, split_file),
                    out_root,
                )


if __name__ == "__main__":
    main()
