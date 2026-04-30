import argparse
import os

import torch


def split_model_weights(input_pth):
    # Load the checkpoint
    checkpoint = torch.load(input_pth, map_location="cpu")

    # Try to get the actual state_dict
    if "state_dict" in checkpoint:
        print("🔍 Found 'state_dict' in checkpoint, extracting weights...")
        state_dict = checkpoint["state_dict"]
    else:
        print("⚠️ No 'state_dict' key found — using the top-level dict as weights.")
        state_dict = checkpoint

    # Initialize split dicts
    backbone_dict = {}
    decoder_dict = {}

    # Split weights
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            backbone_dict[k] = v
        elif (
            k.startswith("decode_head.")
            or k.startswith("head.")
            or k.startswith("decoder.")
        ):
            decoder_dict[k] = v
        else:
            print(f"Skipping unknown key: {k}")

    # Save paths (same dir as input)
    output_dir = os.path.dirname(input_pth)
    backbone_path = os.path.join(output_dir, "backbone.pth")
    decoder_path = os.path.join(output_dir, "decoder_head.pth")

    # Save weights
    torch.save(backbone_dict, backbone_path)
    torch.save(decoder_dict, decoder_path)

    print(f"✅ Backbone weights saved to: {backbone_path}")
    print(f"✅ Decoder head weights saved to: {decoder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split model weights into backbone and decoder head."
    )
    parser.add_argument(
        "input_pth",
        type=str,
        help="Path to the model .pth file (checkpoint or state_dict).",
    )
    args = parser.parse_args()

    split_model_weights(args.input_pth)
