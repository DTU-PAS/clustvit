import colorsys
import random

import matplotlib.pyplot as plt
import numpy as np
import umap
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

NUM_CLUSTERS = 2
METRIC = "cosine"


def unnormalize_channel_rev(
    np_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    """Unnormalize a Numpy tensor image."""
    unnormalized_image = (np_image * std) + mean
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    # unnormalized_image = unnormalized_image[:,:,::-1]  # Convert from RGB to BGR
    return np.clip(unnormalized_image, 0, 255).astype(np.uint8)


# Compute Cosine Similarity Between Tokens at Different Layers
def compute_similarity(tokens):
    similarity_matrix = cosine_similarity(tokens)
    return np.mean(similarity_matrix)  # Average similarity


def plot_image(
    image, mask=None, segm=None, save_instead_of_show=None, only_return=False
):
    # Convert the image to a PIL Image
    image = Image.fromarray(image)

    if mask is not None:
        # Convert the mask to a PIL Image and resize it to match the image dimensions
        mask = Image.fromarray(mask).convert("RGB")
        mask = mask.resize(image.size, Image.NEAREST)  # Resize mask to match image size

        # Add an alpha channel to the mask
        mask.putalpha(100)
        maskmask = np.array(mask)
        # maskmask[:,:,3] = maskmask[:,:,2]/(2.55)

        mask = Image.fromarray(maskmask.astype(np.uint8)).convert("RGBA")

        # Convert the image to RGBA mode for blending
        image = image.convert("RGBA")

        # Blend the image and mask
        blended_image = Image.blend(image, mask, alpha=0.8)
        out = np.concatenate([image, blended_image], axis=1)
    else:
        out = image

    if segm is not None:
        # Convert the segmentation mask to a PIL Image and resize it to match the image dimensions
        segm = Image.fromarray(segm).convert("RGBA")
        out = np.concatenate([out, segm], axis=1)

    # Display the blended image
    if only_return:
        return out

    if save_instead_of_show is not None:
        plt.imsave(save_instead_of_show, out)
        plt.close()
    else:
        plt.imshow(out)
        plt.axis("off")
        plt.show()
    return out


def generate_rgb_vector(n, seed=None):
    """
    Generates an array of shape (n, 3) with random RGB values in the range [0, 255].

    :param n: Number of colors to generate
    :param seed: Seed for random number generator (optional)
    :return: NumPy array of shape (n, 3) containing RGB values
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.randint(1, 256, size=(n, 3), dtype=np.uint8)


def generate_distinct_colors(n=20):
    """
    Generate N highly distinguishable RGB colors.

    Parameters:
    n (int): Number of colors to generate.

    Returns:
    list: A list of N RGB colors in 0-255 format.
    """
    colors = []
    for i in range(n):
        hue = i / n  # Evenly spaced hues
        rgb = colorsys.hsv_to_rgb(hue, 1, 1)  # Convert HSV to RGB
        rgb = tuple(int(c * 255) for c in rgb)  # Scale to 0-255
        colors.append(rgb)

    random.Random(24).shuffle(colors)

    return colors
