import csv
import cv2
import numpy as np
from pathlib import Path
from skimage.feature import hog


TARGET_SIZE = (256, 256)
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True,
}


def hog_feature_columns(target_size=TARGET_SIZE, hog_kwargs=HOG_PARAMS):
    dummy_img = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    feats = hog(dummy_img, feature_vector=True, **hog_kwargs)
    return [f"hog_{i}" for i in range(len(feats))]


def extract_hog_features(img_path, target_size=TARGET_SIZE, hog_kwargs=HOG_PARAMS):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image for HOG or image is corrupted.")
    img_resized = cv2.resize(img, target_size)
    features = hog(img_resized, feature_vector=True, **hog_kwargs)
    return {f"hog_{i}": val for i, val in enumerate(features)}

