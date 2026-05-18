import numpy as np
from PIL import Image

DEFAULT_HIST_BINS = 8


def feature_columns(hist_bins=DEFAULT_HIST_BINS):
    """Ordered list of feature column names produced by :func:`extract_color_features`."""
    cols = []
    for name in ("r", "g", "b", "h", "s", "v"):
        cols.append(f"{name}_mean")
        cols.append(f"{name}_std")
    for name in ("r", "g", "b"):
        cols.extend(f"{name}_hist_{i}" for i in range(hist_bins))
    return cols


def extract_color_features(image_path, hist_bins=DEFAULT_HIST_BINS):
    """Return a dict of colour-distribution features for a single image.

    Features (defaults):
      - mean, std per channel in RGB and HSV  (12 features)
      - normalized per-channel RGB histogram  (3 * hist_bins features)
    """
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
        hsv = np.asarray(im.convert("HSV"), dtype=np.float32)

    features = {}
    for i, name in enumerate(("r", "g", "b")):
        ch = rgb[..., i]
        features[f"{name}_mean"] = float(ch.mean())
        features[f"{name}_std"] = float(ch.std())
    for i, name in enumerate(("h", "s", "v")):
        ch = hsv[..., i]
        features[f"{name}_mean"] = float(ch.mean())
        features[f"{name}_std"] = float(ch.std())
    for i, name in enumerate(("r", "g", "b")):
        hist, _ = np.histogram(rgb[..., i], bins=hist_bins, range=(0.0, 256.0))
        total = hist.sum()
        if total > 0:
            hist = hist / total
        for b, v in enumerate(hist):
            features[f"{name}_hist_{b}"] = float(v)
    return features
