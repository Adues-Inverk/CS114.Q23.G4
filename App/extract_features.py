import cv2
import numpy as np
import pywt
from skimage import feature


def extract_custom_features(image_rgb):
    """
    Custom feature extraction function for brush stroke analysis.
    Input: image_rgb (RGB matrix)
    Output: feature_vector (vector containing feature values)
    """
    features = []

    # 1. HSV Histogram extraction
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    # Calc histograms for Hue and Saturation channels
    hist_h = cv2.calcHist([image_hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([image_hsv], [1], None, [16], [0, 256]).flatten()

    # Normalize to speed up model convergence
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    features.extend(hist_h)
    features.extend(hist_s)

    # Grayscale
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # 2. LBP (Local Binary Patterns) for texture analysis
    radius = 1
    n_points = 8 * radius
    # Calc LBP
    lbp = feature.local_binary_pattern(image_gray, n_points, radius, method="uniform")
    # Create histogram of LBP
    n_bins = int(lbp.max() + 1)
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-7)  # Normalize
    features.extend(hist_lbp)

    # 3. Edge/Gradient feature extraction using algorithms like Canny or Sobel
    # Edge density can reflect the sharpness of brush strokes
    edges = cv2.Canny(image_gray, 100, 200)
    edge_density = np.sum(edges / 255.0) / (image_gray.shape[0] * image_gray.shape[1])
    features.append(edge_density)

    # 4. GLCM (Gray Level Co-occurrence Matrix) features
    # X  X
    # | /
    # O - X
    #   \
    #    X
    glcm = feature.graycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Extract GLCM properties and flatten
    features.extend(feature.graycoprops(glcm, 'contrast').flatten())
    features.extend(feature.graycoprops(glcm, 'dissimilarity').flatten())
    features.extend(feature.graycoprops(glcm, 'homogeneity').flatten())
    features.extend(feature.graycoprops(glcm, 'energy').flatten())
    features.extend(feature.graycoprops(glcm, 'correlation').flatten())

    # 5. Wavelet transform features
    # (LL, LH, HL, HH) using Haar wavelet
    coeffs2 = pywt.dwt2(image_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # Tính năng lượng ở 3 dải tần số cao (nơi AI thường để lộ sơ hở nhiễu hạt)
    energy_LH = np.sum(LH ** 2) / (LH.shape[0] * LH.shape[1])
    energy_HL = np.sum(HL ** 2) / (HL.shape[0] * HL.shape[1])
    energy_HH = np.sum(HH ** 2) / (HH.shape[0] * HH.shape[1])

    features.extend([energy_LH, energy_HL, energy_HH])

    # Combine all features into a single vector
    return np.array(features, dtype=np.float32)