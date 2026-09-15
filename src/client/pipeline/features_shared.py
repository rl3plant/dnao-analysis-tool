import cv2
import numpy as np
from pyefd import elliptic_fourier_descriptors
import logging

logger = logging.getLogger(__name__)

def extract_sift_features(image, mask=None):
    """
    Extract SIFT features from an image patch.
    Args:
        image: numpy array of the image
        mask: optional mask for the region of interest
    Returns:
        SIFT descriptors for the image
    """
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, mask)
        if descriptors is None:
            return np.array([])
        return descriptors
    except Exception as e:
        logger.error(f"Error extracting SIFT features: {str(e)}")
        return np.array([])

def extract_fourier_descriptors(contour, n_descriptors=20):
    """
    Extract Fourier descriptors from a contour.
    Args:
        contour: numpy array of contour points
        n_descriptors: number of descriptors to return
    Returns:
        Fourier descriptors for the contour
    """
    try:
        if len(contour.shape) == 3:
            contour = contour.squeeze()
        descriptors = elliptic_fourier_descriptors(contour, n_descriptors)
        return descriptors
    except Exception as e:
        logger.error(f"Error extracting Fourier descriptors: {str(e)}")
        return np.zeros((n_descriptors, 4))

def extract_features(image, contour):
    """
    Extract both SIFT and Fourier descriptors for a region.
    Args:
        image: numpy array of the image
        contour: numpy array of contour points
    Returns:
        Combined feature vector
    """
    try:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        sift_features = extract_sift_features(image, mask)
        fourier_features = extract_fourier_descriptors(contour)
        if len(sift_features) == 0:
            sift_mean = np.zeros(128)
            logger.warning("No SIFT features found, using default values")
        else:
            sift_mean = np.mean(sift_features, axis=0)
        sift_mean = sift_mean.flatten()
        fourier_features = fourier_features.flatten()
        if len(sift_mean) < 128:
            sift_mean = np.pad(sift_mean, (0, 128 - len(sift_mean)))
        elif len(sift_mean) > 128:
            sift_mean = sift_mean[:128]
        if len(fourier_features) < 20:
            fourier_features = np.pad(fourier_features, (0, 20 - len(fourier_features)))
        elif len(fourier_features) > 20:
            fourier_features = fourier_features[:20]
        combined_features = np.concatenate([sift_mean, fourier_features])
        return combined_features
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return np.zeros(148)

def extract_mask_features(mask: dict, image: np.ndarray = None) -> np.ndarray:
    """
    Extract features from a segmentation mask.
    Args:
        mask: Dictionary containing mask data with 'contours' key
        image: Optional original image for appearance features
    Returns:
        Feature vector as numpy array
    """
    try:
        contour = None
        area = 0.0
        perimeter = 0.0
        bbox_area = 0.0
        w, h = 0, 0
        if 'contours' in mask and mask['contours']:
            contour = np.array(mask['contours'][0])
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            bbox_area = w * h
        features = []
        # features.extend(_extract_contour_features(area, perimeter, contour))
        # features.extend(_extract_area_features(mask, area, bbox_area))
        # features.extend(_extract_shape_features(area, perimeter, w, h, contour))
        features.extend(_extract_fourier_features(contour))
        feature_vector = np.array(features, dtype=np.float32)
        return np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=0.0)
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return np.zeros(18, dtype=np.float32)

def _extract_contour_features(area: float, perimeter: float, contour: np.ndarray) -> list:
    features = []
    if contour is None:
        return [0.0] * 3
    features.append(area)
    features.append(perimeter)
    num_vertices = len(contour)
    features.append(float(num_vertices))
    return features

def _extract_area_features(mask: dict, area: float, bbox_area: float) -> list:
    features = []
    features.append(area)
    features.append(bbox_area)
    if bbox_area > 0:
        area_ratio = area / bbox_area
    else:
        area_ratio = 0.0
    features.append(area_ratio)
    return features

def _extract_shape_features(area: float, perimeter: float, w: int, h: int, contour: np.ndarray) -> list:
    features = []
    if contour is None:
        return [0.0] * 4
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    else:
        circularity = 0.0
    features.append(circularity)
    if h > 0:
        aspect_ratio = w / h
    else:
        aspect_ratio = 1.0
    features.append(aspect_ratio)
    if w > 0 and h > 0:
        extent = area / (w * h)
    else:
        extent = 0.0
    features.append(extent)
    try:
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 0.0
    except:
        solidity = 0.0
    features.append(solidity)
    return features

def _extract_fourier_features(contour: np.ndarray) -> list:
    features = []
    if contour is None or len(contour) < 3:
        return [0.0] * 16
    try:
        if len(contour.shape) == 3:
            contour = contour.squeeze()
        descriptors = elliptic_fourier_descriptors(contour, order=15, normalize=True)
        fourier_features = descriptors.flatten()
        features.extend(fourier_features)
    except Exception as e:
        logger.error(f"Error extracting Fourier features: {e}")
        features.extend([0.0] * 16)
    return features

def extract_features_batch(masks: list, image: np.ndarray = None) -> np.ndarray:
    if not masks:
        return np.array([])
    feature_vectors = []
    for mask in masks:
        features = extract_mask_features(mask, image)
        feature_vectors.append(features)
    return np.array(feature_vectors) 