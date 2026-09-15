import numpy as np
from sklearn.cluster import KMeans
from typing import List
from src.client.pipeline.features_shared import _extract_fourier_features


def fit_kmeans_chirality(annotations: List, n_clusters: int = 2, random_state: int = 42):
    """
    Fit a KMeans model to the Fourier features of the annotation contours.
    Args:
        annotations: List of Annotation objects (must have .contour attribute)
        n_clusters: Number of clusters (should be 2 for chirality)
        random_state: Random state for reproducibility
    Returns:
        kmeans: Trained KMeans model
        features: Feature array used for clustering
    """
    features = np.array([
        _extract_fourier_features(ann.contour) for ann in annotations
    ])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(features)
    return kmeans, features


def assign_chirality_labels_kmeans(kmeans: KMeans, annotations: List, features: np.ndarray = None, update_label: bool = True):
    """
    Assign chirality labels to annotations using a trained KMeans model.
    Args:
        kmeans: Trained KMeans model
        annotations: List of Annotation objects
        features: Optional, precomputed features (if None, will be computed)
        update_label: If True, update annotation.class_label and annotation.confidence
    Returns:
        List of (label, cluster_idx) for each annotation
    """
    if features is None:
        features = np.array([
            _extract_fourier_features(ann.contour) for ann in annotations
        ])
    cluster_labels = kmeans.predict(features)

    # Always use proper chirality labels when we have a trained KMeans model
    # The model was trained with 2 clusters, so we can safely map them to Z-shape/S-shape
    label_map = {0: 'Z-shape', 1: 'S-shape'}

    results = []
    for ann, cluster_idx in zip(annotations, cluster_labels):
        label = label_map.get(cluster_idx, str(cluster_idx))
        results.append((label, cluster_idx))
        if update_label:
            ann.class_label = label
            ann.confidence = 1.0  # KMeans does not provide probability
    return results 