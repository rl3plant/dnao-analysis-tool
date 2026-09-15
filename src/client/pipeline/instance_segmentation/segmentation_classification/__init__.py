"""
Segmentation Classification Module

This module provides classifiers for filtering segmentation masks based on user annotations.
"""

from .base_classifier import BaseClassifier
from .svm_classifier import SVMClassifier

__all__ = ['BaseClassifier', 'SVMClassifier'] 