import numpy as np
import logging
from typing import List, Dict, Any, Callable

logger = logging.getLogger(__name__)

class MaskFilter:
    """Class for filtering masks based on various criteria"""
    
    @staticmethod
    def filter_masks_by_area(masks: List[Dict[str, Any]], threshold_factor: float = 0.5) -> List[Dict[str, Any]]:
        """Filter masks based on area, removing outliers
        
        Args:
            masks: List of mask dictionaries
            threshold_factor: Factor to determine the acceptable range around the median area
            
        Returns:
            Filtered list of masks
        """
        if not masks:
            return []
        
        # Extract areas from all masks
        areas = [mask.get('area', 0) for mask in masks]
        median_area = np.median(areas)
        
        # Calculate allowed range around median
        min_area = median_area * (1 - threshold_factor)
        max_area = median_area * (1 + threshold_factor)
        
        # Filter masks within the acceptable range
        filtered_masks = [
            mask for mask in masks 
            if min_area <= mask.get('area', 0) <= max_area
        ]
        
        logger.info(f"Filtered {len(masks)} masks to {len(filtered_masks)} based on area")
        return filtered_masks
    
    @staticmethod
    def apply_filters(masks: List[Dict[str, Any]], filters: List[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """Apply multiple filters to a list of masks
        
        Args:
            masks: List of mask dictionaries
            filters: List of filter functions to apply
            
        Returns:
            Filtered list of masks
        """
        filtered_masks = masks
        
        for filter_func in filters:
            filtered_masks = filter_func(filtered_masks)
            
        return filtered_masks
    
    @staticmethod
    def create_area_filter(threshold_factor: float = 0.5) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Create an area filter function with the specified threshold
        
        Args:
            threshold_factor: Factor to determine the acceptable range around the median area
            
        Returns:
            A filter function that can be used with apply_filters
        """
        def area_filter(masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return MaskFilter.filter_masks_by_area(masks, threshold_factor)
            
        return area_filter 