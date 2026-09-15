#!/usr/bin/env python3
"""
Visualize YOLO predictions by overlaying masks on original images.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random
from pycocotools import mask as maskUtils
import cv2

def load_predictions(pred_file):
    """Load YOLO predictions from JSON file."""
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    return predictions

def load_image_by_id(image_id, data_root):
    """Load image by finding it in the data directory."""
    # YOLO image IDs are typically filename stems
    image_id_str = str(image_id)
    
    # Search for the image file
    possible_extensions = ['.spm', '.jpg', '.png', '.tiff', '.tif']
    
    for root_dir in ['raw', 'yolo_view']:
        search_root = data_root / root_dir
        if not search_root.exists():
            continue
            
        for ext in possible_extensions:
            # Try direct match
            img_files = list(search_root.rglob(f"{image_id_str}{ext}"))
            if img_files:
                return str(img_files[0])
            
            # Try with parentheses (like "0.0_00007 (9).spm")
            img_files = list(search_root.rglob(f"*{image_id_str}*{ext}"))
            if img_files:
                # Filter to exact matches
                for img_file in img_files:
                    if image_id_str in img_file.stem:
                        return str(img_file)
    
    return None

def decode_rle_mask(rle_dict, height, width):
    """Decode RLE mask to binary array."""
    if isinstance(rle_dict, dict) and 'counts' in rle_dict:
        # COCO RLE format
        rle_dict['size'] = [height, width]
        mask = maskUtils.decode(rle_dict)
        return mask
    else:
        print(f"Unsupported mask format: {type(rle_dict)}")
        return np.zeros((height, width), dtype=np.uint8)

def visualize_predictions(model_name, pred_file, data_root, output_dir, num_samples=5):
    """Visualize random predictions for a model."""
    print(f"\n=== Visualizing {model_name} ===")
    
    # Load predictions
    predictions = load_predictions(pred_file)
    print(f"Loaded {len(predictions)} predictions")
    
    if not predictions:
        print("No predictions found!")
        return
    
    # Group predictions by image_id
    image_predictions = {}
    for pred in predictions:
        img_id = str(pred['image_id'])
        if img_id not in image_predictions:
            image_predictions[img_id] = []
        image_predictions[img_id].append(pred)
    
    print(f"Predictions for {len(image_predictions)} images")
    
    # Sample random images
    sample_image_ids = random.sample(list(image_predictions.keys()), 
                                   min(num_samples, len(image_predictions)))
    
    # Create output directory
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img_id in enumerate(sample_image_ids):
        print(f"Processing image {img_id}...")
        
        # Find and load image
        img_path = load_image_by_id(img_id, data_root)
        if not img_path:
            print(f"Could not find image for ID: {img_id}")
            continue
            
        print(f"Found image: {img_path}")
        
        # Load image
        if img_path.endswith('.spm'):
            # Handle SPM files (assuming they're grayscale)
            try:
                img = np.fromfile(img_path, dtype=np.float32)
                # Assume square image for now - might need to adjust
                size = int(np.sqrt(len(img)))
                if size * size == len(img):
                    img = img.reshape(size, size)
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                    img = np.stack([img, img, img], axis=-1)  # Convert to RGB
                else:
                    print(f"Could not determine SPM image dimensions for {img_path}")
                    continue
            except Exception as e:
                print(f"Error loading SPM file {img_path}: {e}")
                continue
        else:
            try:
                img = np.array(Image.open(img_path))
                if len(img.shape) == 2:  # Grayscale
                    img = np.stack([img, img, img], axis=-1)
                elif img.shape[2] == 4:  # RGBA
                    img = img[:, :, :3]
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
        
        height, width = img.shape[:2]
        
        # Get predictions for this image
        img_preds = image_predictions[img_id]
        print(f"Image {img_id}: {len(img_preds)} predictions")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title(f'Original Image\n{Path(img_path).name}')
        axes[0].axis('off')
        
        # Image with predictions
        overlay = img.copy()
        
        # Color map for different predictions
        colors = plt.cm.Set1(np.linspace(0, 1, len(img_preds)))
        
        for j, pred in enumerate(img_preds):
            try:
                # Decode mask
                if 'segmentation' in pred:
                    mask = decode_rle_mask(pred['segmentation'], height, width)
                    
                    # Create colored overlay
                    color = (colors[j][:3] * 255).astype(np.uint8)
                    mask_colored = np.zeros_like(img)
                    mask_colored[mask > 0] = color
                    
                    # Blend with original image
                    alpha = 0.5
                    overlay = cv2.addWeighted(overlay, 1-alpha, mask_colored, alpha, 0)
                    
                    # Add score text
                    score = pred.get('score', 0.0)
                    bbox = pred.get('bbox', [0, 0, 50, 20])  # x, y, w, h
                    x, y = int(bbox[0]), int(bbox[1])
                    
                    cv2.putText(overlay, f'{score:.2f}', (x, y-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 1)
                              
            except Exception as e:
                print(f"Error processing prediction {j}: {e}")
                continue
        
        axes[1].imshow(overlay)
        axes[1].set_title(f'Predictions (n={len(img_preds)})\nModel: {model_name}')
        axes[1].axis('off')
        
        # Save visualization
        output_file = model_output_dir / f'sample_{i:02d}_{img_id}.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_file}")
    
    print(f"Completed visualization for {model_name}")

def main():
    # Configuration
    data_root = Path('../data')
    results_dir = Path('results/yolo_eval_proper')
    output_dir = Path('results/yolo_visual_samples')
    
    # Models to visualize (best, worst, average)
    models_to_viz = {
        'single_bowtie': 'Best (AP=0.822)',
        'leave_out_triangle': 'Worst (AP=0.210)', 
        'pooled_shape': 'Average (AP=0.439)',
        'single_triangle': 'Moderate (AP=0.388)'
    }
    
    print("🎨 YOLO Prediction Visualization")
    print("=" * 50)
    
    random.seed(42)  # For reproducible sampling
    
    for model_name, description in models_to_viz.items():
        pred_file = results_dir / model_name / 'val' / 'predictions.json'
        
        if not pred_file.exists():
            print(f"❌ Predictions not found for {model_name}: {pred_file}")
            continue
            
        print(f"📊 {model_name} ({description})")
        try:
            visualize_predictions(model_name, pred_file, data_root, output_dir, num_samples=3)
        except Exception as e:
            print(f"❌ Error visualizing {model_name}: {e}")
    
    print("\n✅ Visualization completed!")
    print(f"📁 Check results in: {output_dir}")

if __name__ == '__main__':
    main() 