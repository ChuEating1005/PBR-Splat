#!/usr/bin/env python3
"""
Create comparison figures for G-buffers: GT (top row) vs Rendered (bottom row)
Usage: python scripts/create_comparison_figure.py <vis_dir> [output_path]
Example: python scripts/create_comparison_figure.py output/car/visuals/iter_1
"""

import sys
import os
from PIL import Image
import numpy as np

def create_comparison_figure(vis_dir, output_path=None):
    """
    Create a comparison figure with GT on top row and Rendered on bottom row.
    
    Args:
        vis_dir: Directory containing gt_*.png and rendered_*.png files
        output_path: Output path for the comparison figure (default: vis_dir/comparison.png)
    """
    gbuffer_types = ['albedo', 'metallic', 'normal', 'roughness']
    
    # Load images
    gt_images = []
    rendered_images = []
    
    for gbuffer_type in gbuffer_types:
        # Try masked GT first, then fallback to unmasked
        gt_path = os.path.join(vis_dir, f'gt_{gbuffer_type}_masked.png')
        if not os.path.exists(gt_path):
            gt_path = os.path.join(vis_dir, f'gt_{gbuffer_type}.png')
        
        rendered_path = os.path.join(vis_dir, f'rendered_{gbuffer_type}.png')
        
        if not os.path.exists(gt_path):
            print(f"Warning: {gt_path} not found, skipping...")
            continue
        if not os.path.exists(rendered_path):
            print(f"Warning: {rendered_path} not found, skipping...")
            continue
            
        gt_img = Image.open(gt_path).convert('RGB')
        rendered_img = Image.open(rendered_path).convert('RGB')
        
        gt_images.append(gt_img)
        rendered_images.append(rendered_img)
    
    if not gt_images or not rendered_images:
        print("Error: No valid image pairs found!")
        return
    
    # Get image dimensions
    img_width, img_height = gt_images[0].size
    
    # Create comparison figure
    # Width: sum of all image widths (no spacing between columns)
    # Height: 2 * image height + vertical spacing
    total_width = img_width * len(gt_images)
    vertical_spacing = 20  # pixels between rows
    total_height = 2 * img_height + vertical_spacing
    
    # Create white canvas
    comparison = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    
    # Paste GT images on top row
    x_offset = 0
    for img in gt_images:
        comparison.paste(img, (x_offset, 0))
        x_offset += img_width
    
    # Paste Rendered images on bottom row
    x_offset = 0
    y_offset = img_height + vertical_spacing
    for img in rendered_images:
        comparison.paste(img, (x_offset, y_offset))
        x_offset += img_width
    
    # Save result
    if output_path is None:
        output_path = os.path.join(vis_dir, 'comparison.png')
    
    comparison.save(output_path)
    print(f"✅ Comparison figure saved to: {output_path}")
    print(f"   Size: {total_width}x{total_height} pixels")
    print(f"   Layout: GT (top) vs Rendered (bottom)")
    print(f"   G-buffers: {', '.join(gbuffer_types[:len(gt_images)])}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/create_comparison_figure.py <vis_dir> [output_path]")
        print("Example: python scripts/create_comparison_figure.py output/car/visuals/iter_1")
        sys.exit(1)
    
    vis_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.isdir(vis_dir):
        print(f"Error: Directory not found: {vis_dir}")
        sys.exit(1)
    
    create_comparison_figure(vis_dir, output_path)

if __name__ == '__main__':
    main()

