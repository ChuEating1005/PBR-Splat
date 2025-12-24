#!/usr/bin/env python3
#
# Render G-buffer video from trained model
#

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path

from scene import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render_gbuffer
from arguments import ModelParams, PipelineParams
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat
from utils.graphics_utils import getWorld2View2, focal2fov


def load_colmap_cameras(sparse_dir, image_dir=None):
    """
    Load camera parameters from COLMAP sparse reconstruction.
    
    Args:
        sparse_dir: Path to COLMAP sparse folder (e.g., data/refnerf/car/sparse/0/)
        image_dir: Path to images folder (optional, for loading actual images)
    
    Returns:
        List of Camera objects
    """
    cameras_file = os.path.join(sparse_dir, "cameras.txt")
    images_file = os.path.join(sparse_dir, "images.txt")
    
    # Read intrinsics and extrinsics
    cam_intrinsics = read_intrinsics_text(cameras_file)
    cam_extrinsics = read_extrinsics_text(images_file)
    
    cameras = []
    
    print(f"Loading {len(cam_extrinsics)} cameras from COLMAP...")
    
    for idx, key in enumerate(sorted(cam_extrinsics.keys())):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        
        # Extract camera parameters
        height = intr.height
        width = intr.width
        
        # Get rotation and translation
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        
        # Get focal length and compute FoV
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
        else:
            raise ValueError(f"Unsupported camera model: {intr.model}")
        
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        
        # Create dummy image (we don't need actual images for rendering)
        dummy_image = torch.zeros((3, height, width), dtype=torch.float32, device="cuda")
        
        # Create Camera object
        cam = Camera(
            colmap_id=key,
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            image=dummy_image,
            gt_alpha_mask=None,
            image_name=extr.name,
            uid=idx,
            data_device="cuda"
        )
        
        cameras.append(cam)
    
    print(f"✅ Loaded {len(cameras)} cameras")
    return cameras


def load_model(checkpoint_path, sh_degree=3):
    """
    Load trained Gaussian model from checkpoint.
    
    Args:
        checkpoint_path: Path to .ply or .pth checkpoint
        sh_degree: Spherical harmonics degree
    
    Returns:
        Loaded GaussianModel
    """
    print(f"Loading model from: {checkpoint_path}")
    
    gaussians = GaussianModel(sh_degree)
    
    if checkpoint_path.endswith('.ply'):
        # Try loading as GIR checkpoint first
        try:
            gaussians.load_ply(checkpoint_path)
            print(f"✅ Loaded GIR checkpoint with {gaussians._xyz.shape[0]} Gaussians")
        except ValueError as e:
            # If it fails, try loading as standard 3DGS checkpoint
            if "no field of name albedo_r" in str(e):
                print("Checkpoint is standard 3DGS format, loading with random PBR initialization...")
                gaussians.load_ply_standard_3dgs(checkpoint_path)
                print(f"✅ Loaded standard 3DGS checkpoint with {gaussians._xyz.shape[0]} Gaussians")
            else:
                raise e
    elif checkpoint_path.endswith('.pth'):
        # Load PyTorch checkpoint
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, tuple):
            model_params, iteration = checkpoint
            print(f"Loaded checkpoint from iteration {iteration}")
        else:
            model_params = checkpoint
        
        # Restore model
        from arguments import OptimizationParams
        dummy_parser = ArgumentParser(description="Rendering parameters")
        opt = OptimizationParams(dummy_parser)
        gaussians.restore(model_params, opt)
        print(f"✅ Loaded PyTorch checkpoint with {gaussians._xyz.shape[0]} Gaussians")
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    
    return gaussians


def save_gbuffer_image(tensor, output_path, gbuffer_type):
    """
    Save G-buffer tensor as image.
    
    Args:
        tensor: [C, H, W] tensor
        output_path: Output file path
        gbuffer_type: Type of G-buffer (albedo, normal, metallic, roughness)
    """
    # Move to CPU and convert to numpy
    img = tensor.detach().cpu().clamp(0, 1).numpy()
    
    # Handle different channel counts
    if img.shape[0] == 1:
        # Single channel (metallic, roughness)
        img = np.repeat(img, 3, axis=0)  # Convert to RGB for visualization
    elif img.shape[0] == 3:
        # RGB (albedo, normal)
        pass
    else:
        raise ValueError(f"Unexpected number of channels: {img.shape[0]}")
    
    # Transpose to HWC and convert to uint8
    img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
    
    # Save image
    Image.fromarray(img).save(output_path)


def render_video(args):
    """
    Main rendering function.
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each G-buffer type
    gbuffer_types = ['albedo', 'normal', 'metallic', 'roughness']
    for gtype in gbuffer_types:
        (output_dir / gtype).mkdir(exist_ok=True)
    
    # Load model
    gaussians = load_model(args.checkpoint, sh_degree=args.sh_degree)
    
    # Load cameras
    cameras = load_colmap_cameras(args.sparse_dir)
    
    # Limit to specified number of frames
    if args.num_frames > 0:
        cameras = cameras[:args.num_frames]
        print(f"Rendering first {len(cameras)} frames")
    
    # Setup rendering
    from arguments import PipelineParams
    dummy_parser = ArgumentParser(description="Rendering parameters")
    pipe = PipelineParams(dummy_parser)
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Render each frame
    print(f"\nRendering {len(cameras)} frames...")
    for idx, camera in enumerate(tqdm(cameras)):
        # Render G-buffers
        render_pkg = render_gbuffer(camera, gaussians, pipe, background)
        
        # Save each G-buffer type
        frame_name = f"{idx:05d}.png"
        
        save_gbuffer_image(
            render_pkg["rendered_albedo"],
            output_dir / "albedo" / frame_name,
            "albedo"
        )
        
        save_gbuffer_image(
            render_pkg["rendered_normal"],
            output_dir / "normal" / frame_name,
            "normal"
        )
        
        save_gbuffer_image(
            render_pkg["rendered_metallic"],
            output_dir / "metallic" / frame_name,
            "metallic"
        )
        
        save_gbuffer_image(
            render_pkg["rendered_roughness"],
            output_dir / "roughness" / frame_name,
            "roughness"
        )
    
    print(f"\n✅ Rendering complete! Output saved to: {output_dir}")
    print(f"   - albedo/")
    print(f"   - normal/")
    print(f"   - metallic/")
    print(f"   - roughness/")


def main():
    parser = ArgumentParser(description="Render G-buffer video from trained model")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.ply or .pth)")
    parser.add_argument("--sparse_dir", type=str, required=True,
                        help="Path to COLMAP sparse reconstruction (e.g., data/refnerf/car/sparse/0/)")
    parser.add_argument("--output_dir", type=str, default="output/rendered_video",
                        help="Output directory for rendered frames")
    
    # Rendering parameters
    parser.add_argument("--num_frames", type=int, default=120,
                        help="Number of frames to render (0 = all cameras)")
    parser.add_argument("--sh_degree", type=int, default=3,
                        help="Spherical harmonics degree")
    parser.add_argument("--white_background", action="store_true",
                        help="Use white background instead of black")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.sparse_dir):
        print(f"❌ Error: Sparse directory not found: {args.sparse_dir}")
        return
    
    # Render video
    render_video(args)


if __name__ == "__main__":
    main()

