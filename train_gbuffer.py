#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, smooth_loss, regularizer_loss, get_mask, tv_loss
from gaussian_renderer import render_gbuffer, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import torchvision
from envlight.utils import cubemap_to_latlong

# G-buffer Loss Wrapper Function
def compute_gbuffer_loss(rendered, gt, use_l2=False):
    """
    Compute G-buffer loss using L1 or L2.
    
    Args:
        rendered: Rendered G-buffer tensor
        gt: Ground truth G-buffer tensor
        use_l2: If True, use L2 loss; otherwise use L1 loss
    
    Returns:
        Loss value
    """
    if use_l2:
        # L2 loss
        loss = torch.mean((rendered - gt) ** 2)
    else:
        # L1 loss
        loss = l1_loss(rendered, gt)
    
    return loss

def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    smooth_normal_weight: float = 0.03,
    smooth_albedo_weight: float = 0.05,
    smooth_metallic_weight: float = 0.05,
    smooth_roughness_weight: float = 0.05,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    
    # Load checkpoint BEFORE creating Scene to avoid wasting time on dummy point cloud initialization
    checkpoint_loaded = False
    if checkpoint:
        # Handle different checkpoint formats
        if checkpoint.endswith('.ply'):
            # Standard 3DGS PLY checkpoint
            gaussians.load_ply_standard_3dgs(checkpoint)
            checkpoint_loaded = True
            first_iter = 0
        else:
            # GIR-style PyTorch checkpoint
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
            print(f"Restored from checkpoint at iteration {first_iter}")
            checkpoint_loaded = True
    else:
        print("Warning: No checkpoint provided! This script is intended for finetuning from existing 3DGS.")
    
    # Create Scene (Scene.__init__ now handles None point_cloud gracefully)
    scene = Scene(dataset, gaussians)
    
    if checkpoint_loaded:
        print(f"✅ Loaded {gaussians._xyz.shape[0]} Gaussians from checkpoint")
    
    # Check if masks are loaded
    num_with_mask = sum(1 for cam in scene.getTrainCameras() if cam.mask is not None)
    print(f"✅ Loaded masks for {num_with_mask}/{len(scene.getTrainCameras())} training cameras")
    
    # Setup training after loading checkpoint to ensure correct parameter counts
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        # Update learning rate (simplified, maybe we only want to optimize PBR params)
        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render G-buffers
        render_pkg = render_gbuffer(viewpoint_cam, gaussians, pipe, background)
        
        rendered_albedo = render_pkg["rendered_albedo"]
        rendered_normal = render_pkg["rendered_normal"]
        rendered_metallic = render_pkg["rendered_metallic"]
        rendered_roughness = render_pkg["rendered_roughness"]

        
        
        # Loss Calculation with Mask
        gt_gbuffers = viewpoint_cam.original_gbuffers
        mask = viewpoint_cam.mask  # Get foreground mask
                
        loss = 0.0
        loss_albedo = loss_normal = loss_metallic = loss_roughness = 0.0
        
        # Configuration: use L1 or L2 loss
        use_l2_loss = False  # Set to True to use L2 loss instead of L1
        
        if gt_gbuffers:
            # Apply mask to GT: set background to black (0)
            if mask is not None:
                masked_gt_gbuffers = {}
                for key, gt in gt_gbuffers.items():
                    # Set background (mask == 0) to black color (0)
                    masked_gt_gbuffers[key] = gt * mask
                
                # Compute loss on full image (including black background)
                loss_albedo = compute_gbuffer_loss(rendered_albedo, masked_gt_gbuffers["albedo"], use_l2=use_l2_loss)
                loss_normal = compute_gbuffer_loss(rendered_normal, masked_gt_gbuffers["normal"], use_l2=use_l2_loss)
                loss_metallic = compute_gbuffer_loss(rendered_metallic, masked_gt_gbuffers["metallic"], use_l2=use_l2_loss)
                loss_roughness = compute_gbuffer_loss(rendered_roughness, masked_gt_gbuffers["roughness"], use_l2=use_l2_loss)
                loss = loss_albedo + loss_normal + loss_metallic + loss_roughness
            else:
                # Fallback to regular loss if no mask
                loss_albedo = compute_gbuffer_loss(rendered_albedo, gt_gbuffers["albedo"], use_l2=use_l2_loss)
                loss_normal = compute_gbuffer_loss(rendered_normal, gt_gbuffers["normal"], use_l2=use_l2_loss)
                loss_metallic = compute_gbuffer_loss(rendered_metallic, gt_gbuffers["metallic"], use_l2=use_l2_loss)
                loss_roughness = compute_gbuffer_loss(rendered_roughness, gt_gbuffers["roughness"], use_l2=use_l2_loss)
                loss = loss_albedo + loss_normal + loss_metallic + loss_roughness
            
            # Log individual losses to tensorboard
            if iteration % 10 == 0 and tb_writer:
                tb_writer.add_scalar('Loss/albedo', loss_albedo.item() if isinstance(loss_albedo, torch.Tensor) else loss_albedo, iteration)
                tb_writer.add_scalar('Loss/normal', loss_normal.item() if isinstance(loss_normal, torch.Tensor) else loss_normal, iteration)
                tb_writer.add_scalar('Loss/metallic', loss_metallic.item() if isinstance(loss_metallic, torch.Tensor) else loss_metallic, iteration)
                tb_writer.add_scalar('Loss/roughness', loss_roughness.item() if isinstance(loss_roughness, torch.Tensor) else loss_roughness, iteration)
                tb_writer.add_scalar('Loss/total', loss.item(), iteration)
        else:
            # Fallback if no GT G-buffer found (should warn)
             if iteration % 100 == 0:
                print("Warning: No GT G-buffer found for current view.")

        # Regularization (optional, inspired by GIR stage2)
        # These losses encourage smoother/less noisy G-buffer predictions.
        reg_loss = 0.0
        if mask is not None:
            albedo_reg = rendered_albedo * mask
            normal_reg = rendered_normal * mask
            metallic_reg = rendered_metallic * mask
            roughness_reg = rendered_roughness * mask
        else:
            albedo_reg = rendered_albedo
            normal_reg = rendered_normal
            metallic_reg = rendered_metallic
            roughness_reg = rendered_roughness

        # Prepare guide image for edge-aware smoothness
        # Use original RGB image if available, otherwise rendered albedo
        guide_img = None
        if hasattr(viewpoint_cam, "original_image") and viewpoint_cam.original_image is not None:
            guide_img = viewpoint_cam.original_image.cuda()
            guide_img = guide_img[0:3, ...] if guide_img.shape[0] >= 3 else guide_img
        else:
            guide_img = rendered_albedo.detach()
        
        # Mask guide image if needed
        if mask is not None:
            guide_img = guide_img * mask

        # Unsqueeze guide for smooth_loss which expects (B, C, H, W)
        guide_img_batch = guide_img.unsqueeze(0)
        
        # Smoothness Losses (Edge-Aware)
        reg_loss = reg_loss + smooth_loss(normal_reg.unsqueeze(0), guide_img_batch) * smooth_normal_weight
        reg_loss = reg_loss + smooth_loss(albedo_reg.unsqueeze(0), guide_img_batch) * smooth_albedo_weight
        reg_loss = reg_loss + smooth_loss(metallic_reg.unsqueeze(0), guide_img_batch) * smooth_metallic_weight
        reg_loss = reg_loss + smooth_loss(roughness_reg.unsqueeze(0), guide_img_batch) * smooth_roughness_weight

        if reg_loss != 0.0:
            loss = loss + reg_loss
            if iteration % 10 == 0 and tb_writer:
                tb_writer.add_scalar('Loss/reg_total', reg_loss.item() if isinstance(reg_loss, torch.Tensor) else float(reg_loss), iteration)

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration % 1000 == 0:
                 training_report(tb_writer, iteration, loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_gbuffer, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Save rendered G-buffers periodically for visualization
            if iteration % 500 == 0 or iteration == 1:
                import os
                from torchvision.utils import save_image
                vis_dir = os.path.join(scene.model_path, f"visuals", f"iter_{iteration}")
                os.makedirs(vis_dir, exist_ok=True)
                
                # Save rendered G-buffers
                save_image(rendered_albedo, os.path.join(vis_dir, f"rendered_albedo.png"))
                save_image(rendered_normal, os.path.join(vis_dir, f"rendered_normal.png"))
                save_image(rendered_metallic, os.path.join(vis_dir, f"rendered_metallic.png"))
                save_image(rendered_roughness, os.path.join(vis_dir, f"rendered_roughness.png"))
                
                # Save mask if available
                if mask is not None:
                    save_image(mask, os.path.join(vis_dir, f"mask.png"))
                
                # Save GT (masked version used for training) for comparison
                if gt_gbuffers:
                    if mask is not None:
                        # Save masked GT (background set to black)
                        for key, value in gt_gbuffers.items():
                            masked_gt = value * mask
                            save_image(masked_gt, os.path.join(vis_dir, f"gt_{key}_masked.png"))
                    else:
                        # Save original GT if no mask
                        for key, value in gt_gbuffers.items():
                            save_image(value, os.path.join(vis_dir, f"gt_{key}.png"))
                
                # Log to tensorboard
                if tb_writer:
                    tb_writer.add_image('Rendered/albedo', rendered_albedo, iteration)
                    tb_writer.add_image('Rendered/normal', rendered_normal, iteration)
                    tb_writer.add_image('Rendered/metallic', rendered_metallic, iteration)
                    tb_writer.add_image('Rendered/roughness', rendered_roughness, iteration)
                    if gt_gbuffers:
                        if "albedo" in gt_gbuffers:
                            tb_writer.add_image('GT/albedo', gt_gbuffers["albedo"], iteration)
                        if "normal" in gt_gbuffers:
                            tb_writer.add_image('GT/normal', gt_gbuffers["normal"], iteration)
                        if "metallic" in gt_gbuffers:
                            tb_writer.add_image('GT/metallic', gt_gbuffers["metallic"], iteration)
                        if "roughness" in gt_gbuffers:
                            tb_writer.add_image('GT/roughness', gt_gbuffers["roughness"], iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    
                    # Log images to tensorboard
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render_albedo".format(viewpoint.image_name), render_pkg["rendered_albedo"][None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_normal".format(viewpoint.image_name), render_pkg["rendered_normal"][None], global_step=iteration)
                        
                        if viewpoint.original_gbuffers:
                             if "albedo" in viewpoint.original_gbuffers:
                                tb_writer.add_images(config['name'] + "_view_{}/gt_albedo".format(viewpoint.image_name), viewpoint.original_gbuffers["albedo"][None], global_step=iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    # Optional regularization weights (inspired by GIR stage2) to encourage smoother G-buffers
    parser.add_argument("--smooth_normal_weight", type=float, default=0.01)
    parser.add_argument("--smooth_albedo_weight", type=float, default=0.0)
    parser.add_argument("--smooth_metallic_weight", type=float, default=0.0)
    parser.add_argument("--smooth_roughness_weight", type=float, default=0.0)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # We ignore first/second stage parameters as we are doing G-buffer training only
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        smooth_normal_weight=args.smooth_normal_weight,
        smooth_albedo_weight=args.smooth_albedo_weight,
        smooth_metallic_weight=args.smooth_metallic_weight,
        smooth_roughness_weight=args.smooth_roughness_weight,
    )

    # All done
    print("\nTraining complete.")

