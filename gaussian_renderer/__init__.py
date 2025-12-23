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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import RGB2SH

def create_rasterizer(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0):
    """
    Helper function to create rasterization settings and rasterizer.
    
    Returns:
        tuple: (rasterizer, raster_settings)
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    return rasterizer, raster_settings


def rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                        colors_precomp=None, shs=None, cov3D_precomp=None):
    """
    Helper function to perform Gaussian rasterization.
    
    Returns:
        tuple: (rendered_image, radii, depth, alpha)
    """
    return rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )


def render_gbuffer(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Simplified render function for G-buffer training.
    Directly renders Albedo, Normal, Metallic, Roughness without PBR computation.
    Uses GIR modified rasterizer that returns 4 values: color, radii, depth, alpha.
    """
    # Create screenspace points
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Create rasterizer
    rasterizer, _ = create_rasterizer(viewpoint_camera, pc, pipe, bg_color, scaling_modifier)
    
    # Common parameters
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # Render Albedo
    rendered_albedo, radii_albedo, depth_albedo, alpha_albedo = rasterize_gaussians(
        rasterizer, means3D, means2D, opacity, scales, rotations,
        colors_precomp=pc.get_albedo_init)

    # Render Normal
    render_normal = (pc.get_eigenvector + 1) / 2
    rendered_normal, *_ = rasterize_gaussians(
        rasterizer, means3D, means2D, opacity, scales, rotations,
        colors_precomp=render_normal)

    # Render Material (Metallic & Roughness)
    render_material = torch.cat([
        pc.get_metallic_init, 
        pc.get_roughness_init.clamp(0.08, 0.5), 
        torch.zeros((means3D.shape[0], 1), device="cuda")
    ], -1)
    
    rendered_material, *_ = rasterize_gaussians(
        rasterizer, means3D, means2D, opacity, scales, rotations,
        colors_precomp=render_material)
    
    rendered_metallic = rendered_material[0:1, ...].repeat(3, 1, 1)
    rendered_roughness = rendered_material[1:2, ...].repeat(3, 1, 1)

    return {
        "rendered_albedo": rendered_albedo,
        "rendered_normal": rendered_normal,
        "rendered_metallic": rendered_metallic,
        "rendered_roughness": rendered_roughness,
        "depth": depth_albedo,
        "alpha": alpha_albedo,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii_albedo > 0,
        "radii": radii_albedo
    }

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, random_bg_color = None, iteration=None, scaling_modifier = 1.0, is_train=None, first_stage_step=5000, second_stage_step=30000, remove_noise=False, hdr_rotation=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create screenspace points
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Create rasterizers
    rasterizer, raster_settings = create_rasterizer(viewpoint_camera, pc, pipe, bg_color, scaling_modifier)
    
    if random_bg_color is not None:
        random_rasterizer, _ = create_rasterizer(viewpoint_camera, pc, pipe, random_bg_color, scaling_modifier)
    else:
        random_rasterizer = rasterizer

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    diffuse_color = None
    specular_indirect_light = None
    specular_indirect_color = None
    if iteration <= first_stage_step:
        colors_precomp = pc.get_albedo_init
        if iteration == first_stage_step:
            pc._albedo_init.data = torch.zeros_like(pc._albedo_init)
    else:
        if iteration == second_stage_step+1:
            pc._albedo_init.data = torch.zeros_like(pc._albedo_init)
            pc._features_dc.data = torch.zeros_like(pc._features_dc)
            pc._features_rest.data = torch.zeros_like(pc._features_rest)
            pc._metallic_init.data = torch.rand_like(pc._metallic_init) * 0.2
            pc._roughness_init.data = torch.rand_like(pc._roughness_init)
        result = pc.compute_color(viewpoint_camera.camera_center, iteration, is_train, first_stage_step, second_stage_step, remove_noise, hdr_rotation, viewpoint_camera.exposure)

        colors_precomp = result["color"]
        albedo = result["albedo"]
        diffuse_albedo = result["diffuse_albedo"]
        diffuse_light = result["diffuse_light"]
        diffuse_color = result["diffuse_color"]
        specular_albedo = result["specular_albedo"]
        specular_indirect_light = result["specular_indirect_light"]
        specular_direct_light = result["specular_direct_light"]
        specular_indirect_color = result["specular_indirect_color"]
        specular_direct_color = result["specular_direct_color"]
        specular_light = result["specular_light"]
        specular_color = result["specular_color"]
        occ = result["occ"]

    rendered_image, radii, depth, alpha = rasterize_gaussians(
        random_rasterizer, means3D, means2D, opacity, scales, rotations,
        colors_precomp=colors_precomp, shs=shs, cov3D_precomp=cov3D_precomp)
    
    rendered_normal = None
    rendered_metallic = None
    rendered_roughness = None
    rendered_albedo = None
    rendered_diffuse_color = None
    rendered_specular_color = None
    rendered_diffuse_albedo=None
    rendered_specular_albedo=None
    rendered_diffuse_light = None
    rendered_specular_light = None
    rendered_specular_indirect_color = None
    rendered_specular_indirect_light = None
    rendered_specular_direct_light = None
    rendered_specular_direct_color = None
    rendered_occ = None
    if iteration > second_stage_step:
        render_normal = (pc.get_eigenvector + 1) / 2
        render_material = torch.cat([pc.get_metallic_init, pc.get_roughness_init.clamp(0.08, 0.5), torch.zeros((render_normal.shape[0],1), device="cuda")], -1)
        
        rendered_normal, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                   colors_precomp=render_normal, shs=shs, cov3D_precomp=cov3D_precomp)
        rendered_material, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                     colors_precomp=render_material, shs=shs, cov3D_precomp=cov3D_precomp)
        rendered_metallic = rendered_material[0:1,...].repeat(3,1,1)
        rendered_roughness = rendered_material[1:2,...].repeat(3,1,1)
        rendered_albedo, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                   colors_precomp=albedo, shs=shs, cov3D_precomp=cov3D_precomp)
    elif iteration > first_stage_step:
        if iteration % 500 == 0:
            with torch.no_grad():
                render_normal = (pc.get_eigenvector + 1) / 2
                render_material = torch.cat([pc.get_metallic_init, pc.get_roughness_init.clamp(0.08, 0.5), torch.zeros((render_normal.shape[0],1), device="cuda")], -1)
                
                rendered_normal, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                          colors_precomp=render_normal, shs=shs, cov3D_precomp=cov3D_precomp)
                rendered_material, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                            colors_precomp=render_material, shs=shs, cov3D_precomp=cov3D_precomp)
                rendered_metallic = rendered_material[0:1,...].repeat(3,1,1)
                rendered_roughness = rendered_material[1:2,...].repeat(3,1,1)
                rendered_albedo, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                          colors_precomp=albedo, shs=shs, cov3D_precomp=cov3D_precomp)
    if iteration % 500 == 0:
        with torch.no_grad():
            if diffuse_color is not None:
                rendered_diffuse_color, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                                  colors_precomp=diffuse_color, shs=shs, cov3D_precomp=cov3D_precomp)
                rendered_specular_color, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                                   colors_precomp=specular_color, shs=shs, cov3D_precomp=cov3D_precomp)
                rendered_diffuse_albedo, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                                   colors_precomp=diffuse_albedo, shs=shs, cov3D_precomp=cov3D_precomp)
                rendered_specular_albedo, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                                    colors_precomp=specular_albedo, shs=shs, cov3D_precomp=cov3D_precomp)
                rendered_diffuse_light, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                                  colors_precomp=diffuse_light, shs=shs, cov3D_precomp=cov3D_precomp)
                rendered_specular_light, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                                   colors_precomp=specular_light, shs=shs, cov3D_precomp=cov3D_precomp)
            if specular_indirect_color is not None:
                rendered_specular_indirect_color, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                                            colors_precomp=specular_indirect_color, shs=shs, cov3D_precomp=cov3D_precomp)
                rendered_specular_direct_color, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                                          colors_precomp=specular_direct_color, shs=shs, cov3D_precomp=cov3D_precomp)
            if specular_indirect_light is not None:
                rendered_specular_indirect_light, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                                            colors_precomp=specular_indirect_light, shs=shs, cov3D_precomp=cov3D_precomp)
                rendered_specular_direct_light, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                                          colors_precomp=specular_direct_light, shs=shs, cov3D_precomp=cov3D_precomp)
                rendered_occ, *_ = rasterize_gaussians(rasterizer, means3D, means2D, opacity, scales, rotations, 
                                                        colors_precomp=occ.repeat(1,3), shs=shs, cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": depth,
            "alpha": alpha,
            "rendered_normal": rendered_normal,
            "rendered_albedo": rendered_albedo,
            "rendered_metallic": rendered_metallic,
            "rendered_roughness": rendered_roughness,
            "rendered_diffuse_color": rendered_diffuse_color,
            "rendered_specular_color": rendered_specular_color,
            "rendered_diffuse_light": rendered_diffuse_light,
            "rendered_specular_light": rendered_specular_light,
            "rendered_diffuse_albedo": rendered_diffuse_albedo,
            "rendered_specular_albedo": rendered_specular_albedo,
            "rendered_specular_indirect_light": rendered_specular_indirect_light,
            "rendered_specular_direct_light": rendered_specular_direct_light,
            "rendered_specular_indirect_color": rendered_specular_indirect_color,
            "rendered_specular_direct_color": rendered_specular_direct_color,
            "rendered_occ": rendered_occ,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
