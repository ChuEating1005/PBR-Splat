# PBR-Splat

A modified version of [GIR](https://github.com/guduxiaolang/GIR) for G-buffer supervision training using 3D Gaussian Splatting.

## Overview

This project extends GIR to support direct G-buffer supervision for training PBR (Physically Based Rendering) parameters. Instead of relying on RGB image loss, we use ground truth G-buffers (albedo, normal, metallic, roughness) from renderers like DiffusionRender.

**Key Features:**
- Direct G-buffer supervision (albedo, normal, metallic, roughness)
- Foreground mask support for training on objects only
- Configurable L1/L2 loss for different training needs
- Single-stage training without multi-stage complexity
- Compatible with standard 3DGS checkpoints for initialization

## Environment Setup

### 1. Set Compiler and CUDA Environment Variables

```bash
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
```

**Note**: These need to be set before compiling submodules. For permanent setup, add them to `~/.bashrc` or conda activation script.

### 2. Create Conda Environment

```bash
conda create -n gir python=3.7 -y
conda activate gir
```

### 3. Initialize Git Submodules

```bash
git submodule update --init --recursive
```

This will clone the following dependencies:
- `simple-knn` - Fast KNN computation
- `envlight` - Environment lighting toolkit
- `nvdiffrast` - Differentiable rasterization

**Note**: `diff-gaussian-rasterization` is now directly included (GIR modified version with depth/alpha support).

### 4. Install PyTorch

```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116
```

### 5. Install Python 3.7 Backport Packages

```bash
pip install importlib-metadata
```

### 6. Install Submodules

All submodules require `--no-build-isolation` to use the current environment's torch:

```bash
pip install -e submodules/diff-gaussian-rasterization --no-build-isolation
pip install -e submodules/simple-knn --no-build-isolation
pip install -e submodules/envlight --no-build-isolation
pip install -e submodules/nvdiffrast --no-build-isolation
```

### 7. Install Other Dependencies

```bash
pip install tqdm plyfile einops imageio[full] tensorboard gdown
```

**Package purposes:**
- `tensorboard` - Training visualization and logging
- `gdown` - Download data from Google Drive (optional)

### 8. Verify Installation

```bash
python -c "import torch; import diff_gaussian_rasterization; import nvdiffrast; import einops; print('✅ All packages installed successfully!')"
python train_gbuffer.py --help
```

## Data Preparation

Your dataset should follow the COLMAP structure with additional G-buffers:

```
data/<scene_name>/
├── sparse/0/
│   ├── cameras.txt       # COLMAP camera parameters
│   └── images.txt        # COLMAP image list
├── images/               # Training images
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── gbuffers/             # Ground truth G-buffers
│   ├── 0000.0000.basecolor.png
│   ├── 0000.0000.normal.png
│   ├── 0000.0000.metallic.png
│   ├── 0000.0000.roughness.png
│   └── ...
├── masks/                # Foreground masks (optional but recommended)
│   ├── 00000.png        # White = foreground, Black = background
│   ├── 00001.png
│   └── ...
└── point_cloud/          # Pre-trained 3DGS checkpoint (optional)
    └── iteration_30000/
        └── point_cloud.ply
```

**Naming Conventions:**

- **G-buffers**: `0000.{image_idx:04d}.{type}.png`
  - `basecolor` → albedo
  - `normal` → surface normal
  - `metallic` → metallic parameter
  - `roughness` → roughness parameter

- **Masks**: `{image_idx:05d}.png`
  - White pixels (255) = foreground object
  - Black pixels (0) = background
  - Used to mask out background during training

## Training

### Basic Training

```bash
python train_gbuffer.py \
    -s data/<scene_name> \
    -m output/<scene_name> \
    --iterations 30000
```

### Training from Checkpoint

If you have a pre-trained 3DGS checkpoint:

```bash
python train_gbuffer.py \
    -s data/<scene_name> \
    -m output/<scene_name> \
    --start_checkpoint data/<scene_name>/point_cloud/iteration_30000/point_cloud.ply \
    --iterations 30000
```

### Using Training Script

```bash
bash scripts/train_gbuffer.sh
```

## Loss Configuration

The training script uses L1 loss by default. You can switch to L2 loss by editing `train_gbuffer.py`:

```python
# Line ~135 in train_gbuffer.py
use_l2_loss = False  # Set to True to use L2 loss instead of L1
```

**Recommended Settings:**
- L1 loss (default): Better for sharp details and edges
- L2 loss: Better for smoother gradients and less sensitive to outliers

## Visualization

### During Training

G-buffers are automatically saved every 500 iterations to:
```
output/<scene_name>/visuals/iter_<N>/
├── rendered_albedo.png
├── rendered_normal.png
├── rendered_metallic.png
├── rendered_roughness.png
├── gt_albedo_masked.png      # GT with background masked out
├── gt_normal_masked.png
├── gt_metallic_masked.png
├── gt_roughness_masked.png
└── mask.png
```

### Create Comparison Figures

Generate side-by-side comparison images:

```bash
python create_comparison_figure.py output/<scene_name>/visuals/iter_<N>
```

This creates `comparison.png` with GT (top row) vs Rendered (bottom row).

### Tensorboard

View training progress in real-time:

```bash
tensorboard --logdir output/<scene_name>
```

Logs include:
- Loss curves (total, albedo, normal, metallic, roughness)
- G-buffer visualizations (GT and rendered)

## Key Modifications

1. **G-buffer Supervision**: Direct loss calculation on albedo, normal, metallic, roughness instead of RGB
2. **Mask-based Training**: Supports foreground masks to focus training on objects (background set to black in GT)
3. **Configurable Loss Function**: Support for both L1 and L2 loss modes
4. **Simplified Training**: Single-stage training focused on G-buffer parameters (no multi-stage complexity)
5. **Checkpoint Compatibility**: Supports loading from standard 3DGS PLY checkpoints for initialization
6. **GIR Modified Rasterizer**: Uses GIR's modified `diff-gaussian-rasterization` that outputs depth and alpha channels
7. **Visualization Tools**: Automatic G-buffer visualization during training + comparison script
8. **Python 3.7 Support**: Backport compatibility for `importlib.metadata`

## Training Tips

1. **Use a pre-trained 3DGS checkpoint**: Training from scratch is difficult. Always start with a converged standard 3DGS model.

2. **Mask quality matters**: High-quality foreground masks significantly improve results by preventing background interference.

3. **Monitor loss curves**: Check Tensorboard to ensure all G-buffer losses are decreasing. If one component isn't improving, adjust its weight.

4. **Iteration count**: 
   - Quick test: 5,000 iterations (~5-10 minutes)
   - Standard training: 15,000-30,000 iterations
   - Fine-tuning: 10,000-15,000 iterations from checkpoint

5. **Loss type selection**: 
   - L1 loss (default): Preserves sharp edges and fine details
   - L2 loss: Produces smoother results, better for noisy data

## Troubleshooting

**Problem: Loss not decreasing**
- Check if masks are loaded correctly (should print at startup)
- Verify GT G-buffers are in correct format and location
- Try reducing learning rate in `arguments/__init__.py`

**Problem: Background artifacts**
- Ensure masks are binary (white=255, black=0)
- Check that mask naming matches image indices

**Problem: Normal map looks wrong**
- Normals should be in world space, range [0, 1] (mapped from [-1, 1])
- Check GT normal map orientation

**Problem: CUDA out of memory**
- Reduce batch size or image resolution in arguments
- Use smaller initial checkpoint

## Acknowledgements

- [GIR](https://github.com/guduxiaolang/GIR) - Original implementation with modified diff-gaussian-rasterization
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Base 3DGS implementation
- [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) - Official Gaussian rasterization (modified by GIR)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast) - Differentiable rasterization
- [envlight](https://github.com/ashawkey/envlight) - Environment lighting toolkit
- [simple-knn](https://github.com/camenduru/simple-knn) - Fast KNN implementation
