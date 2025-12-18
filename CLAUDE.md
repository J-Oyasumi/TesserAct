# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TesserAct is a 4D Embodied World Model for robotics that generates RGB, depth, and normal videos from input images and text instructions. The model is built on top of CogVideoX-5B-I2V and supports both full fine-tuning (SFT) and LoRA fine-tuning approaches.

**Key capabilities:**
- Image-to-video generation conditioned on text instructions
- Multi-modal output: RGB, depth, and surface normal predictions
- Supports three robot types: Google Robot, Franka Emika Panda, Trossen WidowX 250
- Training on robotics datasets: RT1, Bridge, RLBench

## Installation

```bash
conda create -n tesseract python=3.9
conda activate tesseract
pip install -r requirements.txt
pip install -e .
```

For data processing, use a separate environment:
```bash
conda create -n tesseract-data python=3.10
conda activate tesseract-data
pip install -r requirements.txt
pip install git+https://github.com/anyeZHY/rollingdepth.git
```

## Common Commands

### Inference

**RGB+Depth+Normal generation (SFT model):**
```bash
python inference/inference_rgbdn_sft.py \
  --weights_path anyeZHY/tesseract/tesseract_v01e_rgbdn_sft \
  --image_path asset/images/fruit_vangogh.png \
  --prompt "pick up the apple google robot"
```

**RGB-only generation (LoRA model, best generalization):**
```bash
python inference/inference_rgb_lora.py \
  --weights_path anyeZHY/tesseract/tesseract_v01e_rgb_lora \
  --image_path asset/images/fruit_vangogh.png \
  --prompt "pick up the apple google robot"
```

**RGB+Depth+Normal generation (LoRA model):**
```bash
python inference/inference_rgbdn_lora.py \
  --base_weights_path anyeZHY/tesseract/tesseract_v01e_rgbdn_sft \
  --lora_weights_path ./your_local_lora_weights \
  --image_path asset/images/fruit_vangogh.png \
  --prompt "pick up the apple google robot"
```

**Generate depth/normal annotations with Marigold:**
```bash
python inference/inference_marigold.py --image_folder data/images
```

**Render point clouds from RGBD videos (requires Blender 4.3+):**
```bash
blender-4.4.3/blender -b -P scripts/rendering_points.py -- \
  --combined_video ./results/val_0_pick_up_the_apple_google_robot_0.mp4 \
  --render_output ./results/rendered_results
```

### Training

**Full fine-tuning (SFT):**
```bash
bash train_i2v_depth_normal_sft.sh
```

**LoRA fine-tuning (~30GB GPU memory, experimental):**
```bash
bash train_i2v_depth_normal_lora.sh
```

Both scripts use `torchrun` for distributed training and support multi-node training via SLURM environment variables.

### Data Processing Pipeline

See [DATA.md](DATA.md) for complete details. Basic workflow:

1. Download raw dataset (e.g., Bridge):
```bash
wget https://rail.eecs.berkeley.edu/datasets/bridge_release/data/demos_8_17.zip
mv demos_8_17.zip data && cd data && unzip demos_8_17.zip && cd ..
```

2. Preprocess dataset:
```bash
python scripts/preprocess_bridge.py
```

3. Generate depth maps:
```bash
python scripts/video_depth.py -i data/bridge/processed -o data/bridge/processed --verbose
```

4. Generate normal maps:
```bash
python scripts/video_normal.py --dataset bridge --num_gpus 8
```

## Architecture

### Core Model Components

**TesserActDepthNormal** ([tesseract/modules/tesseract_model.py](tesseract/modules/tesseract_model.py)):
- Extends CogVideoXTransformer3DModel with additional input/output channels for depth and normal
- Processes RGB+Depth+Normal data (9 input channels total, mapped to 48 latent channels via VAE)
- Custom patch embedding layers for each modality (RGB, depth, normal)
- Supports LoRA adapters on attention layers and patch embeddings

**TesserActImageToDepthNormalVideoPipeline** ([tesseract/modules/tesseract_pipeline.py](tesseract/modules/tesseract_pipeline.py)):
- Modified CogVideoX pipeline for RGB+Depth+Normal generation
- Handles 9-channel inputs (3 RGB + 3 depth + 3 normal)
- Splits outputs back into separate modalities for visualization

### Training Architecture

The training process uses a multi-modal approach where RGB, depth, and normal are treated as separate channels but trained jointly:

1. **Input processing**: Images are encoded as [RGB, Depth, Normal] with shape [B, 9, H, W]
2. **VAE encoding**: Each modality is encoded separately then concatenated in latent space
3. **Conditioning**: Text prompts are encoded with T5, concatenated with robot type
4. **Diffusion**: Standard noise prediction on the concatenated latent space
5. **Loss computation**: Separate losses for RGB, depth, and normal predictions with masking support

**Key training files:**
- [tesseract/i2v_depth_normal_sft.py](tesseract/i2v_depth_normal_sft.py): Full fine-tuning
- [tesseract/i2v_depth_normal_lora.py](tesseract/i2v_depth_normal_lora.py): LoRA fine-tuning
- [tesseract/robodataset.py](tesseract/robodataset.py): Dataset loader with bucketing for efficient training

### Dataset Structure

The dataset uses a bucket-based approach to handle varying video resolutions and lengths efficiently:

```python
DATASET2ROBOT = {
    "fractal20220817_data": "google robot",
    "bridge": "Trossen WidowX 250 robot arm",
    "rlbench": "Franka Emika Panda",
}
```

Expected data structure after processing:
```
data/bridge/processed/
├── <scene_id>/
│   ├── video/
│   │   ├── rgb.mp4
│   │   └── normal.mp4
│   ├── depth/
│   │   └── npz/
│   │       └── depth.npz
│   └── instruction.txt
```

The dataset file (`cache/samples_depth_normal.json`) contains entries with paths relative to `data_root`.

## Important Implementation Details

### Prompt Format
Prompts must follow the format: `[Instruction] + [Robot Name]`

Supported robot names:
- `google robot`
- `Franka Emika Panda`
- `Trossen WidowX 250 robot arm`

Example: `"pick up the apple google robot"`

### Depth and Normal Map Format

**Depth maps** (`*_depth.npy`):
- Stored as 1-channel float numpy array
- Range: [0, 1] where 0 is far, 1 is near
- For asset images: depth is inverted (1 - depth) before use

**Normal maps** (`*_normal.png`):
- 3-channel PNG format
- RGB channels represent X, Y, Z components of normal vector
- Generated using Temporal Marigold for temporal consistency

### Model Resolution
- Default: 480x640 (height x width)
- Model automatically resizes inputs to this resolution
- Other resolutions may not work well with current model

### Training Configuration

**SFT training:**
- Base model: CogVideoX-5B-I2V
- Mixed precision: bf16
- Adds 16*4 = 64 additional input channels for depth+normal conditioning
- Uses gradient checkpointing, VAE slicing/tiling for memory efficiency
- Learned positional embeddings are disabled (`ignore_learned_positional_embeddings`)

**LoRA training:**
- Rank: 512, Alpha: 512
- Targets: `to_k`, `to_q`, `to_v`, `to_out.0`, patch embedding projections
- Different learning rates for LoRA vs. non-LoRA parameters (depth/normal output heads get 3x LR)
- Requires pretrained SFT model as base

### Distributed Training Setup

Both training scripts detect SLURM environment and configure distributed training:
- `NUM_GPUS`: Auto-detected via `nvidia-smi`
- `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `NODE_RANK`: From SLURM env vars
- Uses NCCL backend with extended timeouts (1800s) for stability
- Important flags: `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` for compatibility

### Validation and Checkpointing
- Validation runs every `validation_steps` (250 for SFT, 200 for LoRA)
- Checkpoints saved every `checkpointing_steps` with limit of 15 checkpoints
- Supports resuming from `latest` checkpoint automatically
- Validation prompts/images specified via `:::` separator for multiple examples

## Model Versions

Current available models on HuggingFace (`anyeZHY/tesseract`):

| Model | Description | Memory | Training |
|-------|-------------|--------|----------|
| `tesseract_v01e_rgbdn_sft` | RGB+Depth+Normal, full finetuned from CogVideoX-5B | 25-29GB | 40k steps on RT1+Bridge+RLBench |
| `tesseract_v01e_rgb_lora` | RGB-only with LoRA (best generalization) | 25-29GB | 12k steps |

The `v01e` suffix indicates experimental version. Production models will use `v##p` naming.

## Known Limitations

- Normal data quality depends on Temporal Marigold; considering upgrade to NormalCrafter
- Current model only tested at 640x480 resolution
- Requires specific robot names in prompts for proper conditioning
- LoRA fine-tuning is experimental and not fully validated
- Point cloud rendering requires Blender 4.3+ and PyBlend setup
