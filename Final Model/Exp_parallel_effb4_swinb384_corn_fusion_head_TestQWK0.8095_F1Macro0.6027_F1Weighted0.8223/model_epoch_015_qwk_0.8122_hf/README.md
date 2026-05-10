---
library_name: pytorch
pipeline_tag: image-classification
tags:
- diabetic-retinopathy
- image-classification
- pytorch
- timm
- corn
---

# exp_parallel_effb4_swinb384_corn_fusion_head

Custom PyTorch model for diabetic retinopathy grading.

## Architecture

- EfficientNet branch: `efficientnet_b4.ra2_in1k`
- Swin Transformer branch: `swin_base_patch4_window12_384.ms_in22k_ft_in1k`
- Fusion head: LayerNorm → Linear → GELU → Dropout → CORN output
- Number of DR classes: `5`
- Number of CORN outputs: `4`

## Exported checkpoint

- Epoch: `15`
- Validation QWK: `0.8122`

## Labels

| Class ID | Label |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Important note

This model is for research purposes only and should not be used for clinical diagnosis.
