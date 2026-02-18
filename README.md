# ComfyUI Depth Nodes (Local)

A ComfyUI custom node pack with three separate nodes, loaded from local files:

- Metric3Dv2 (local Metric3D repo + local checkpoint)
- UniDepth v2 (local pretrained folder)
- Facebook DepthLM Official style (local pretrained folder)

## Nodes

- `Metric3Dv2DepthNode`
- `UniDepthV2DepthNode`
- `DepthLMDepthNode`
- Category: `depth` for all
- Input image type: `IMAGE`
- Output type: `IMAGE` (grayscale depth visualization repeated to 3 channels)

## Inputs

Required:

- `image`
- `normalize_output` (bool)
- `invert_output` (bool)
- `min_depth_m` (float)
- `max_depth_m` (float)

Optional, per node:

- `Metric3Dv2DepthNode`: `focal_length_px`, `metric3d_variant`, `depth_models_root`, `metric3d_repo_dir`, `metric3d_checkpoint`
- `UniDepthV2DepthNode`: `depth_models_root`, `unidepth_model_id`
- `DepthLMDepthNode`: `depth_models_root`, `facebook_model_id`, `depthlm_point_x`, `depthlm_point_y`

## Install

1. Put this folder under your ComfyUI custom nodes directory.
2. Install dependencies in your ComfyUI Python environment:

```bash
pip install -r requirements.txt
```

With `uv`:

```bash
uv pip install -r requirements.txt
```

If Metric3Dv2 fails with `mmengine`/`mmcv` import errors, run:

```bash
pip install mmengine mmcv-lite
```

3. Restart ComfyUI.

## Notes

- Models are lazy-loaded and cached by backend/model/device.
- No backend downloads model weights at inference time in this implementation.
- The Facebook backend is implemented in official DepthLM query style: it draws a red arrow at the selected point and asks the model for numeric metric depth.
- DepthLM Official is point-depth oriented in this integration (single queried point), so output visualization is a constant depth image based on the predicted metric value.
- The backend uses `trust_remote_code=True` and transformers vision-language generation APIs.

## Local model layout (example)

```text
ComfyUI/
  models/
    depth_models/
      Metric3D/                  # local Metric3D repo
      metric3d_vit_small.pth     # local checkpoint (or set metric3d_checkpoint)
      unidepth-v2-vitl14/        # local UniDepth folder
      DepthLM/                   # local DepthLM folder
```
