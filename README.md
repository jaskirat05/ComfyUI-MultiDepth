# ComfyUI Multi-Depth Node

A ComfyUI custom node that supports three depth backends from one node:

- Metric3Dv2 (`YvanYin/Metric3D` via `torch.hub`)
- UniDepth v2 (`lpiccinelli/unidepth-v2-*`)
- Facebook DepthLM Official style (configurable Hugging Face model id, default `facebook/DepthLM`)

## Node

- `MultiDepthEstimateNode`
- Category: `depth`
- Input image type: `IMAGE`
- Output type: `IMAGE` (grayscale depth visualization repeated to 3 channels)

## Inputs

Required:

- `image`
- `backend`: `metric3dv2`, `unidepth_v2`, `facebook_depth_lm`
- `normalize_output` (bool)
- `invert_output` (bool)
- `min_depth_m` (float)
- `max_depth_m` (float)

Optional:

- `focal_length_px` (used by Metric3Dv2 scale conversion)
- `metric3d_variant`: `metric3d_vit_small`, `metric3d_vit_large`, `metric3d_vit_giant2`
- `unidepth_model_id`
- `facebook_model_id`
- `depthlm_point_x`, `depthlm_point_y` (normalized [0,1] point queried by DepthLM)

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
- The Facebook backend is implemented in official DepthLM query style: it draws a red arrow at the selected point and asks the model for numeric metric depth.
- DepthLM Official is point-depth oriented in this integration (single queried point), so output visualization is a constant depth image based on the predicted metric value.
- The backend uses `trust_remote_code=True` and transformers vision-language generation APIs.
