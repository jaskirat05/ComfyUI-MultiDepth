# ComfyUI Multi-Depth Node

A ComfyUI custom node that supports three depth backends from one node:

- Metric3Dv2 (`YvanYin/Metric3D` via `torch.hub`)
- UniDepth v2 (`lpiccinelli/unidepth-v2-*`)
- Facebook depth LM (configurable Hugging Face model id, default `facebook/map-anything`)

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

## Install

1. Put this folder under your ComfyUI custom nodes directory.
2. Install dependencies in your ComfyUI Python environment:

```bash
pip install -r requirements.txt
```

3. Restart ComfyUI.

## Notes

- Models are lazy-loaded and cached by backend/model/device.
- The Facebook backend uses `trust_remote_code=True` because many recent depth models expose custom forward APIs.
- If the selected Facebook model does not expose a standard depth tensor key, switch `facebook_model_id` to another supported checkpoint.
