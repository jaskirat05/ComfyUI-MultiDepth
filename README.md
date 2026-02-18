# ComfyUI Depth Nodes

A ComfyUI custom node pack with three separate nodes:

- Metric3D (model-zoo via `torch.hub`)
- UniDepth v1/v2 (model-zoo via `from_pretrained`)
- Facebook DepthLM Official style (local pretrained folder)

## Nodes

- `Metric3Dv2DepthNode`
- `UniDepthV2DepthNode`
- `DepthLMDepthNode`
- Category: `depth` for all
- Input image type: `IMAGE`
- Output type:
- `Metric3Dv2DepthNode`: `IMAGE depth_image`, `IMAGE normal_image`
- `UniDepthV2DepthNode`: `IMAGE depth_image`
- `DepthLMDepthNode`: `IMAGE depth_image`

## Inputs

Required:

- `image`
- `normalize_output` (bool)
- `invert_output` (bool)
- `min_depth_m` (float)
- `max_depth_m` (float)

Optional, per node:

- `Metric3Dv2DepthNode`: `focal_length_px`, `metric3d_model_name`
- `UniDepthV2DepthNode`: `unidepth_model_name`
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
- Metric3D loads with `torch.hub.load("yvanyin/metric3d", model_name, pretrain=True)`.
- Metric3D model selector values:
- `metric3d_convnext_tiny`
- `metric3d_convnext_large`
- `metric3d_vit_small`
- `metric3d_vit_large`
- `metric3d_vit_giant2`
- Metric3D normal output is available for v2 ViT models; for non-normal models the node returns a blank normal image.
- UniDepth loads via `from_pretrained("lpiccinelli/<name>")` and may download on first run if not cached.
- The Facebook backend is implemented in official DepthLM query style: it draws a red arrow at the selected point and asks the model for numeric metric depth.
- DepthLM Official is point-depth oriented in this integration (single queried point), so output visualization is a constant depth image based on the predicted metric value.
- The backend uses `trust_remote_code=True` and transformers vision-language generation APIs.

## UniDepth model zoo selector

`UniDepthV2DepthNode` loads via:

```python
from unidepth.models import UniDepthV1, UniDepthV2
model_v1 = UniDepthV1.from_pretrained(f"lpiccinelli/{name}")
model_v2 = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
```

Available selector values:

- `unidepth-v1-cnvnxtl`
- `unidepth-v1-vitl14`
- `unidepth-v2-vits14`
- `unidepth-v2-vitl14`

## Local model layout (example)

```text
ComfyUI/
  models/
    depth_models/
      unidepth-v2-vitl14/        # local UniDepth folder
      DepthLM/                   # local DepthLM folder
```
