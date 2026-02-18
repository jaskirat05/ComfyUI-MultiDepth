import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


_MODEL_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()

_METRIC_MEAN = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(3, 1, 1)
_METRIC_STD = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(3, 1, 1)


class MultiDepthEstimateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "backend": (
                    ["metric3dv2", "unidepth_v2", "facebook_depth_lm"],
                    {"default": "metric3dv2"},
                ),
                "normalize_output": ("BOOLEAN", {"default": True}),
                "invert_output": ("BOOLEAN", {"default": False}),
                "min_depth_m": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "max_depth_m": ("FLOAT", {"default": 80.0, "min": 0.01, "max": 10000.0, "step": 0.1}),
            },
            "optional": {
                "focal_length_px": ("FLOAT", {"default": 1000.0, "min": 1.0, "max": 10000.0, "step": 1.0}),
                "metric3d_variant": (
                    ["metric3d_vit_small", "metric3d_vit_large", "metric3d_vit_giant2"],
                    {"default": "metric3d_vit_small"},
                ),
                "unidepth_model_id": ("STRING", {"default": "lpiccinelli/unidepth-v2-vitl14"}),
                "facebook_model_id": ("STRING", {"default": "facebook/map-anything"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"
    CATEGORY = "depth"

    def estimate(
        self,
        image,
        backend,
        normalize_output,
        invert_output,
        min_depth_m,
        max_depth_m,
        focal_length_px=1000.0,
        metric3d_variant="metric3d_vit_small",
        unidepth_model_id="lpiccinelli/unidepth-v2-vitl14",
        facebook_model_id="facebook/map-anything",
    ):
        if max_depth_m <= min_depth_m:
            raise ValueError("max_depth_m must be greater than min_depth_m")

        device = _select_device()
        images = image.detach().cpu()
        out = []

        for i in range(images.shape[0]):
            rgb_uint8 = _comfy_frame_to_uint8(images[i])

            if backend == "metric3dv2":
                depth = _infer_metric3d(
                    rgb_uint8=rgb_uint8,
                    device=device,
                    variant=metric3d_variant,
                    focal_length_px=focal_length_px,
                    min_depth_m=min_depth_m,
                    max_depth_m=max_depth_m,
                )
            elif backend == "unidepth_v2":
                depth = _infer_unidepth(
                    rgb_uint8=rgb_uint8,
                    device=device,
                    model_id=unidepth_model_id,
                    min_depth_m=min_depth_m,
                    max_depth_m=max_depth_m,
                )
            else:
                depth = _infer_facebook_depth_lm(
                    rgb_uint8=rgb_uint8,
                    device=device,
                    model_id=facebook_model_id,
                    min_depth_m=min_depth_m,
                    max_depth_m=max_depth_m,
                )

            depth_vis = _depth_to_vis(depth, normalize_output=normalize_output, invert_output=invert_output)
            out.append(depth_vis)

        return (torch.stack(out, dim=0),)


def _select_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _comfy_frame_to_uint8(frame: torch.Tensor) -> np.ndarray:
    # ComfyUI IMAGE is HWC float32 in [0, 1].
    frame = frame.clamp(0.0, 1.0)
    arr = (frame.numpy() * 255.0).astype(np.uint8)
    return arr


def _depth_to_vis(depth: torch.Tensor, normalize_output: bool, invert_output: bool) -> torch.Tensor:
    depth = depth.float().cpu()

    if normalize_output:
        dmin = float(depth.min())
        dmax = float(depth.max())
        if dmax - dmin < 1e-8:
            norm = torch.zeros_like(depth)
        else:
            norm = (depth - dmin) / (dmax - dmin)
    else:
        norm = depth.clamp(0.0, 1.0)

    if invert_output:
        norm = 1.0 - norm

    return norm.unsqueeze(-1).repeat(1, 1, 3)


def _cache_key(*parts: str) -> str:
    return "::".join(parts)


def _get_or_load_model(key: str, loader):
    with _CACHE_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

    model = loader()

    with _CACHE_LOCK:
        _MODEL_CACHE[key] = model

    return model


def _infer_metric3d(
    rgb_uint8: np.ndarray,
    device: torch.device,
    variant: str,
    focal_length_px: float,
    min_depth_m: float,
    max_depth_m: float,
) -> torch.Tensor:
    key = _cache_key("metric3d", variant, str(device))

    def _loader():
        model = torch.hub.load("YvanYin/Metric3D", variant, pretrain=True, trust_repo=True)
        model.to(device).eval()
        return model

    model = _get_or_load_model(key, _loader)

    h, w, _ = rgb_uint8.shape
    input_size = (616, 1064)

    img = torch.from_numpy(rgb_uint8).permute(2, 0, 1).float()

    scale = min(input_size[0] / h, input_size[1] / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized = F.interpolate(
        img.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    canvas = _METRIC_MEAN.repeat(1, input_size[0], input_size[1]).clone()
    pad_h = input_size[0] - new_h
    pad_w = input_size[1] - new_w
    top = pad_h // 2
    left = pad_w // 2
    canvas[:, top : top + new_h, left : left + new_w] = resized
    norm = (canvas - _METRIC_MEAN) / _METRIC_STD

    with torch.no_grad():
        pred_depth, _, _ = model.inference({"input": norm.unsqueeze(0).to(device)})

    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[top : top + new_h, left : left + new_w]
    pred_depth = F.interpolate(
        pred_depth.unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    scaled_focal = focal_length_px * scale
    pred_depth = pred_depth * (scaled_focal / 1000.0)
    return pred_depth.clamp(min_depth_m, max_depth_m).cpu()


def _infer_unidepth(
    rgb_uint8: np.ndarray,
    device: torch.device,
    model_id: str,
    min_depth_m: float,
    max_depth_m: float,
) -> torch.Tensor:
    key = _cache_key("unidepth_v2", model_id, str(device))

    def _loader():
        try:
            from unidepth.models import UniDepthV2
        except Exception as exc:
            raise RuntimeError(
                "UniDepth import failed. Install with: pip install unidepth"
            ) from exc

        model = UniDepthV2.from_pretrained(model_id)
        model.to(device).eval()
        return model

    model = _get_or_load_model(key, _loader)

    rgb = torch.from_numpy(rgb_uint8).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    rgb = rgb.to(device)

    with torch.no_grad():
        try:
            outputs = model.infer(rgb)
        except Exception:
            outputs = model(rgb)

    depth = _extract_depth_tensor(outputs)
    if depth is None:
        raise RuntimeError("UniDepth output did not contain a recognizable depth tensor.")

    depth = _ensure_hw(depth, rgb_uint8.shape[0], rgb_uint8.shape[1])
    return depth.clamp(min_depth_m, max_depth_m).cpu()


def _infer_facebook_depth_lm(
    rgb_uint8: np.ndarray,
    device: torch.device,
    model_id: str,
    min_depth_m: float,
    max_depth_m: float,
) -> torch.Tensor:
    key = _cache_key("facebook_depth_lm", model_id, str(device))

    def _loader():
        try:
            from transformers import AutoModel, AutoModelForDepthEstimation, AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "Transformers import failed. Install with: pip install transformers"
            ) from exc

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        try:
            model = AutoModelForDepthEstimation.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

        model.to(device).eval()
        return processor, model

    processor, model = _get_or_load_model(key, _loader)

    pil_img = Image.fromarray(rgb_uint8)
    inputs = processor(images=pil_img, return_tensors="pt")
    model_inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        try:
            outputs = model(**model_inputs)
        except TypeError:
            # Some trust_remote_code models expose custom infer signatures.
            if hasattr(model, "infer"):
                outputs = model.infer(**model_inputs)
            else:
                raise

    depth = _extract_depth_tensor(outputs)
    if depth is None and hasattr(model, "infer"):
        with torch.no_grad():
            depth = _extract_depth_tensor(model.infer(**model_inputs))

    if depth is None:
        raise RuntimeError(
            "Depth tensor not found for Facebook depth LM output. "
            "Try a different model_id or ensure the model supports image depth inference."
        )

    depth = _ensure_hw(depth, rgb_uint8.shape[0], rgb_uint8.shape[1])
    return depth.clamp(min_depth_m, max_depth_m).cpu()


def _extract_depth_tensor(outputs: Any) -> Optional[torch.Tensor]:
    keys = [
        "predicted_depth",
        "depth",
        "depth_map",
        "metric_depth",
        "disparity",
        "pred_depth",
    ]

    if outputs is None:
        return None

    if isinstance(outputs, torch.Tensor):
        return outputs

    for key in keys:
        if hasattr(outputs, key):
            value = getattr(outputs, key)
            if isinstance(value, torch.Tensor):
                return value

    if isinstance(outputs, dict):
        for key in keys:
            if key in outputs and isinstance(outputs[key], torch.Tensor):
                return outputs[key]
        for value in outputs.values():
            found = _extract_depth_tensor(value)
            if found is not None:
                return found

    if hasattr(outputs, "values"):
        try:
            for value in outputs.values():
                found = _extract_depth_tensor(value)
                if found is not None:
                    return found
        except Exception:
            pass

    if isinstance(outputs, (list, tuple)):
        for value in outputs:
            found = _extract_depth_tensor(value)
            if found is not None:
                return found

    return None


def _ensure_hw(depth: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    depth = depth.float()

    if depth.ndim == 4:
        depth = depth[0, 0] if depth.shape[1] == 1 else depth[0].mean(dim=0)
    elif depth.ndim == 3:
        depth = depth[0] if depth.shape[0] == 1 else depth.mean(dim=0)

    if depth.ndim != 2:
        raise RuntimeError(f"Expected a 2D depth map, got shape {tuple(depth.shape)}")

    h, w = depth.shape
    if h != target_h or w != target_w:
        depth = F.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    return depth


NODE_CLASS_MAPPINGS = {
    "MultiDepthEstimateNode": MultiDepthEstimateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiDepthEstimateNode": "Depth Estimate (Metric3Dv2 / UniDepthV2 / Facebook LM)",
}
