import threading
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


_MODEL_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()

_METRIC_MEAN = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(3, 1, 1)
_METRIC_STD = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(3, 1, 1)


class Metric3Dv2DepthNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
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
                "depth_models_root": ("STRING", {"default": ""}),
                "metric3d_repo_dir": ("STRING", {"default": "Metric3D"}),
                "metric3d_checkpoint": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"
    CATEGORY = "depth"

    def estimate(
        self,
        image,
        normalize_output,
        invert_output,
        min_depth_m,
        max_depth_m,
        focal_length_px=1000.0,
        metric3d_variant="metric3d_vit_small",
        depth_models_root="",
        metric3d_repo_dir="Metric3D",
        metric3d_checkpoint="",
    ):
        if max_depth_m <= min_depth_m:
            raise ValueError("max_depth_m must be greater than min_depth_m")

        device = _select_device()
        return _estimate_batch(
            image=image,
            normalize_output=normalize_output,
            invert_output=invert_output,
            infer_fn=lambda rgb_uint8: _infer_metric3d(
                rgb_uint8=rgb_uint8,
                device=device,
                variant=metric3d_variant,
                focal_length_px=focal_length_px,
                depth_models_root=depth_models_root,
                metric3d_repo_dir=metric3d_repo_dir,
                metric3d_checkpoint=metric3d_checkpoint,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            ),
        )


class UniDepthV2DepthNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "normalize_output": ("BOOLEAN", {"default": True}),
                "invert_output": ("BOOLEAN", {"default": False}),
                "min_depth_m": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "max_depth_m": ("FLOAT", {"default": 80.0, "min": 0.01, "max": 10000.0, "step": 0.1}),
            },
            "optional": {
                "depth_models_root": ("STRING", {"default": ""}),
                "unidepth_model_id": ("STRING", {"default": "unidepth-v2-vitl14"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"
    CATEGORY = "depth"

    def estimate(
        self,
        image,
        normalize_output,
        invert_output,
        min_depth_m,
        max_depth_m,
        depth_models_root="",
        unidepth_model_id="unidepth-v2-vitl14",
    ):
        if max_depth_m <= min_depth_m:
            raise ValueError("max_depth_m must be greater than min_depth_m")
        device = _select_device()
        return _estimate_batch(
            image=image,
            normalize_output=normalize_output,
            invert_output=invert_output,
            infer_fn=lambda rgb_uint8: _infer_unidepth(
                rgb_uint8=rgb_uint8,
                device=device,
                model_id=unidepth_model_id,
                depth_models_root=depth_models_root,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            ),
        )


class DepthLMDepthNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "normalize_output": ("BOOLEAN", {"default": True}),
                "invert_output": ("BOOLEAN", {"default": False}),
                "min_depth_m": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "max_depth_m": ("FLOAT", {"default": 80.0, "min": 0.01, "max": 10000.0, "step": 0.1}),
                "depthlm_point_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "depthlm_point_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "depth_models_root": ("STRING", {"default": ""}),
                "facebook_model_id": ("STRING", {"default": "DepthLM"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"
    CATEGORY = "depth"

    def estimate(
        self,
        image,
        normalize_output,
        invert_output,
        min_depth_m,
        max_depth_m,
        depthlm_point_x,
        depthlm_point_y,
        depth_models_root="",
        facebook_model_id="DepthLM",
    ):
        if max_depth_m <= min_depth_m:
            raise ValueError("max_depth_m must be greater than min_depth_m")
        device = _select_device()
        return _estimate_batch(
            image=image,
            normalize_output=normalize_output,
            invert_output=invert_output,
            infer_fn=lambda rgb_uint8: _infer_facebook_depth_lm(
                rgb_uint8=rgb_uint8,
                device=device,
                model_id=facebook_model_id,
                depth_models_root=depth_models_root,
                point_x=depthlm_point_x,
                point_y=depthlm_point_y,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            ),
        )


def _estimate_batch(image, normalize_output: bool, invert_output: bool, infer_fn):
    images = image.detach().cpu()
    out = []
    for i in range(images.shape[0]):
        rgb_uint8 = _comfy_frame_to_uint8(images[i])
        depth = infer_fn(rgb_uint8)
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
    depth_models_root: str,
    metric3d_repo_dir: str,
    metric3d_checkpoint: str,
    min_depth_m: float,
    max_depth_m: float,
) -> torch.Tensor:
    key = _cache_key("metric3d", variant, str(device), depth_models_root, metric3d_repo_dir, metric3d_checkpoint)

    def _loader():
        repo_dir = _resolve_local_model_path(metric3d_repo_dir, depth_models_root, expect_dir=True)
        try:
            model = torch.hub.load(
                str(repo_dir),
                variant,
                pretrain=False,
                source="local",
                trust_repo=True,
            )
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", "")
            if missing in {"mmengine", "mmcv", "mmcv._ext"}:
                raise RuntimeError(
                    "Metric3Dv2 dependencies missing. Install in ComfyUI env with: "
                    "pip install mmengine mmcv-lite"
                ) from exc
            raise

        ckpt_path = _resolve_metric3d_checkpoint(
            variant=variant,
            depth_models_root=depth_models_root,
            metric3d_repo_dir=metric3d_repo_dir,
            metric3d_checkpoint=metric3d_checkpoint,
        )
        state = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = state
        if isinstance(state, dict):
            for key_name in ("state_dict", "model", "model_state_dict"):
                if key_name in state and isinstance(state[key_name], dict):
                    state_dict = state[key_name]
                    break
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Unsupported Metric3D checkpoint format: {ckpt_path}")
        model.load_state_dict(state_dict, strict=False)
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
    depth_models_root: str,
    min_depth_m: float,
    max_depth_m: float,
) -> torch.Tensor:
    key = _cache_key("unidepth_v2", model_id, str(device), depth_models_root)

    def _loader():
        try:
            from unidepth.models import UniDepthV2
        except Exception as exc:
            raise RuntimeError(
                "UniDepth import failed. Install with: pip install unidepth"
            ) from exc

        local_model_path = _resolve_local_model_path(model_id, depth_models_root, expect_dir=True)
        model = UniDepthV2.from_pretrained(str(local_model_path), local_files_only=True)
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
    depth_models_root: str,
    point_x: float,
    point_y: float,
    min_depth_m: float,
    max_depth_m: float,
) -> torch.Tensor:
    key = _cache_key("facebook_depth_lm", model_id, str(device), depth_models_root)

    def _loader():
        try:
            from transformers import AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "Transformers import failed. Install with: pip install transformers"
            ) from exc

        local_model_path = _resolve_local_model_path(model_id, depth_models_root, expect_dir=True)
        processor = AutoProcessor.from_pretrained(
            str(local_model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        model = _load_depthlm_text_model(str(local_model_path), device)

        model.to(device).eval()
        return processor, model

    processor, model = _get_or_load_model(key, _loader)

    pil_img = Image.fromarray(rgb_uint8)
    marked_image = _draw_depthlm_arrow_marker(pil_img, point_x, point_y)
    prompt = (
        "Given this image, how far is the point pointed by the red arrow from the camera? "
        "Output the meter number only."
    )

    model_path_lower = model_id.lower()
    text_output = _generate_depthlm_text(
        model=model,
        processor=processor,
        image=marked_image,
        prompt=prompt,
        is_pixtral=("pixtral" in model_path_lower or "depthlm" in model_path_lower),
        device=device,
    )

    point_depth = _extract_float(text_output)
    if point_depth is None:
        raise RuntimeError(
            "DepthLM response did not contain a parseable numeric depth value. "
            f"Response: {text_output!r}"
        )

    point_depth = float(np.clip(point_depth, min_depth_m, max_depth_m))
    h, w, _ = rgb_uint8.shape
    depth = torch.full((h, w), point_depth, dtype=torch.float32)
    return depth


def _load_depthlm_text_model(model_id: str, device: torch.device):
    try:
        from transformers import (
            AutoModelForImageTextToText,
            LlavaForConditionalGeneration,
            Qwen2_5_VLForConditionalGeneration,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not import DepthLM model classes from transformers."
        ) from exc

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_path_lower = model_id.lower()

    if "pixtral" in model_path_lower or "depthlm" in model_path_lower:
        try:
            return LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            pass

    if "qwen" in model_path_lower:
        try:
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            pass

    return AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )


def _default_depth_models_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "models" / "depth_models",
        here.parent.parent / "models" / "depth_models",
        here.parent.parent.parent / "models" / "depth_models",
        Path.cwd() / "models" / "depth_models",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[2]


def _resolve_local_model_path(name_or_path: str, depth_models_root: str, expect_dir: bool) -> Path:
    raw = (name_or_path or "").strip()
    if not raw:
        raise RuntimeError("Empty local model path provided.")

    p = Path(raw).expanduser()
    if not p.is_absolute():
        base = Path(depth_models_root).expanduser() if (depth_models_root or "").strip() else _default_depth_models_root()
        p = (base / raw).resolve()

    if not p.exists():
        raise RuntimeError(
            f"Local model path not found: {p}. "
            "Place models under models/depth_models or set depth_models_root."
        )
    if expect_dir and not p.is_dir():
        raise RuntimeError(f"Expected directory for model path, got file: {p}")
    return p


def _resolve_metric3d_checkpoint(
    variant: str,
    depth_models_root: str,
    metric3d_repo_dir: str,
    metric3d_checkpoint: str,
) -> Path:
    if (metric3d_checkpoint or "").strip():
        p = Path(metric3d_checkpoint).expanduser()
        if not p.is_absolute():
            base = Path(depth_models_root).expanduser() if (depth_models_root or "").strip() else _default_depth_models_root()
            p = (base / metric3d_checkpoint).resolve()
        if not p.exists():
            raise RuntimeError(f"Metric3D checkpoint not found: {p}")
        return p

    base = Path(depth_models_root).expanduser() if (depth_models_root or "").strip() else _default_depth_models_root()
    repo_dir = _resolve_local_model_path(metric3d_repo_dir, depth_models_root, expect_dir=True)
    search_roots = [base, repo_dir, repo_dir / "checkpoints", repo_dir / "weights"]
    patterns = [
        f"{variant}*.pth",
        f"{variant}*.pt",
        f"*{variant}*.pth",
        f"*{variant}*.pt",
        "*.pth",
        "*.pt",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for pat in patterns:
            found = sorted(root.glob(pat))
            if found:
                return found[0]

    raise RuntimeError(
        "Metric3D checkpoint not found locally. Set metric3d_checkpoint explicitly, "
        "or place weights under models/depth_models."
    )


def _draw_depthlm_arrow_marker(image: Image.Image, point_x: float, point_y: float) -> Image.Image:
    w, h = image.size
    px = int(np.clip(point_x, 0.0, 1.0) * (w - 1))
    py = int(np.clip(point_y, 0.0, 1.0) * (h - 1))

    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    arrow_len = max(16, int(min(w, h) * 0.12))
    start_x = max(0, px - arrow_len)
    start_y = max(0, py - arrow_len)
    draw.line((start_x, start_y, px, py), fill=(255, 0, 0), width=max(2, arrow_len // 16))

    head = max(6, arrow_len // 4)
    draw.polygon(
        [
            (px, py),
            (max(0, px - head), max(0, py - head // 2)),
            (max(0, px - head // 2), max(0, py - head)),
        ],
        fill=(255, 0, 0),
    )
    draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(255, 0, 0))
    return out


def _generate_depthlm_text(model, processor, image: Image.Image, prompt: str, is_pixtral: bool, device: torch.device) -> str:
    if is_pixtral:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "content": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")

    model_inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=64,
            do_sample=False,
            top_p=None,
            top_k=None,
        )

    input_ids = model_inputs.get("input_ids", None)
    if input_ids is not None and isinstance(input_ids, torch.Tensor):
        trimmed = generated_ids[:, input_ids.shape[-1] :]
    else:
        trimmed = generated_ids

    decoded = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0] if decoded else ""


def _extract_float(text: str) -> Optional[float]:
    if not text:
        return None

    match = re.search(r"[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


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
    "Metric3Dv2DepthNode": Metric3Dv2DepthNode,
    "UniDepthV2DepthNode": UniDepthV2DepthNode,
    "DepthLMDepthNode": DepthLMDepthNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Metric3Dv2DepthNode": "Depth - Metric3Dv2 (Local)",
    "UniDepthV2DepthNode": "Depth - UniDepthV2 (Local)",
    "DepthLMDepthNode": "Depth - DepthLM (Local)",
}
