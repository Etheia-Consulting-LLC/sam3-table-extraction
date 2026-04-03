from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision.transforms import v2

from sam3.model.model_misc import SAM3Output
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
)

from sam3_table.lora_layers import LoRAConfig as LoRALayerConfig
from sam3_table.lora_layers import apply_lora_to_model
from sam3_table.train_sam3_lora_native import (
    convert_predictions_to_coco_format_original_res,
    resolve_bpe_vocab_path,
)
from sam3_table.training_config import SAM3LoRAConfig

DEFAULT_QUERY_TEXT = "table"
MODEL_RESOLUTION = 1008


@dataclass(frozen=True)
class TableDetection:
    bbox_xywh: tuple[float, float, float, float]
    bbox_xyxy: tuple[float, float, float, float]
    score: float


class _SingleImageInferenceDataset(Dataset):
    def __init__(self, image: PILImage.Image, query_text: str):
        self.image = image.convert("RGB")
        self.query_text = query_text.strip().lower() or DEFAULT_QUERY_TEXT
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Datapoint:
        if idx != 0:
            raise IndexError(idx)

        orig_w, orig_h = self.image.size
        resized = self.image.resize(
            (MODEL_RESOLUTION, MODEL_RESOLUTION),
            PILImage.BILINEAR,
        )
        image_tensor = self.transform(resized)
        image_obj = Image(
            data=image_tensor,
            objects=[],
            size=(MODEL_RESOLUTION, MODEL_RESOLUTION),
        )
        query = FindQueryLoaded(
            query_text=self.query_text,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=0,
                original_image_id=0,
                original_category_id=0,
                original_size=(orig_h, orig_w),
                object_id=-1,
                frame_index=-1,
            ),
        )
        return Datapoint(find_queries=[query], images=[image_obj], raw_images=[resized])


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, list):
        return [_move_to_device(item, device) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(item, device) for item in obj)
    if isinstance(obj, dict):
        return {key: _move_to_device(value, device) for key, value in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            setattr(obj, field, _move_to_device(getattr(obj, field), device))
        return obj
    return obj


def _resolve_config_path(
    weights_path: Path,
    config_path: str | Path | None,
) -> Path:
    if config_path is not None:
        resolved_config_path = Path(config_path)
        if not resolved_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {resolved_config_path}")
        return resolved_config_path

    sibling_config_path = weights_path.with_name("run_config.json")
    if sibling_config_path.exists():
        return sibling_config_path

    raise FileNotFoundError(
        "Could not find a config for the LoRA weights. "
        "Expected a sibling run_config.json next to the weights file, "
        "or pass config_path explicitly."
    )


def _load_config(
    weights_path: Path,
    config_path: str | Path | None,
) -> SAM3LoRAConfig:
    resolved_config_path = _resolve_config_path(weights_path, config_path)
    if resolved_config_path.suffix.lower() == ".json":
        return SAM3LoRAConfig.model_validate(
            json.loads(resolved_config_path.read_text())
        )
    return SAM3LoRAConfig.from_yaml(resolved_config_path)


def _build_lora_model(
    config: SAM3LoRAConfig,
    device: torch.device,
) -> torch.nn.Module:
    from sam3_table.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        device=device.type,
        compile=False,
        load_from_HF=True,
        bpe_path=resolve_bpe_vocab_path(),
        eval_mode=True,
    )
    lora_cfg = config.lora
    return apply_lora_to_model(
        model,
        LoRALayerConfig(
            rank=lora_cfg.rank,
            alpha=lora_cfg.alpha,
            dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
            apply_to_vision_encoder=lora_cfg.apply_to_vision_encoder,
            apply_to_text_encoder=lora_cfg.apply_to_text_encoder,
            apply_to_geometry_encoder=lora_cfg.apply_to_geometry_encoder,
            apply_to_detr_encoder=lora_cfg.apply_to_detr_encoder,
            apply_to_detr_decoder=lora_cfg.apply_to_detr_decoder,
            apply_to_mask_decoder=lora_cfg.apply_to_mask_decoder,
        ),
    )


def infer_table_detections(
    weights_path: str | Path,
    image: PILImage.Image,
    *,
    config_path: str | Path | None = None,
    score_threshold: float = 0.25,
    query_text: str = DEFAULT_QUERY_TEXT,
    device: str | torch.device | None = None,
) -> list[TableDetection]:
    """Run SAM3 LoRA inference on a PIL image and return detected table boxes."""

    resolved_weights_path = Path(weights_path)
    if not resolved_weights_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {resolved_weights_path}")

    device_obj = _resolve_device(device)
    config = _load_config(resolved_weights_path, config_path)
    dataset = _SingleImageInferenceDataset(image=image, query_text=query_text)

    batch = collate_fn_api(
        [dataset[0]],
        dict_key="input",
        with_seg_masks=True,
    )

    model = _build_lora_model(config, device_obj)
    lora_state_dict = torch.load(resolved_weights_path, map_location=device_obj)
    model.load_state_dict(lora_state_dict, strict=False)
    model.to(device_obj)
    model.eval()

    input_batch = _move_to_device(batch["input"], device_obj)
    with torch.inference_mode():
        outputs = model(input_batch)

    with SAM3Output.iteration_mode(outputs, SAM3Output.IterMode.LAST_STEP_PER_STAGE):
        final_stage = list(outputs)[-1]

    pred_logits = final_stage["pred_logits"]
    pred_boxes = final_stage["pred_boxes"]
    pred_masks = final_stage["pred_masks"]
    if pred_logits.dim() == 2:
        predictions_list = [
            {
                "pred_logits": pred_logits.detach(),
                "pred_boxes": pred_boxes.detach(),
                "pred_masks": pred_masks.detach(),
            }
        ]
    else:
        predictions_list = [
            {
                "pred_logits": pred_logits[idx].detach(),
                "pred_boxes": pred_boxes[idx].detach(),
                "pred_masks": pred_masks[idx].detach(),
            }
            for idx in range(pred_logits.shape[0])
        ]

    coco_predictions = convert_predictions_to_coco_format_original_res(
        predictions_list=predictions_list,
        image_ids=[0],
        dataset=dataset,
        score_threshold=score_threshold,
    )

    detections: list[TableDetection] = []
    for prediction in sorted(
        coco_predictions,
        key=lambda item: float(item["score"]),
        reverse=True,
    ):
        x, y, w, h = prediction["bbox"]
        x1 = float(x)
        y1 = float(y)
        x2 = float(x + w)
        y2 = float(y + h)
        detections.append(
            TableDetection(
                bbox_xywh=(x1, y1, float(w), float(h)),
                bbox_xyxy=(x1, y1, x2, y2),
                score=float(prediction["score"]),
            )
        )

    return detections


def infer_table_bboxes(
    weights_path: str | Path,
    image: PILImage.Image,
    *,
    config_path: str | Path | None = None,
    score_threshold: float = 0.25,
    query_text: str = DEFAULT_QUERY_TEXT,
    device: str | torch.device | None = None,
) -> list[tuple[float, float, float, float]]:
    """Convenience wrapper that returns only `xyxy` boxes."""

    detections = infer_table_detections(
        weights_path=weights_path,
        image=image,
        config_path=config_path,
        score_threshold=score_threshold,
        query_text=query_text,
        device=device,
    )
    return [detection.bbox_xyxy for detection in detections]
