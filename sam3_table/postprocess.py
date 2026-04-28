from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.ops import nms


@dataclass
class ProcessedPrediction:
    bbox_xywh: tuple[float, float, float, float]
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    binary_mask: torch.Tensor


def _bbox_from_binary_mask(binary_mask: torch.Tensor) -> tuple[float, float, float, float] | None:
    nonzero = torch.nonzero(binary_mask, as_tuple=False)
    if nonzero.numel() == 0:
        return None

    y_min = float(nonzero[:, 0].min().item())
    y_max = float(nonzero[:, 0].max().item())
    x_min = float(nonzero[:, 1].min().item())
    x_max = float(nonzero[:, 1].max().item())
    return (x_min, y_min, x_max - x_min + 1.0, y_max - y_min + 1.0)


def postprocess_sam3_predictions(
    *,
    pred_logits: torch.Tensor,
    pred_masks: torch.Tensor,
    original_size: tuple[int, int],
    score_threshold: float,
    duplicate_iou_threshold: float = 0.5,
    min_box_area: float = 16.0,
    mask_threshold: float = 0.5,
) -> list[ProcessedPrediction]:
    """Convert raw SAM3 outputs into deduplicated detections in original image coordinates."""

    orig_h, orig_w = original_size
    scores = torch.sigmoid(pred_logits).squeeze(-1)
    valid_mask = scores > score_threshold
    scores = scores[valid_mask]
    masks = pred_masks[valid_mask]

    if len(scores) == 0:
        return []

    masks_sigmoid = torch.sigmoid(masks)
    masks_upsampled = torch.nn.functional.interpolate(
        masks_sigmoid.unsqueeze(1).float(),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    binary_masks = (masks_upsampled > mask_threshold).cpu()
    scores_cpu = scores.detach().cpu()

    del masks_sigmoid, masks_upsampled
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    candidates: list[ProcessedPrediction] = []
    candidate_boxes_xyxy: list[list[float]] = []
    candidate_scores: list[float] = []

    for binary_mask, score in zip(binary_masks, scores_cpu.tolist()):
        bbox_xywh = _bbox_from_binary_mask(binary_mask)
        if bbox_xywh is None:
            continue

        x, y, w, h = bbox_xywh
        if w < 1.0 or h < 1.0 or (w * h) < min_box_area:
            continue

        bbox_xyxy = (x, y, x + w, y + h)
        candidates.append(
            ProcessedPrediction(
                bbox_xywh=bbox_xywh,
                bbox_xyxy=bbox_xyxy,
                score=float(score),
                binary_mask=binary_mask,
            )
        )
        candidate_boxes_xyxy.append([float(x), float(y), float(x + w), float(y + h)])
        candidate_scores.append(float(score))

    if not candidates:
        return []

    keep_indices = nms(
        torch.tensor(candidate_boxes_xyxy, dtype=torch.float32),
        torch.tensor(candidate_scores, dtype=torch.float32),
        duplicate_iou_threshold,
    ).tolist()
    return [candidates[idx] for idx in keep_indices]
