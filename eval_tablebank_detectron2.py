#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

from sam3_table.coco_schema import COCODataset
from voc_to_coco import convert_voc_to_coco 

TABLEBANK_R101_CONFIG_URL = (
    "https://huggingface.co/layoutparser/detectron2/resolve/main/"
    "TableBank/faster_rcnn_R_101_FPN_3x/config.yml"
)
TABLEBANK_R101_WEIGHTS_URL = (
    "https://huggingface.co/layoutparser/detectron2/resolve/main/"
    "TableBank/faster_rcnn_R_101_FPN_3x/model_final.pth"
)

DETECTRON2_TABLEBANK_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "gcc",
        "g++",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
    )
    .pip_install_from_requirements("requirements.txt")
    .pip_install_from_requirements("requirements-detectron2.txt")
    .add_local_python_source("sam3_table")
    .add_local_file("voc_to_coco.py", remote_path="/root/voc_to_coco.py")
)

app = modal.App(name="tablebank-detectron2-eval", image=DETECTRON2_TABLEBANK_IMAGE)

tablebank_vol = modal.Volume.from_name("tablebank-vol")
artifacts_vol = modal.Volume.from_name("artifacts-vol", create_if_missing=True)

MODAL_DATA_DIR = "/data"
MODAL_ARTIFACTS_DIR = "/artifacts"


@dataclass(frozen=True)
class DetectionImage:
    image_id: int
    file_name: str
    width: int
    height: int
    image_path: Path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _score_annotation_candidate(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    score = 0
    if "annotation" in name:
        score += 4
    if "coco" in name:
        score += 3
    if "detect" in name or "table" in name:
        score += 1
    return (-score, len(path.parts), str(path))


def _resolve_annotations_source(
    annotations_path: Path,
    dataset_root_path: Path,
) -> tuple[Path, str]:
    if annotations_path.is_file():
        return annotations_path, "coco"

    search_roots: list[Path] = []
    if annotations_path.exists():
        search_roots.append(annotations_path)
    if dataset_root_path not in search_roots:
        search_roots.append(dataset_root_path)

    json_candidates: list[Path] = []
    xml_candidates: list[Path] = []
    seen_paths: set[Path] = set()
    for root in search_roots:
        if not root.exists() or root in seen_paths:
            continue
        seen_paths.add(root)
        if root.is_file():
            if root.suffix.lower() == ".json":
                json_candidates.append(root)
            elif root.suffix.lower() == ".xml":
                xml_candidates.append(root)
            continue
        json_candidates.extend(path for path in root.rglob("*.json") if path.is_file())
        xml_candidates.extend(path for path in root.rglob("*.xml") if path.is_file())

    if json_candidates:
        return sorted(set(json_candidates), key=_score_annotation_candidate)[0], "coco"

    if xml_candidates:
        xml_root = annotations_path if annotations_path.exists() else dataset_root_path
        return xml_root, "voc"

    raise FileNotFoundError(
        f"Could not find COCO JSON or VOC XML annotations under "
        f"{annotations_path} or {dataset_root_path}"
    )


def _bbox_iou_xywh(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _match_detections_at_iou50(
    predictions: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> dict[str, float]:
    preds_by_image: dict[int, list[dict[str, Any]]] = {}
    anns_by_image: dict[int, list[dict[str, Any]]] = {}

    for pred in predictions:
        preds_by_image.setdefault(int(pred["image_id"]), []).append(pred)
    for ann in annotations:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    all_image_ids = set(preds_by_image) | set(anns_by_image)
    for image_id in all_image_ids:
        preds = sorted(
            preds_by_image.get(image_id, []),
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        gts = anns_by_image.get(image_id, [])
        matched_gt_indices: set[int] = set()

        for pred in preds:
            best_iou = 0.0
            best_gt_idx = -1
            for idx, gt in enumerate(gts):
                if idx in matched_gt_indices:
                    continue
                iou = _bbox_iou_xywh(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou >= 0.5 and best_gt_idx >= 0:
                true_positives += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                false_positives += 1

        false_negatives += len(gts) - len(matched_gt_indices)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision_iou50": precision,
        "recall_iou50": recall,
        "f1_iou50": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _render_eval_visualizations(
    detection_images: list[DetectionImage],
    predictions: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    output_dir_path: Path,
    max_images: int,
) -> dict[str, Any]:
    """Save a small sample of evaluation images with GT and predicted boxes."""
    from PIL import Image as PILImage
    from PIL import ImageDraw
    from PIL import ImageFont

    if max_images <= 0:
        return {
            "visualizations_dir": None,
            "num_visualizations": 0,
            "files": [],
        }

    preds_by_image: dict[int, list[dict[str, Any]]] = {}
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for pred in predictions:
        preds_by_image.setdefault(int(pred["image_id"]), []).append(pred)
    for ann in annotations:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    vis_dir = output_dir_path / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    saved_files: list[str] = []
    for item in detection_images[:max_images]:
        image = PILImage.open(item.image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        gt_boxes = anns_by_image.get(item.image_id, [])
        pred_boxes = sorted(
            preds_by_image.get(item.image_id, []),
            key=lambda pred: float(pred.get("score", 0.0)),
            reverse=True,
        )

        for ann in gt_boxes:
            x, y, w, h = ann["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline="#00ff00", width=3)

        for pred in pred_boxes:
            x, y, w, h = pred["bbox"]
            score = float(pred.get("score", 0.0))
            draw.rectangle((x, y, x + w, y + h), outline="#ff3b30", width=2)
            label = f"pred {score:.2f}"
            text_y = max(0.0, y - 12.0)
            draw.text((x, text_y), label, fill="#ff3b30", font=font)

        legend = (
            f"GT={len(gt_boxes)} green | "
            f"Pred={len(pred_boxes)} red | "
            f"image_id={item.image_id}"
        )
        draw.text((8, 8), legend, fill="#ffffff", font=font)

        safe_stem = f"{item.image_id}_{Path(item.file_name).stem}"
        out_path = vis_dir / f"{safe_stem}.png"
        image.save(out_path)
        saved_files.append(str(out_path))

    return {
        "visualizations_dir": str(vis_dir),
        "num_visualizations": len(saved_files),
        "files": saved_files,
    }


def _select_detection_subset(
    items: list[DetectionImage],
    dataset_fraction: float,
    sample_seed: int | None,
) -> list[DetectionImage]:
    if dataset_fraction >= 1.0 or not items:
        return items

    subset_size = max(1, int(len(items) * dataset_fraction))
    if sample_seed is None:
        return items[:subset_size]

    rng = random.Random(sample_seed)
    selected_indices = sorted(rng.sample(range(len(items)), k=subset_size))
    return [items[idx] for idx in selected_indices]


def _resolve_model_input(
    file_or_url: str,
    download_dir: Path,
    fallback_name: str,
) -> Path:
    """Resolve a local path or download a URL into *download_dir*."""
    parsed = urllib.parse.urlparse(file_or_url)
    if parsed.scheme in {"http", "https"}:
        download_dir.mkdir(parents=True, exist_ok=True)
        raw_name = Path(parsed.path).name or fallback_name
        filename = raw_name if "." in raw_name else fallback_name
        local_path = download_dir / filename
        if not local_path.exists():
            print(f"Downloading {file_or_url} -> {local_path}")
            urllib.request.urlretrieve(file_or_url, local_path)
        else:
            print(f"Reusing downloaded file {local_path}")
        return local_path

    local_path = Path(file_or_url)
    if not local_path.exists():
        raise FileNotFoundError(f"Model file not found: {local_path}")
    return local_path


@app.function(
    gpu="T4",
    image=DETECTRON2_TABLEBANK_IMAGE,
    volumes={MODAL_DATA_DIR: tablebank_vol, MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=3600 * 24,
)
def run_tablebank_detectron2_eval(
    config: str,
    weights: str,
    dataset_root: str,
    annotations_path: str,
    output_dir: str,
    score_threshold: float = 0.9,
    dataset_fraction: float = 1.0,
    sample_seed: int | None = None,
    visualize_max_images: int = 20,
) -> dict[str, Any]:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    import numpy as np
    from PIL import Image as PILImage
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    tablebank_vol.reload()
    artifacts_vol.reload()

    dataset_root_path = Path(dataset_root)
    annotations_root_path = Path(annotations_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    downloads_dir = output_dir_path / "_downloads"
    resolved_config_path = _resolve_model_input(config, downloads_dir, "config.yaml")
    resolved_weights_path = _resolve_model_input(weights, downloads_dir, "model_final.pth")

    resolved_annotations_input, annotations_format = _resolve_annotations_source(
        annotations_root_path,
        dataset_root_path,
    )
    if dataset_fraction <= 0.0 or dataset_fraction > 1.0:
        raise ValueError("dataset_fraction must be in the range (0.0, 1.0].")
    if annotations_format == "voc":
        resolved_annotations_path = output_dir_path / "tablebank_annotations.coco.json"
        convert_voc_to_coco(
            resolved_annotations_input,
            resolved_annotations_path,
            single_category_name="table",
        )
    else:
        resolved_annotations_path = resolved_annotations_input

    coco_dataset = COCODataset.from_json(resolved_annotations_path)

    image_lookup: dict[str, Path] = {}
    image_root = dataset_root_path / "images"
    if image_root.exists():
        for path in image_root.rglob("*"):
            if path.is_file():
                image_lookup[path.name] = path
    else:
        for path in dataset_root_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                image_lookup[path.name] = path

    detection_images: list[DetectionImage] = []
    missing_files: list[str] = []
    for image_entry in coco_dataset.images:
        image_path = image_lookup.get(Path(image_entry.file_name).name)
        if image_path is None:
            missing_files.append(image_entry.file_name)
            continue
        detection_images.append(
            DetectionImage(
                image_id=image_entry.id,
                file_name=image_entry.file_name,
                width=image_entry.width,
                height=image_entry.height,
                image_path=image_path,
            )
        )

    if missing_files:
        preview = ", ".join(missing_files[:10])
        raise FileNotFoundError(
            f"Could not locate {len(missing_files)} image(s) under {dataset_root_path}. "
            f"First missing entries: {preview}"
        )

    detection_images = _select_detection_subset(
        detection_images,
        dataset_fraction=dataset_fraction,
        sample_seed=sample_seed,
    )
    selected_image_ids = {item.image_id for item in detection_images}
    selected_annotations = [
        annotation.model_dump(mode="json")
        for annotation in coco_dataset.annotations
        if annotation.image_id in selected_image_ids
    ]
    selected_images = [
        image.model_dump()
        for image in coco_dataset.images
        if image.id in selected_image_ids
    ]
    selected_categories = [category.model_dump() for category in coco_dataset.categories]

    cfg = get_cfg()
    cfg.merge_from_file(str(resolved_config_path))
    cfg.MODEL.WEIGHTS = str(resolved_weights_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.DEVICE = "cuda"
    predictor = DefaultPredictor(cfg)

    predictions: list[dict[str, Any]] = []
    next_prediction_id = 0
    for idx, item in enumerate(detection_images, start=1):
        image_rgb = PILImage.open(item.image_path).convert("RGB")
        image_bgr = np.asarray(image_rgb)[:, :, ::-1].copy()

        outputs = predictor(image_bgr)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.tolist() if instances.has("pred_boxes") else []
        scores = instances.scores.tolist() if instances.has("scores") else []

        for box_xyxy, score in zip(boxes, scores):
            x1, y1, x2, y2 = box_xyxy
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w < 1.0 or h < 1.0:
                continue
            predictions.append(
                {
                    "id": next_prediction_id,
                    "image_id": item.image_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )
            next_prediction_id += 1

        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(detection_images)} images")

    predictions_path = output_dir_path / "predictions.coco.json"
    predictions_path.write_text(json.dumps(predictions, indent=2, default=_json_default))

    gt_payload = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": selected_categories,
        "info": {"description": "TableBank evaluation ground truth"},
    }
    gt_path = output_dir_path / "ground_truth.coco.json"
    gt_path.write_text(json.dumps(gt_payload, indent=2))

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(predictions_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    prf_metrics = _match_detections_at_iou50(
        predictions=predictions,
        annotations=selected_annotations,
    )
    visualization_summary = _render_eval_visualizations(
        detection_images=detection_images,
        predictions=predictions,
        annotations=selected_annotations,
        output_dir_path=output_dir_path,
        max_images=visualize_max_images,
    )

    summary = {
        "baseline_name": "TableBank Detectron2 R101 FPN 3x",
        "config_path": str(resolved_config_path),
        "weights_path": str(resolved_weights_path),
        "dataset_root": str(dataset_root_path),
        "annotations_path": str(resolved_annotations_path),
        "num_images": len(detection_images),
        "num_predictions": len(predictions),
        "score_threshold": score_threshold,
        "dataset_fraction": dataset_fraction,
        "sample_seed": sample_seed,
        "metrics": {
            "map": float(coco_eval.stats[0]),
            "map50": float(coco_eval.stats[1]),
            "map75": float(coco_eval.stats[2]),
            "precision_iou50": float(prf_metrics["precision_iou50"]),
            "recall_iou50": float(prf_metrics["recall_iou50"]),
            "f1_iou50": float(prf_metrics["f1_iou50"]),
            "true_positives": int(prf_metrics["true_positives"]),
            "false_positives": int(prf_metrics["false_positives"]),
            "false_negatives": int(prf_metrics["false_negatives"]),
        },
        "artifacts": {
            "predictions_coco_json": str(predictions_path),
            "ground_truth_coco_json": str(gt_path),
            "visualizations_dir": visualization_summary["visualizations_dir"],
        },
        "visualizations": visualization_summary,
    }
    metrics_path = output_dir_path / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, default=_json_default))

    artifacts_vol.commit()
    return summary


@app.local_entrypoint()
def main(
    config: str = TABLEBANK_R101_CONFIG_URL,
    weights: str = TABLEBANK_R101_WEIGHTS_URL,
    dataset_root: str = "/data/tablebank/extracted/TableBank/TableBank/Detection",
    annotations: str = "/data/tablebank/extracted/TableBank/TableBank/Detection/annotations",
    output_dir: str = "/artifacts/tablebank_detectron2_eval",
    score_threshold: float = 0.9,
    dataset_fraction: float = 1.0,
    sample_seed: int | None = None,
    visualize_max_images: int = 20,
):
    result = run_tablebank_detectron2_eval.remote(
        config=config,
        weights=weights,
        dataset_root=dataset_root,
        annotations_path=annotations,
        output_dir=output_dir,
        score_threshold=score_threshold,
        dataset_fraction=dataset_fraction,
        sample_seed=sample_seed,
        visualize_max_images=visualize_max_images,
    )
    print(json.dumps(result, indent=2, default=_json_default))
