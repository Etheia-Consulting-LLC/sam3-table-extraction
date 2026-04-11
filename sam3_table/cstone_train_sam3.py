from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from sam3_table.coco_schema import COCODataset
from sam3_table.training_config import SAM3LoRAConfig

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("triton")
    .add_local_python_source("sam3_table")
)

app = modal.App(name="training-sam3", image = image)

data_vol = modal.Volume.from_name("pubtables-vol")
artifacts_vol = modal.Volume.from_name("artifacts-vol", create_if_missing=True)

MODAL_DATA_DIR = "/data"
MODAL_ARTIFACTS_DIR = "/artifacts"


def _config_fingerprint(config_dict: dict) -> str:
    """Deterministic hash of a config dict, ignoring output paths."""
    d = copy.deepcopy(config_dict)
    d.pop("output", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


CHECKPOINT_NAMES = ("checkpoint_epoch.pt", "checkpoint_best.pt", "checkpoint_signal.pt")


def _has_any_checkpoint(run_dir: Path) -> bool:
    """Return True if *run_dir* contains at least one resumable checkpoint."""
    return any((run_dir / name).exists() for name in CHECKPOINT_NAMES)


def _find_resumable_run(config_dict: dict) -> Path | None:
    """Scan artifacts-vol for an unfinished run whose config matches *config_dict*.

    Returns the run directory if any checkpoint file is found with a matching
    config fingerprint, otherwise ``None``.  When multiple matches exist the
    most recently modified checkpoint wins.
    """
    target_fp = _config_fingerprint(config_dict)
    artifacts_root = Path(MODAL_ARTIFACTS_DIR)
    if not artifacts_root.exists():
        return None

    matched_dirs: dict[Path, float] = {}
    for ckpt_name in CHECKPOINT_NAMES:
        for ckpt in artifacts_root.rglob(ckpt_name):
            run_dir = ckpt.parent
            mtime = ckpt.stat().st_mtime
            if run_dir in matched_dirs:
                matched_dirs[run_dir] = max(matched_dirs[run_dir], mtime)
                continue
            run_config_path = run_dir / "run_config.json"
            if not run_config_path.exists():
                continue
            try:
                existing = json.loads(run_config_path.read_text())
                if _config_fingerprint(existing) == target_fp:
                    matched_dirs[run_dir] = mtime
            except (json.JSONDecodeError, KeyError):
                continue
    candidates = [(mtime, d) for d, mtime in matched_dirs.items()]

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


@app.function(
    gpu="H200",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODAL_DATA_DIR: data_vol, MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=3600*24,
)
def train_sam3(
    config_dict: dict,
    train_coco_dataset: COCODataset | None = None,
    val_coco_dataset: COCODataset | None = None,
    test_coco_dataset: COCODataset | None = None,
    device: list[int] | None = None,
    fresh_run: bool = False,
) -> dict[str, str]:

    from sam3_table.train_sam3_lora_native import SAM3TrainerNative

    config = SAM3LoRAConfig.model_validate(config_dict)

    artifacts_vol.reload()
    resumable_dir = None if fresh_run else _find_resumable_run(config_dict)

    if resumable_dir is not None:
        timestamp = resumable_dir.name
        config.output.output_dir = str(resumable_dir)
        config.output.logging_dir = f"{MODAL_ARTIFACTS_DIR}/{config.output.logging_dir}/{timestamp}"
        print(f"Auto-resuming interrupted run {timestamp} from {resumable_dir}")
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        config.output.output_dir = (
            f"{MODAL_ARTIFACTS_DIR}/{config.output.output_dir}/{timestamp}"
        )
        config.output.logging_dir = f"{MODAL_ARTIFACTS_DIR}/{config.output.logging_dir}/{timestamp}"

    out_dir = Path(config.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2)
    )

    if train_coco_dataset is None:
        print("Loading training annotations from pubtables-vol...")
        train_ann = Path(config.training.data.train.annotation_file)
        train_coco_dataset = COCODataset.from_json(train_ann)

    if val_coco_dataset is None and config.training.data.valid is not None:
        print("Loading validation annotations from pubtables-vol...")
        val_ann = Path(config.training.data.valid.annotation_file)
        if val_ann.exists():
            val_coco_dataset = COCODataset.from_json(val_ann)

    if device is None:
        device = [0]

    multi_gpu = len(device) > 1 and "LOCAL_RANK" in os.environ

    if not multi_gpu and len(device) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0])
        print(f"Using single GPU: {device[0]}")

    trainer = SAM3TrainerNative(
        config,
        train_coco_dataset=train_coco_dataset,
        val_coco_dataset=val_coco_dataset,
        test_coco_dataset=test_coco_dataset,
        multi_gpu=multi_gpu,
        on_checkpoint=lambda: artifacts_vol.commit(),
    )
    trainer.train()

    artifacts_vol.commit()
    result = {
        "timestamp": timestamp,
        "output_dir": config.output.output_dir,
        "logging_dir": config.output.logging_dir,
    }
    print(f"Training artifacts committed to 'artifacts-vol' at {config.output.output_dir}.")
    return result


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=3600 * 24,
)
def run_sweep(configs: list[dict], fresh_run: bool = False) -> list[dict]:
    """Orchestrate a sweep entirely in the cloud (CPU-only, no GPU).

    Spawns all training runs in parallel, waits for results server-side,
    and commits a summary to artifacts-vol.  Each run automatically resumes
    from a checkpoint if one exists for its config unless *fresh_run* is True.
    """
    handles = []
    for i, config in enumerate(configs):
        handle = train_sam3.spawn(config, fresh_run=fresh_run)
        handles.append((i, config, handle))
        lora = config["lora"]
        lr = config["training"]["learning_rate"]
        print(f"Spawned run {i}: rank={lora['rank']} alpha={lora['alpha']} lr={lr}")

    print(f"\nAll {len(handles)} runs spawned. Waiting for results...\n")

    results = []
    for i, config, handle in handles:
        lora = config["lora"]
        lr = config["training"]["learning_rate"]
        try:
            result = handle.get()
        except Exception as exc:
            print(f"--- Run {i} FAILED (rank={lora['rank']} alpha={lora['alpha']} lr={lr}) ---")
            print(f"  Error: {exc}")
            results.append({"error": str(exc), "run_index": i})
            continue
        print(f"--- Run {i} (rank={lora['rank']} alpha={lora['alpha']} lr={lr}) ---")
        print(f"  Timestamp:  {result['timestamp']}")
        print(f"  Output dir: {result['output_dir']}")
        results.append(result)

    artifacts_vol.commit()
    return results


def _resolve_run_dir(timestamp: str) -> Path:
    artifacts_root = Path(MODAL_ARTIFACTS_DIR)
    candidates = sorted(
        path
        for path in artifacts_root.rglob(timestamp)
        if path.is_dir()
        and (
            (path / "best_lora_weights.pt").exists()
            or (path / "last_lora_weights.pt").exists()
            or _has_any_checkpoint(path)
        )
    )
    if not candidates:
        raise FileNotFoundError(
            f"No artifact directory found for timestamp '{timestamp}' under {artifacts_root}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Found multiple artifact directories for timestamp '{timestamp}': "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def _jsonify_segmentation(segmentation: dict[str, Any]) -> dict[str, Any]:
    counts = segmentation.get("counts")
    if isinstance(counts, bytes):
        segmentation = dict(segmentation)
        segmentation["counts"] = counts.decode("utf-8")
    return segmentation


@app.function(
    gpu="H200",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=3600,
)
def infer_sam3(
    timestamp: str,
    image_bytes_list: list[bytes],
    image_names: list[str] | None = None,
    query_text: str = "table",
    score_threshold: float = 0.25,
) -> dict[str, Any]:
    import io

    import torch
    from PIL import Image as PILImage
    from torch.utils.data import Dataset
    from torchvision.transforms import v2

    from sam3.model.model_misc import SAM3Output
    from sam3.model_builder import build_sam3_image_model
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image,
        InferenceMetadata,
    )

    from sam3_table.lora_layers import (
        LoRAConfig as LoRALayerConfig,
        apply_lora_to_model,
        load_lora_weights,
    )
    from sam3_table.train_sam3_lora_native import (
        convert_predictions_to_coco_format_original_res,
        resolve_bpe_vocab_path,
    )

    if not image_bytes_list:
        raise ValueError("image_bytes_list must contain at least one image")

    if image_names is None:
        image_names = [f"image_{idx}.png" for idx in range(len(image_bytes_list))]
    elif len(image_names) != len(image_bytes_list):
        raise ValueError("image_names must have the same length as image_bytes_list")

    artifacts_vol.reload()
    run_dir = _resolve_run_dir(timestamp)

    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = SAM3LoRAConfig.model_validate(json.loads(config_path.read_text()))
    else:
        fallback_config = Path(__file__).resolve().parent / "testSamples" / "full_lora_config.yaml"
        config = SAM3LoRAConfig.from_yaml(fallback_config)

    weights_path = run_dir / "best_lora_weights.pt"
    if not weights_path.exists():
        weights_path = run_dir / "last_lora_weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"No LoRA weights found in {run_dir}")

    images = [PILImage.open(io.BytesIO(blob)).convert("RGB") for blob in image_bytes_list]
    normalized_query = query_text.strip().lower() or "table"
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _InferenceDataset(Dataset):
        def __init__(self, pil_images: list[PILImage.Image], names: list[str], query: str):
            self.images = pil_images
            self.names = names
            self.query = query
            self.resolution = 1008
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, idx: int) -> Datapoint:
            original = self.images[idx]
            orig_w, orig_h = original.size
            resized = original.resize((self.resolution, self.resolution), PILImage.BILINEAR)
            image_tensor = self.transform(resized)
            image_obj = Image(data=image_tensor, objects=[], size=(self.resolution, self.resolution))
            query = FindQueryLoaded(
                query_text=self.query,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=idx,
                    original_image_id=idx,
                    original_category_id=0,
                    original_size=(orig_h, orig_w),
                    object_id=-1,
                    frame_index=-1,
                ),
            )
            return Datapoint(find_queries=[query], images=[image_obj], raw_images=[resized])

    def move_to_device(obj: Any, device: torch.device) -> Any:
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, list):
            return [move_to_device(item, device) for item in obj]
        if isinstance(obj, tuple):
            return tuple(move_to_device(item, device) for item in obj)
        if isinstance(obj, dict):
            return {key: move_to_device(value, device) for key, value in obj.items()}
        if hasattr(obj, "__dataclass_fields__"):
            for field in obj.__dataclass_fields__:
                setattr(obj, field, move_to_device(getattr(obj, field), device))
            return obj
        return obj

    dataset = _InferenceDataset(images, image_names, normalized_query)
    batch = collate_fn_api(
        [dataset[idx] for idx in range(len(dataset))],
        dict_key="input",
        with_seg_masks=True,
    )

    model = build_sam3_image_model(
        device=device_obj.type,
        compile=False,
        load_from_HF=True,
        bpe_path=resolve_bpe_vocab_path(),
        eval_mode=True,
    )
    lora_cfg = config.lora
    model = apply_lora_to_model(
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
    load_lora_weights(model, str(weights_path))
    model.to(device_obj)
    model.eval()

    input_batch = move_to_device(batch["input"], device_obj)
    with torch.no_grad():
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
        image_ids=list(range(len(dataset))),
        dataset=dataset,
        score_threshold=score_threshold,
    )

    predictions_by_image: dict[int, list[dict[str, Any]]] = {
        idx: [] for idx in range(len(image_names))
    }
    for prediction in coco_predictions:
        cleaned_prediction = dict(prediction)
        cleaned_prediction["segmentation"] = _jsonify_segmentation(
            cleaned_prediction["segmentation"]
        )
        predictions_by_image[cleaned_prediction["image_id"]].append(cleaned_prediction)

    return {
        "timestamp": timestamp,
        "run_dir": str(run_dir),
        "weights_path": str(weights_path),
        "query_text": normalized_query,
        "results": [
            {
                "image_id": idx,
                "image_name": image_names[idx],
                "predictions": predictions_by_image[idx],
            }
            for idx in range(len(image_names))
        ],
    }
